"""Opt-in adapter that reuses the ICISC payload, crop rule and delta-36 raster insertions.

Without ``payload_source``, ``evaluation.crop_rule`` and ``reuse`` in the configuration the earlier
behaviour is unchanged: seeded random payload, runner crop rule, a fresh embedding for every method.

Access boundary:
* prepare time reads JSON/CSV bytes and file stats only. No image bytes are opened, so calibration
  and test pixels stay unread until their own stage;
* a legacy marked image is read only while its source is measured by the stage that owns the split,
  and its SHA-256 is recorded in the raw rows;
* legacy outcome files (shard CSV/NPY) are never opened here. Every measurement is a new decode.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

LEGACY_CROP_RULE = "runner_floor_sample_hash"
ICISC_CROP_RULE = "icisc_filename_sha256_round"
CROP_RULES = (LEGACY_CROP_RULE, ICISC_CROP_RULE)
PAYLOAD_KINDS = ("legacy_xl",)
PAYLOAD_JSON_KEY = "p_xl"
PAYLOAD_KEY = "xl"
LEGACY_PAYLOAD_BITS = 48
ON_INCOMPATIBLE = "recompute_and_record"

CARRIER_ICISC = "icisc_prepared_carrier"
CARRIER_CONFIGURED = "configured_data_root"
EMBED_NEW = "new_embed"
EMBED_LEGACY = "legacy_xl_marked_image"
EMBED_UNMARKED = "none_unmarked_carrier"
EMBED_NONE = "not_embedded"
MEASUREMENT_ORIGIN = "new_decode"

CACHE_HIT = "CACHE_HIT"
CANDIDATE = "CACHE_CANDIDATE"
DISABLED = "DISABLED"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_ATTEMPTED = "NOT_ATTEMPTED"
RECOMPUTED = "RECOMPUTED_INCOMPATIBLE"

CROP_MANIFEST_COLUMNS = ("domain", "work_file", "group_id", "area", "out_h", "out_w", "crop_h", "crop_w", "top",
                         "left", "seed_string", "legacy_artifact_id", "crop_rule", "resize_after_crop")
REUSE_MANIFEST_COLUMNS = ("artifact_id", "domain", "work_file", "group_id", "carrier_path", "legacy_marked_path",
                          "payload_key", "L", "attack", "delta", "verification_status")


def sha256_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_path(first, second) -> bool:
    return os.path.normcase(os.path.realpath(str(first))) == os.path.normcase(os.path.realpath(str(second)))


def crop_rule(cfg) -> str:
    return cfg["evaluation"].get("crop_rule", LEGACY_CROP_RULE)


def reuse_enabled(cfg) -> bool:
    return bool((cfg.get("reuse") or {}).get("enabled", False))


def _absolute(value, name):
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")


def validate_config(cfg):
    """Check the optional keys of this adapter; a configuration without them is accepted unchanged."""
    evaluation = cfg["evaluation"]
    rule = evaluation.get("crop_rule", LEGACY_CROP_RULE)
    if rule not in CROP_RULES:
        raise ValueError("evaluation.crop_rule must be one of " + ", ".join(CROP_RULES))
    source = cfg.get("payload_source")
    if source is not None:
        if not isinstance(source, dict) or source.get("kind") not in PAYLOAD_KINDS:
            raise ValueError("payload_source.kind must be one of " + ", ".join(PAYLOAD_KINDS))
        _absolute(source.get("path"), "payload_source.path")
        if int(cfg["codec"]["payload_bits"]) != LEGACY_PAYLOAD_BITS:
            raise ValueError(f"payload_source legacy_xl requires codec.payload_bits={LEGACY_PAYLOAD_BITS}")
    reuse = cfg.get("reuse")
    if reuse is None:
        return
    if not isinstance(reuse, dict) or not isinstance(reuse.get("enabled"), bool):
        raise ValueError("reuse.enabled must be a boolean")
    if not reuse["enabled"]:
        return
    _absolute(reuse.get("manifest"), "reuse.manifest")
    _absolute(reuse.get("crop_manifest"), "reuse.crop_manifest")
    allowed = reuse.get("allow_embed_cache_delta")
    if (not isinstance(allowed, list)
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in allowed)
            or len({float(x) for x in allowed}) != len(allowed)):
        raise ValueError("reuse.allow_embed_cache_delta must be a list of unique positive numbers (empty disables the cache)")
    if reuse.get("on_incompatible") != ON_INCOMPATIBLE:
        raise ValueError(f"reuse.on_incompatible must be {ON_INCOMPATIBLE}")
    if rule != ICISC_CROP_RULE:
        raise ValueError(f"reuse.crop_manifest freezes ICISC crop boxes; set evaluation.crop_rule={ICISC_CROP_RULE}")
    if int(evaluation.get("image_maxside", 0)) != 0:
        raise ValueError("reuse requires evaluation.image_maxside=0 so that the prepared resolution is preserved")
    if allowed and source is None:
        raise ValueError("a legacy marked-image cache requires payload_source.kind=legacy_xl")


def load_payload_source(cfg):
    """Read p_xl.bits of the ICISC payload file; None keeps the seeded random payload."""
    source = cfg.get("payload_source")
    if source is None:
        return None
    path = Path(source["path"])
    data = path.read_bytes()
    try:
        entry = json.loads(data.decode("utf-8"))[PAYLOAD_JSON_KEY]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: {PAYLOAD_JSON_KEY} could not be read ({type(exc).__name__}: {exc})") from exc
    bits = entry.get("bits") if isinstance(entry, dict) else None
    if not isinstance(bits, str) or len(bits) != LEGACY_PAYLOAD_BITS or set(bits) - {"0", "1"}:
        raise ValueError(f"{path}: {PAYLOAD_JSON_KEY}.bits must be exactly {LEGACY_PAYLOAD_BITS} binary digits")
    if entry.get("L", LEGACY_PAYLOAD_BITS) != LEGACY_PAYLOAD_BITS:
        raise ValueError(f"{path}: {PAYLOAD_JSON_KEY}.L must be {LEGACY_PAYLOAD_BITS}")
    return {"kind": source["kind"], "path": str(path.resolve()), "json_key": PAYLOAD_JSON_KEY,
            "file_sha256": sha256_bytes(data), "bits": bits, "bits_list": [int(c) for c in bits],
            "bits_sha256": sha256_bytes(bits.encode("ascii")), "length": LEGACY_PAYLOAD_BITS}


def payload_record(source, payload) -> dict:
    """State record of the payload origin and its hashes."""
    bits = "".join(str(int(b)) for b in payload)
    record = {"bits": bits, "bits_sha256": sha256_bytes(bits.encode("ascii")), "length": len(bits)}
    if source is None:
        return dict(record, kind="seeded_random",
                    derivation="numpy default_rng seeded by _seed(config.seed, 'target-payload')")
    return dict(record, kind=source["kind"], path=source["path"], json_key=source["json_key"],
                file_sha256=source["file_sha256"])


def icisc_crop_box(height, width, area, name) -> dict:
    """Crop box of ICISC scripts_conf/attacks.crop, computed without pixels.

    The seed is SHA-256 of "<work_file basename>|crop<area>" and the box size is rounded, not floored.
    Area 1.0 is the identity (the legacy "none" attack); no resize follows a crop.
    """
    height, width, area = int(height), int(width), float(area)
    if height < 1 or width < 1 or not 0 < area <= 1:
        raise ValueError("a crop box needs a positive size and 0 < area <= 1")
    if area == 1.0:
        return {"top": 0, "left": 0, "crop_h": height, "crop_w": width, "seed_string": "", "seed": None}
    scale = float(np.sqrt(area))
    crop_h, crop_w = max(1, round(height * scale)), max(1, round(width * scale))
    seed_string = f"{name}|crop{area}"
    seed = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, max(1, height - crop_h + 1)))
    left = int(rng.integers(0, max(1, width - crop_w + 1)))
    return {"top": top, "left": left, "crop_h": crop_h, "crop_w": crop_w, "seed_string": seed_string, "seed": seed}


def _read_csv(path, required):
    data = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    missing = [name for name in required if name not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return list(reader), sha256_bytes(data)


def load_crop_manifest(path):
    """Read the frozen crop manifest and verify every row against the ICISC rule (metadata only)."""
    rows, digest = _read_csv(path, CROP_MANIFEST_COLUMNS)
    table = {}
    for line, row in enumerate(rows, 2):
        try:
            area, out_h, out_w = float(row["area"]), int(row["out_h"]), int(row["out_w"])
            stored = (int(row["top"]), int(row["left"]), int(row["crop_h"]), int(row["crop_w"]))
        except ValueError as exc:
            raise ValueError(f"{path} line {line}: malformed numeric field") from exc
        if row["crop_rule"] != ICISC_CROP_RULE or row["resize_after_crop"] != "false":
            raise ValueError(f"{path} line {line}: not an {ICISC_CROP_RULE} row without resize")
        box = icisc_crop_box(out_h, out_w, area, row["work_file"])
        if stored != (box["top"], box["left"], box["crop_h"], box["crop_w"]) or row["seed_string"] != box["seed_string"]:
            raise ValueError(f"{path} line {line}: box differs from the ICISC rule for {row['domain']}/{row['work_file']} area {area}")
        key = (row["domain"], row["work_file"], area)
        if key in table:
            raise ValueError(f"{path} line {line}: duplicate row {key}")
        table[key] = {"group_id": row["group_id"], "out_h": out_h, "out_w": out_w, "top": stored[0], "left": stored[1],
                      "crop_h": stored[2], "crop_w": stored[3], "legacy_artifact_id": row["legacy_artifact_id"]}
    return table, {"path": str(Path(path).resolve()), "sha256": digest, "rows": len(rows)}


def load_reuse_manifest(path):
    """Read the R00 link manifest. Its legacy outcome columns (CSV/NPY paths, rows, decodable) are not kept."""
    rows, digest = _read_csv(path, REUSE_MANIFEST_COLUMNS)
    table = {}
    for row in rows:
        kept = {name: row[name] for name in REUSE_MANIFEST_COLUMNS}
        table.setdefault((kept["domain"], kept["work_file"]), []).append(kept)
    return table, {"path": str(Path(path).resolve()), "sha256": digest, "rows": len(rows)}


def link_sources(cfg, sources, payload_source=None):
    """Freeze the link record of every committed source: carrier, prepared size, crop boxes, marked-image stat.

    Structural problems (unlinked source, carrier or identity mismatch, missing crop row) stop the prepare.
    A marked image that is absent only disables the cache for that source and is recorded.
    """
    reuse = cfg["reuse"]
    areas = [float(a) for a in cfg["evaluation"]["retained_area"]]
    crops, crop_pin = load_crop_manifest(reuse["crop_manifest"])
    links, manifest_pin = load_reuse_manifest(reuse["manifest"])
    linked, present, missing = {}, 0, 0
    for source in sources:
        domain, name = source["domain"], Path(source["path"]).name
        rows = links.get((domain, name))
        if not rows:
            raise ValueError(f"reuse manifest has no row for {domain}/{name}")
        for row in rows:
            if row["verification_status"] != "LINKED":
                raise ValueError(f"reuse manifest row {row['artifact_id']} for {domain}/{name} is {row['verification_status']}, not LINKED")
            if row["group_id"] != source["group_id"]:
                raise ValueError(f"reuse manifest group_id differs from the committed source for {domain}/{name}")
            if not _same_path(row["carrier_path"], source["path"]):
                raise ValueError(f"reuse manifest carrier_path differs from the committed carrier for {domain}/{name}")
        facts = {(row["legacy_marked_path"], row["payload_key"], row["L"], row["delta"]) for row in rows}
        if len(facts) != 1:
            raise ValueError(f"reuse manifest rows disagree on marked image, payload or delta for {domain}/{name}")
        marked_path, payload_key, length, delta = next(iter(facts))
        try:
            length, delta = int(length), float(delta)
        except ValueError as exc:
            raise ValueError(f"reuse manifest has a malformed L or delta for {domain}/{name}") from exc
        crop, shapes = {}, set()
        for area in areas:
            row = crops.get((domain, name, area))
            if row is None:
                raise ValueError(f"crop manifest has no row for {domain}/{name} area {area}")
            if row["group_id"] != source["group_id"]:
                raise ValueError(f"crop manifest group_id differs from the committed source for {domain}/{name}")
            shapes.add((row["out_h"], row["out_w"]))
            crop[repr(area)] = {key: row[key] for key in ("legacy_artifact_id", "top", "left", "crop_h", "crop_w")}
        if len(shapes) != 1:
            raise ValueError(f"crop manifest rows disagree on the prepared size for {domain}/{name}")
        marked = {"path": marked_path, "status": "MISSING"}
        if Path(marked_path).is_file():
            stat = Path(marked_path).stat()
            marked.update(status="PRESENT", file_size=stat.st_size, file_mtime_ns=stat.st_mtime_ns)
            present += 1
        else:
            missing += 1
        marked.update(delta=delta, payload_key=payload_key, payload_length=length,
                      payload_bits=payload_source["bits"] if payload_source and payload_key == PAYLOAD_KEY else None,
                      artifacts=sorted(row["artifact_id"] for row in rows))
        linked[source["sample_id"]] = {"carrier_path": source["path"], "prepared_shape": list(next(iter(shapes))),
                                       "crop": crop, "legacy_marked": marked}
    summary = {"enabled": True, "crop_rule": ICISC_CROP_RULE, "sources_linked": len(linked), "areas": areas,
               "legacy_marked_present": present, "legacy_marked_missing": missing,
               "allow_embed_cache_delta": [float(x) for x in reuse["allow_embed_cache_delta"]],
               "on_incompatible": reuse["on_incompatible"]}
    return linked, {"reuse_manifest": manifest_pin, "crop_manifest": crop_pin}, summary


def configured_inputs(cfg) -> dict:
    """External files that must stay byte-identical from prepare to finalize."""
    inputs = {}
    if cfg.get("payload_source") is not None:
        inputs["payload_source"] = cfg["payload_source"]["path"]
    if reuse_enabled(cfg):
        inputs["reuse_manifest"] = cfg["reuse"]["manifest"]
        inputs["crop_manifest"] = cfg["reuse"]["crop_manifest"]
    return inputs


def verify_pins(cfg, state):
    """Block a stage when an external payload/reuse/crop manifest differs from what prepare committed."""
    expected, pinned = configured_inputs(cfg), state.get("external_inputs", {})
    if set(expected) != set(pinned):
        raise ValueError("External reuse inputs differ from those committed at prepare; use a new experiment directory.")
    for name, path in expected.items():
        pin = pinned[name]
        try:
            current = sha256_file(pin["path"])
        except OSError as exc:
            raise ValueError(f"External input unreadable after prepare: {name} ({exc})") from exc
        if not _same_path(path, pin["path"]) or current != pin["sha256"]:
            raise ValueError(f"External input changed after prepare: {name}")


def verify_marked_stat(manifest):
    """Stat-only check that committed legacy marked files are unchanged; pixels are not read."""
    for row in manifest:
        link = row.get("reuse_link")
        marked = link["legacy_marked"] if link else None
        if marked is None or marked["status"] != "PRESENT":
            continue
        try:
            stat = Path(marked["path"]).stat()
        except OSError as exc:
            raise ValueError(f"Committed legacy marked file unreadable: {row['sample_id']} ({exc})") from exc
        if stat.st_size != marked["file_size"] or stat.st_mtime_ns != marked["file_mtime_ns"]:
            raise ValueError(f"Committed legacy marked file changed: {row['sample_id']}")


def carrier_provenance(cfg, source, meta) -> dict:
    """Carrier origin and size check; ``meta`` is None when the image could not be read."""
    link = source.get("reuse_link")
    if link is None:
        size_check = "NOT_CHECKED"
    elif meta is None:
        size_check = "NOT_READ"
    else:
        size_check = "OK" if tuple(meta["processed_shape"][:2]) == tuple(link["prepared_shape"]) else "MISMATCH"
    return {"carrier_origin": CARRIER_ICISC if link else CARRIER_CONFIGURED, "crop_rule": crop_rule(cfg),
            "measurement_origin": MEASUREMENT_ORIGIN, "size_check": size_check}


def size_mismatch_error(source, meta) -> str:
    return (f"SizeCheckFailed: decoded size {list(meta['processed_shape'][:2])} differs from the prepared size "
            f"{source['reuse_link']['prepared_shape']} frozen in the crop manifest")


def crop_geometry(source, shape, area):
    """(top, left, crop_h, crop_w) under the ICISC rule: the frozen manifest box when linked, else the rule."""
    link = source.get("reuse_link")
    if link is not None:
        box = link["crop"][repr(float(area))]
        return box["top"], box["left"], box["crop_h"], box["crop_w"]
    box = icisc_crop_box(shape[0], shape[1], area, Path(source["path"]).name)
    return box["top"], box["left"], box["crop_h"], box["crop_w"]


def legacy_artifact_id(source, area, method, label) -> str:
    """Legacy result artifact of the same condition; only raster decoding of marked images has one."""
    link = source.get("reuse_link")
    if link is None or method != "raster" or int(label) != 1:
        return ""
    return link["crop"][repr(float(area))]["legacy_artifact_id"]


def unmarked_record() -> dict:
    return {"embedding_origin": EMBED_UNMARKED, "reuse_status": NOT_APPLICABLE,
            "reuse_reason": "negative control uses the unmarked carrier"}


def not_embedded_record(reason) -> dict:
    return {"embedding_origin": EMBED_NONE, "reuse_status": NOT_ATTEMPTED, "reuse_reason": str(reason)}


def embedding_plan(cfg, source, mode, codec_values, payload, meta):
    """Decide from metadata only whether the legacy marked image may replace a fresh embedding.

    Returns (status, reason). CACHE_CANDIDATE means every condition holds; the cache is confirmed
    (CACHE_HIT) only after the image has been read and matches the carrier shape.
    """
    if not reuse_enabled(cfg):
        return DISABLED, "reuse is not enabled"
    if mode != "raster":
        return NOT_APPLICABLE, "tile layout has no legacy counterpart; always embedded anew"
    link = source.get("reuse_link")
    if link is None:
        return RECOMPUTED, "source has no frozen link record"
    marked = link["legacy_marked"]
    delta = float(codec_values["delta"])
    allowed = [float(x) for x in cfg["reuse"]["allow_embed_cache_delta"]]
    bits = marked["payload_bits"]
    reasons = []
    if delta not in allowed:
        reasons.append(f"delta {delta:g} is not in reuse.allow_embed_cache_delta {allowed}")
    if delta != marked["delta"]:
        reasons.append(f"delta {delta:g} differs from the legacy marked images' delta {marked['delta']:g}")
    if marked["payload_key"] != PAYLOAD_KEY:
        reasons.append(f"legacy marked images carry payload {marked['payload_key']}, not {PAYLOAD_KEY}")
    elif bits is None:
        reasons.append("no legacy XL payload is pinned (payload_source is not configured)")
    elif "".join(str(int(b)) for b in payload) != bits:
        reasons.append("payload differs from the legacy XL payload")
    if len(payload) != marked["payload_length"]:
        reasons.append(f"payload length {len(payload)} differs from the legacy {marked['payload_length']}")
    if codec_values.get("swap_details", True) is not True:
        reasons.append("swap_details is not true as in the legacy library")
    if list(meta["processed_shape"]) != list(meta["original_shape"]):
        reasons.append("carrier was resized; the prepared resolution is not preserved")
    if tuple(meta["processed_shape"][:2]) != tuple(link["prepared_shape"]):
        reasons.append("carrier size differs from the prepared size in the manifest")
    if marked["status"] != "PRESENT":
        reasons.append("legacy marked image was missing when the sources were committed")
    return (RECOMPUTED, "; ".join(reasons)) if reasons else (CANDIDATE, "")


def read_marked(source, shape):
    """Read the legacy marked image of one source; called only while that source is being measured."""
    marked = source["reuse_link"]["legacy_marked"]
    data = Path(marked["path"]).read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("legacy marked image could not be decoded by OpenCV")
    if image.shape != tuple(shape):
        raise ValueError(f"legacy marked image shape {tuple(image.shape)} differs from the carrier shape {tuple(shape)}")
    return image, {"legacy_marked_path": marked["path"], "legacy_marked_sha256": sha256_bytes(data)}
