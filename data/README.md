# Released data and original-image acquisition

## What is uploaded

The repository contains original numerical measurements and portable manifests,
not the COCO/DiffusionDB image collections. `release_manifest.json` gives the
SHA-256 of each released source/data file and the corresponding original
evidence digest where applicable.

| Directory | Unit and contents |
|---|---|
| `hq` | 848 sources, 3392 decoding rows, 1696 quality measurements; exact source filenames, raw/prepared/pixel hashes, messages, crop boxes, predicted bits, analysis, provenance locks |
| `protocol_s` | Original synchronization results, calibration/test rows, configuration, frozen selection |
| `protocol_e` | Original E analysis, seven-method test/calibration rows, thresholds and protocol records |
| `followup` | A's quality selection and validation rows; B's same-alignment oracle rows |

A method/crop row is not an independent source. The H-Q confidence interval
resamples source pairs within the two 424-source strata. S/E/A/B/H-Q must not
be combined into a single nominal confidence family.

## Obtain the exact H-Q images

- Photographs: [COCO val2017](https://cocodataset.org/#download).
  Download endpoint: `https://images.cocodataset.org/zips/val2017.zip`.
  The 424 selected filenames and byte hashes are in `hq/cohort.json`.
- Generated images: [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb),
  2M image collection, `images/part-000002.zip`.
  Pinned revision: `527749109c86eda3c82027143e45964e75cbf772`.
  Archive SHA-256:
  `cc3b691e4e84f86672ef7984d06350ee9dbc17dafd1a7f98dba5caa45ed5e786`.

Use `python tools/reproduce_hq.py prepare --download`, or supply
`--real-dir` and `--ai-dir` with already acquired original files. The tool checks
every selected raw hash, reproduces short-side cubic preparation and max-side
area resizing, and checks prepared PNG and pixel hashes. It does not choose a
new split or replace missing sources.

The historical 2000 working sources were excluded before confirmation; nine
candidate input exclusions were made before recovery outcomes. The selection
and history records remain in `hq`. Byte/pixel/dHash screening is not proof of
semantic or prompt-group independence.

## Portable exports and provenance

Local absolute paths are replaced with relative identifiers. The released files
therefore need not have the same hash as the original sealed JSON. The original
locks preserve the original hashes, while `release_manifest.json` independently
authenticates the public exports. Do not claim that an original lock authenticates
a transformed file byte-for-byte.

The raw image hashes, pixel hashes, source IDs, message assignments, predicted
bits, quality measurements, and inferential quantities are not changed by this
path transformation. Source snapshots are copied unchanged and checked against
the experiment lock. `python tools/verify_release.py` verifies release integrity
and recomputes H-Q from the uploaded bits and measurements.

## Rights and limits

Original and marked images, original prompts, IEEE template assets, and private
machine credentials are excluded. Respect the original providers' licenses,
individual COCO image rights, dataset content warnings, and attribution terms.
No blanket redistribution license is claimed. The authors' numerical data and
code currently have no selected public reuse license; see `../LICENSE_STATUS.md`.
