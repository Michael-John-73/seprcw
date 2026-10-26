"""Run stages from the project directory; finalization is explicit and one-shot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="ISCIPT prospective H1-H3 crop-recovery experiments")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--out", default="outputs/pilot")
    parser.add_argument("stage", choices=["prepare", "develop", "calibrate", "finalize", "status"])
    args = parser.parse_args()
    from iscipt import experiment
    cfg, out = Path(args.config), Path(args.out)
    cfg = cfg if cfg.is_absolute() else ROOT / cfg
    out = out if out.is_absolute() else ROOT / out
    try:
        if args.stage == "status":
            result = experiment.status(out)
        else:
            result = getattr(experiment, args.stage)(cfg, out)
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError) as exc:
        print(json.dumps({"status": "ERROR", "stage": args.stage, "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
