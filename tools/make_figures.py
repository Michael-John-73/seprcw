"""Generate descriptive H-Q recovery and interval figures from released records."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    print("[START] Generate figures from released numerical data", flush=True)
    data = json.loads((ROOT / "data" / "hq" / "analysis.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    rates = [100 * data["primary_recovery"][m]["ER"] for m in ("E1", "E2")]
    axes[0].bar(["E1D0, delta 36", "E2D0, delta 16"], rates, color=["#d87932", "#006699"])
    axes[0].set(ylabel="Exact recovery (%)", ylim=(0, 105), title="848-source blind confirmation")
    effect = data["effect"]
    axes[1].errorbar(100 * effect["difference"], 0,
                     xerr=[[100 * (effect["difference"] - effect["low"])],
                           [100 * (effect["high"] - effect["difference"])]],
                     fmt="o", capsize=5, color="#006699")
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].set(xlabel="E2 minus E1 (percentage points)", yticks=[],
                title="Nominal source-paired 95% interval", xlim=(-2, 50))
    for extension in ("png", "pdf"):
        fig.savefig(args.output / ("hq_confirmation." + extension), dpi=180)
    plt.close(fig)
    print("[END] Figure files saved to " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
