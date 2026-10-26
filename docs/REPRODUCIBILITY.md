# Reproducibility guide

## Levels of reproduction

1. **Released-record verification:** `python tools/verify_release.py`. No image
   downloads, training, PyTorch, or GPU required. Check file digests, all 3392
   H-Q bit vectors, source/method/crop pairing, 1696 quality records, the
   10000-draw interval, and preservation of original S/E verdicts.
2. **Diagnostic image rerun:** prepare original images, then run
   `python tools/reproduce_hq.py run --limit 4 --output outputs/smoke`.
   This is a subset check only.
3. **Full fixed-cohort H-Q rerun:** omit `--limit`. Reproduce 848 sources,
   3392 decodes, and 1696 quality measurements using the frozen CPU engine.
   It compares predicted bits, estimated phases, coverage, messages, crop boxes,
   PNG hashes, and quality with the released reference. This does not create a
   new confirmation cohort or repeat strength selection.
4. **Original S/E workflows:** the original engines and configurations are
   provided, together with stored calibration/test/analysis records. They are
   different protocols and are not replaced by the H-Q CPU entry point.

## Installation

Tested experiment environment: Python 3.13.3, NumPy 2.4.6, SciPy 1.17.1,
OpenCV 5.0.0, PyWavelets 1.8.0, scikit-image 0.26.0, Torch 2.12.0+cpu.
The OpenCV distribution/package version can differ from `cv2.__version__`;
the release validation report records installed distribution versions.

Create and activate a new virtual environment, install `requirements.txt`,
and for image reruns install the CPU PyTorch build shown in the main README.
Do not install packages globally or overwrite the original sealed experiment.
Figures additionally use Matplotlib. All commands emit START/END and progress.

## Immutable experimental choices

- E1/E2 steps: 36/16; constrained gap eta: 2; decoder D0.
- Payload 48 bits, tile side 8, 16 pilots, blind search over 4096 phases.
- H-Q has 424 photographic and 424 generated sources; primary areas 0.8/0.5.
- Pattern seed 20261008; crop seed 20261021.
- Selection/message and bootstrap seeds 20261026.
- The 1024-word message pool and exact source assignments are uploaded.
- Mean quality tolerances: absolute PSNR difference <=0.25 dB and SSIM <=0.005.
- One source-paired, stratum-preserving 10000-draw percentile bootstrap,
  weights 0.5/0.5, nominal two-sided 95% interval.
- Support requires matching mean quality and an ER-difference lower endpoint >0.
- The 3 pp half-width is a planning target, not the support criterion.

SSIM is evaluated on full BGR images with `channel_axis=2`, `data_range=255`,
scikit-image's default uniform 7x7 window, sample covariance, K1=0.01/K2=0.03.
PSNR uses full-image mean squared error and range 255. The preprocessing,
crop seed helper, and PNG path are preserved in the frozen source.

## Source/data location

`data/hq/cohort.json` lists admitted filenames and hashes. Download exact images
from the official providers as documented in `data/README.md`. The acquisition
tool selects only those files and errors on missing or mismatched inputs.
Its downloaded archives and prepared/marked images stay under ignored
`datasets/` and `outputs/`; they must not be committed.

Original machine paths are not operational dependencies. The historical runner
is preserved for provenance; use the portable `reproduce_hq.py` wrapper for
published reproduction. The original lock hashes refer to frozen originals;
the public release manifest refers to portable exports.

## Original synchronization and embedding workflows

The synchronization entry point is `run_sync.py`; the E stage entry point is
`run_embedding.py`. Their configurations are snapshots, not path-independent
one-command deployments. Adapt only local input/output locations in a **new**
run directory before starting a fresh workflow. Do not copy past PASS flags,
training checkpoints, thresholds, or seals to changed experiments.

The original E production path requires a working CUDA PyTorch environment;
the H-Q CPU entry point does not relax that requirement globally. S/E numerical
records are released for inspection and recount. Full E retraining requires
the original image intake, role assignments, model training, detector fitting,
calibration, and freeze procedure. The H-Q full CPU rerun does not constitute
verification of every original GPU training trajectory.

## Reporting differences

Do not tune steps, sources, thresholds, seeds, or the decoder to force agreement.
Report the repository commit, Python/package versions, command, source checksums,
and first differing source/key. Unexpected I/O/numerical errors fail execution;
they are never silently counted as successful recovery or replaced sources.

H-Q adds no acceptance calibration and no false-acceptance guarantee at its new
strengths. Original S-H3 stays INCONCLUSIVE and E hypotheses stay WITHHELD.
