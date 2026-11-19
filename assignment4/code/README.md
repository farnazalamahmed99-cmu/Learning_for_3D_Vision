# 16-825 Assignment 4: Multi-View Transformers (farnazaa)

Code for all three parts. The completed TODOs are in:

| Part | Files |
|---|---|
| 1. Data preprocessing | `a4data/geometry.py` (`resize_intrinsics`, `unproject_depth`, `canonicalize_cameras`, `camera_rays`, `plucker_coordinates`) |
| 2. Mini-LVSM | `models/lvsm/model.py` (patchify, unpatchify, `build_tokens`, `forward`), `models/lvsm/losses.py`, `models/lvsm/train.py` |
| 3. Mini-VGGT | `models/vggt/model.py` (`aggregate`), `models/vggt/heads.py` (camera and depth heads), `models/vggt/losses.py` |

Bonus additions:

- **2.3 (training view count):** `models/lvsm/train.py` accepts an optional `train_num_source_views` list in the config and samples the source-view count per step. `configs/lvsm_randS.json` uses `[1, 2, 3, 4]`. Without that key, training is unchanged.
- **3.3 (number of views):** `extra/bench_views.py` measures runtime and peak GPU memory for 2–32 views. Accuracy uses the provided `models.vggt.eval` with `--num-views`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
hf download learning3dvision/ScanNetpp_iphone ScanNetpp_A4.zip --repo-type dataset --local-dir data/scannetpp
unzip data/scannetpp/ScanNetpp_A4.zip -d data/scannetpp
```

On Windows, set `PYTHONUTF8=1` so the visualization HTML is written as UTF-8.

## Running

`main.py` runs every step with the settings used in the report:

```bash
python main.py all                 # everything, in order
python main.py part1               # Q1 visualizations + check_data
python main.py lvsm_overfit        # Q2.1
python main.py lvsm_train lvsm_eval    # Q2.2 training + 9-setting ablation
python main.py lvsm_randS          # Q2.3 bonus
python main.py vggt_overfit        # Q3.1
python main.py vggt_train vggt_eval    # Q3.2 training + val/test evaluation
python main.py vggt_views          # Q3.3 bonus
```

Outputs (logs, curves, `metrics.json`, visualizations) go to `outputs/<run>/`.

Hardware used: one NVIDIA A10G (24 GB, AWS g5.xlarge). LVSM training takes about 46 min; Mini-VGGT training (1000 epochs) takes about 9 h.

The original assignment instructions are in `ASSIGNMENT_README.md` and `docs/`.
