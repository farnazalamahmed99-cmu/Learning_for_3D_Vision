# 16-825 Assignment 5: Point Cloud Classification and Segmentation (farnazaa)

Results and visualizations are on the website (`proj5/index.html`).

## What is implemented

| Question | Files |
|---|---|
| Q1. PointNet classification | `models.py` (`cls_model`), `train.py`, `eval_cls.py` |
| Q2. PointNet segmentation | `models.py` (`seg_model`), `train.py`, `eval_seg.py` |
| Q3. Robustness | `--num_points` (random subsampling) and `--rot_deg` (rotation about the x-axis, `utils.rotate_x`) in `eval_cls.py` and `eval_seg.py` |
| Q4. Locality (bonus) | `models.py` (`knn`, `edge_features`, `edgeconv`, `dgcnn_cls`, `dgcnn_seg`): DGCNN with EdgeConv, k = 20 |

Other additions:

- `train.py`: `--model {pointnet,dgcnn}` chooses the architecture. `--num_points` randomly subsamples each batch during training and testing.
- `eval_cls.py`: `--viz` saves GIFs of 5 random test objects and one failure per class (`utils.viz_cls`).
- `eval_seg.py`: prints the accuracy of the chosen object `--i`.
- `--checkpoint_dir` lets the evaluation scripts load the Q4 checkpoints.

## Setup

Use the Assignment 1 environment (`requirements.txt`). Then download the data:

```bash
hf download learning3dvision/assignment5 a5_data.zip --repo-type dataset --local-dir data_download
unzip data_download/a5_data.zip -d ./
```

## Running

`main.py` runs every step with the settings used for the website:

```bash
python main.py all                   # everything, in order
python main.py q1                    # or any of: q1 q2 q3_points q3_rot q4
```

The individual commands it runs:

```bash
# Q1: classification
python train.py --task cls --num_epochs 50
python eval_cls.py --viz --exp_name q1

# Q2: segmentation (objects shown: 0, 471, 434, 351, 577)
python train.py --task seg --num_epochs 30
python eval_seg.py --i 0 --exp_name obj0

# Q3: robustness (no retraining; uses the Q1/Q2 checkpoints)
python eval_cls.py --num_points 100            # also tried 10000, 5000, 1000, 500, 50, 10
python eval_seg.py --num_points 100 --i 0
python eval_cls.py --rot_deg 45                # also tried 0, 15, 30, 60, 90, 180
python eval_seg.py --rot_deg 45 --i 0

# Q4: DGCNN, trained and tested on 1024 points
python train.py --task cls --model dgcnn --num_points 1024 --num_epochs 40 --checkpoint_dir ./checkpoints_dgcnn --exp_name dgcnn
python train.py --task seg --model dgcnn --num_points 1024 --num_epochs 50 --checkpoint_dir ./checkpoints_dgcnn --exp_name dgcnn
python eval_cls.py --model dgcnn --num_points 1024 --checkpoint_dir ./checkpoints_dgcnn --viz --exp_name q4dgcnn
python eval_seg.py --model dgcnn --num_points 1024 --checkpoint_dir ./checkpoints_dgcnn --i 0 --exp_name q4dgcnn_0
```

Notes:

- The pytorch3d build used here was CPU-only, so the GIFs are rendered on the CPU.
- The DGCNN kNN step builds an N×N distance matrix, so Q4 uses 1024 points.
