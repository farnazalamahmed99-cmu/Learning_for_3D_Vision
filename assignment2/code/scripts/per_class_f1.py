"""
Q3.3: mean F1@0.05 per class from an eval_model.py log on the 3-class test set.

The evaluation loader is not shuffled, so step i of the log is test item i; this looks up
each item's class in the 3-class test split.

Usage (from the assignment2 folder):
    python eval_model.py --type point --load_checkpoint > eval_3class.log   (with use_full_dataset = True)
    python scripts/per_class_f1.py eval_3class.log
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset_location
from r2n2_custom import R2N2

NAMES = {"02691156": "airplane", "02958343": "car", "03001627": "chair"}

if __name__ == "__main__":
    full = f"{dataset_location.root_location}/r2n2_shapenet_dataset_full"
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = R2N2("test", f"{full}/shapenet", f"{full}/r2n2", f"{full}/split_3c.json")
    scores = {}
    for line in open(sys.argv[1]):
        match = re.match(r"\[\s*(\d+)/\s*\d+\]; .*F1@0.05: ([\d.]+);", line)
        if match:
            name = NAMES[dataset.synset_ids[int(match.group(1))]]
            scores.setdefault(name, []).append(float(match.group(2)))
    for name, values in sorted(scores.items()):
        print(f"{name:9s} n={len(values):4d}  F1@0.05 = {sum(values) / len(values):.2f}")
