"""Q3.3: runtime and GPU memory of Mini-VGGT vs number of input views (batch size 1).

Run from the repo root:  python bench_views.py outputs/vggt_train/best.pt out.json
"""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from models.vggt.runtime import build_components  # noqa: E402

ckpt_path, out_path = sys.argv[1], sys.argv[2]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model, _, _ = build_components(ckpt["config"], from_checkpoint=True)
model.load_state_dict(ckpt["model"])
model.cuda()
device = torch.device("cuda")
results = []
for v in (2, 4, 8, 16, 24, 32):
    rgb = torch.rand(1, v, 3, 224, 224, device=device)
    row = {"views": v}
    # Inference: no_grad + bf16 autocast, as in evaluation.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            model({"rgb": rgb})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(10):
            model({"rgb": rgb})
        torch.cuda.synchronize()
    row["infer_ms"] = (time.perf_counter() - start) / 10 * 1000
    row["infer_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    # Training step: forward + backward (frozen DINO), bf16 autocast.
    model.train()
    try:
        for i in range(4):
            if i == 1:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model({"rgb": rgb})
            loss = out["depth"].float().mean() + out["pose_enc"].float().pow(2).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row["train_ms"] = (time.perf_counter() - start) / 3 * 1000
        row["train_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    except torch.cuda.OutOfMemoryError:
        row["train_ms"] = row["train_peak_mb"] = None
    torch.cuda.empty_cache()
    print(row, flush=True)
    results.append(row)
json.dump(results, open(out_path, "w"), indent=2)
