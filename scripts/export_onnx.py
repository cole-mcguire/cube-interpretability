"""
Export the flat CubeTransformer to ONNX for browser inference via onnxruntime-web.

Usage:
    uv run python scripts/export_onnx.py
    uv run python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out docs/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))
from model import load_model


def export(checkpoint: Path, out: Path) -> None:
    print(f"Loading model from {checkpoint}...")
    model = load_model(checkpoint, device="cpu")
    model.eval()

    dummy = torch.zeros(1, 144, dtype=torch.float32)

    with torch.no_grad():
        logits = model(dummy)
    print(f"  Output shape: {logits.shape}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces the legacy TorchScript exporter, which embeds weights
    # inline in the protobuf. The newer dynamo exporter (PyTorch ≥2.x default)
    # produces a skeleton without raw_data, which onnxruntime-web cannot run.
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({size_kb:.0f} KB)")

    # Quick sanity check: reload via onnxruntime and compare outputs
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        inp = np.random.default_rng(0).random((4, 144), dtype=np.float32)
        ort_out = sess.run(["logits"], {"state": inp})[0]
        with torch.no_grad():
            pt_out = model(torch.from_numpy(inp)).numpy()
        max_diff = float(np.abs(ort_out - pt_out).max())
        print(f"  Max abs diff (ort vs torch): {max_diff:.2e}  ({'OK' if max_diff < 1e-4 else 'WARN'})")
    except ImportError:
        print("  (onnxruntime not installed — skipping numeric check)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--out",        default=str(ROOT / "docs" / "model.onnx"))
    args = parser.parse_args()
    export(Path(args.checkpoint), Path(args.out))
