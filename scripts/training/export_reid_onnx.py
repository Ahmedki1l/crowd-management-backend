"""Export the OSNet-AIN Re-ID model to a self-contained CPU ONNX graph.

Run in the *training* venv (which has torchreid installed); the server never
needs torch/torchreid — it loads the produced .onnx via onnxruntime (CPU).

The checkpoint on disk (``osnet_ain_x1_0_msmt17.pt``) is a bare state_dict with a
``module.`` DataParallel prefix. We rebuild the architecture, strip the prefix,
load the weights, and wrap the model so the ONNX graph itself does the input
conditioning the server's extractor does *not*:

* the server's ``OSNetEmbeddingExtractor._prepare_crop`` feeds **BGR**, ``/255``,
  CHW crops (no channel swap, no mean/std) — so we bake **BGR->RGB** and the
  **ImageNet mean/std** normalisation into the graph. The server can keep feeding
  exactly what it feeds today and still get correctly-conditioned embeddings.

Output: ``models/reid/osnet_ain_x1_0_msmt17.onnx`` — input ``(N,3,256,128)``
float32 in [0,1] BGR/CHW, output ``(N,512)`` embeddings (dynamic batch).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torchreid.reid.models import build_model

# OSNet x1.0 input geometry and MSMT17 identity count (classifier width).
_INPUT_HW = (256, 128)
_NUM_CLASSES = 4101
_EMBED_DIM = 512


class ReIDGraph(nn.Module):
    """Wraps OSNet so the graph conditions the server's raw BGR/255 CHW input."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        # ImageNet RGB statistics, broadcast over (N,3,H,W).
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,H,W) in [0,1], BGR channel order (as the server feeds it).
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        x = (x - self.mean) / self.std  # ImageNet normalise
        return self.base(x)  # eval mode -> (N, 512) embeddings


def _strip_module_prefix(state: OrderedDict) -> OrderedDict:
    """Drop the ``module.`` DataParallel prefix from every key."""
    cleaned: OrderedDict = OrderedDict()
    for key, value in state.items():
        cleaned[key[len("module.") :] if key.startswith("module.") else key] = value
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/reid/osnet_ain_x1_0_msmt17.pt")
    parser.add_argument("--out", default="models/reid/osnet_ain_x1_0_msmt17.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    weights = Path(args.weights)
    out = Path(args.out)

    model = build_model("osnet_ain_x1_0", num_classes=_NUM_CLASSES, pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(_strip_module_prefix(state), strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval()  # eval() -> forward returns the 512-d feature, not class logits

    graph = ReIDGraph(model).eval().cpu()
    dummy = torch.rand(1, 3, *_INPUT_HW, dtype=torch.float32)

    # Sanity: forward must yield (1, 512) before we commit to the export.
    with torch.no_grad():
        out_dim = graph(dummy).shape
    assert out_dim == (1, _EMBED_DIM), f"unexpected embedding shape {out_dim}"

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        graph,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter (no onnxscript dep; robust for CNNs)
    )
    print(f"  exported {out} ({out.stat().st_size / 1e6:.1f} MB), output dim {_EMBED_DIM}")


if __name__ == "__main__":
    main()
