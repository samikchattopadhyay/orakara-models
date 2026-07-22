#!/usr/bin/env python3
"""One-time, offline conversion of the LAION-AI ViT-B/32 aesthetic linear head to ONNX.

Source checkpoint: https://github.com/LAION-AI/aesthetic-predictor
(`sa_0_4_vit_b_32_linear.pth`, MIT license) — a single `nn.Linear(512, 1)` layer trained
to predict an aesthetic score from a CLIP ViT-B/32 image embedding, matching the 512-dim
embedding this crate's `clip_image.onnx` (CLIP ViT-B/32) produces.

No PyTorch needed: `torch.save(state_dict())` on a single Linear layer is a zip archive
containing raw little-endian float32 tensor bytes in predictable named entries
(`archive/data/0` = weight `[1, 512]`, `archive/data/1` = bias `[1]`, confirmed by
inspecting `archive/data.pkl`'s pickle opcodes) — small enough to read directly with
`zipfile` + `numpy` and re-emit as a 1-node ONNX `Gemm` graph via the `onnx` package.

Usage:
    pip install onnx numpy
    curl -L -o sa_0_4_vit_b_32_linear.pth \
        https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth
    python scripts/export_aesthetic_head.py sa_0_4_vit_b_32_linear.pth models/aesthetic_head.onnx
"""
import sys
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EMBED_DIM = 512


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <sa_0_4_vit_b_32_linear.pth> <output.onnx>")
    pth_path, out_path = sys.argv[1], sys.argv[2]

    z = zipfile.ZipFile(pth_path)
    # Verified via the pickle bytecode in archive/data.pkl: storage "0" is the "weight"
    # key (shape [1, 512]), storage "1" is "bias" (shape [1]) — the natural nn.Linear
    # state_dict order, not assumed blindly.
    weight = np.frombuffer(z.read("archive/data/0"), dtype="<f4").reshape(1, EMBED_DIM).copy()
    bias = np.frombuffer(z.read("archive/data/1"), dtype="<f4").reshape(1).copy()

    weight_init = numpy_helper.from_array(weight.astype(np.float32), name="weight")
    bias_init = numpy_helper.from_array(bias.astype(np.float32), name="bias")
    input_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, EMBED_DIM])
    output_tensor = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])

    # y = embedding @ weight^T + bias
    node = helper.make_node("Gemm", ["embedding", "weight", "bias"], ["score"], transB=1, name="aesthetic_linear")
    graph = helper.make_graph([node], "laion_aesthetic_vit_b_32_linear", [input_tensor], [output_tensor],
                               initializer=[weight_init, bias_init])
    model = helper.make_model(graph, producer_name="framepick-aesthetic-export",
                               opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
