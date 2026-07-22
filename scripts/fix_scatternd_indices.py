"""Post-process fixup for a known torch.onnx (dynamo) exporter rough edge:
ScatterND nodes emitted by the STFT/ISTFT decomposition sometimes get int32
indices, but ONNX's ScatterND spec requires int64 indices. onnxruntime
rejects the graph with an INVALID_GRAPH type error otherwise.

This scans every ScatterND node, and for any whose indices input isn't
declared int64, inserts a Cast(to=int64) immediately before it. Run once
after scripts/export_onnx.py; not part of the Rust runtime pipeline.
"""

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

INT64 = TensorProto.INT64


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "models" / "demucs_v4_two_stems.onnx"

    m = onnx.load(str(path), load_external_data=False)
    g = m.graph

    dtype_by_name = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        dtype_by_name[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        dtype_by_name[init.name] = init.data_type

    fixed = 0
    new_nodes = []
    for node in g.node:
        if node.op_type == "ScatterND":
            idx_name = node.input[1]
            if dtype_by_name.get(idx_name) != INT64:
                cast_out = f"{idx_name}_i64_fix"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[idx_name],
                        outputs=[cast_out],
                        to=INT64,
                        name=f"{node.name}_indices_cast_fix",
                    )
                )
                node.input[1] = cast_out
                dtype_by_name[cast_out] = INT64
                fixed += 1
        new_nodes.append(node)

    del g.node[:]
    g.node.extend(new_nodes)

    onnx.save(m, str(path))
    print(f"fixed {fixed} ScatterND node(s) with non-int64 indices in {path}")


if __name__ == "__main__":
    main()
