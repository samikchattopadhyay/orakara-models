"""One-time offline conversion: pretrained Demucs v4 (htdemucs) -> ONNX.

Not part of the Rust runtime pipeline (see docs/PROMPT.md hard constraints) -
run this once with Python + torch to produce models/demucs_v4_two_stems.onnx,
then the Rust CLI loads that file directly.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from demucs.spec import ispectro

OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "demucs_v4_two_stems.onnx"


def _ispec_no_complex_pad(self, z, length=None, scale=0):
    """Drop-in replacement for HTDemucs._ispec.

    The stock implementation pads the complex spectrogram tensor directly
    (`F.pad` on a complex dtype). The ONNX exporter has no decomposition for
    padding a complex tensor, so this splits it into a real/imaginary pair,
    pads that (numerically identical — padding real and imaginary parts
    separately is the same as padding the complex value), and reassembles.
    """
    hl = self.hop_length // (4**scale)
    zr = torch.view_as_real(z)
    zr = F.pad(zr, (0, 0, 0, 0, 0, 1))
    zr = F.pad(zr, (0, 0, 2, 2))
    z = torch.view_as_complex(zr.contiguous())
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = ispectro(z, hl, length=le)
    x = x[..., pad : pad + length]
    return x


def main():
    HTDemucs._ispec = _ispec_no_complex_pad
    bag = get_model("htdemucs")
    assert len(bag.models) == 1, f"expected a single model, got {len(bag.models)}"
    model = bag.models[0]
    model.eval()

    print("sources:", model.sources)
    print("samplerate:", model.samplerate, "channels:", model.audio_channels)
    print("segment (s):", float(model.segment))

    segment_samples = int(float(model.segment) * model.samplerate)
    print("segment_samples:", segment_samples)

    dummy = torch.zeros(1, model.audio_channels, segment_samples, dtype=torch.float32)

    with torch.no_grad():
        ref_out = model(dummy)
    print("reference output shape:", tuple(ref_out.shape))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dynamo = "--dynamo" in sys.argv
    print(f"exporting with dynamo={dynamo} to {OUT_PATH}")
    torch.onnx.export(
        model,
        (dummy,),
        str(OUT_PATH),
        input_names=["waveform"],
        output_names=["sources"],
        opset_version=18,
        dynamo=dynamo,
    )
    print("done:", OUT_PATH, OUT_PATH.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
