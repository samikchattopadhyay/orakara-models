"""One-time conversion: deezer/skey's PyTorch checkpoint (skey.pt) -> ONNX, for this crate's
Rust ONNX Runtime inference. Same offline-conversion pattern every other model in this
workspace already went through (see crates/voice-separator-*/scripts/export_*.py) — Python is
never a runtime dependency of the shipped app, only of this one-off export step.

Why fixed-shape, no dynamic axes: S-KEY's harmonic VQT front end (nnAudio-based) does not
trace correctly through torch.onnx's dynamo exporter under a dynamic sample-count axis --
confirmed empirically (a dynamic-length export silently produced a malformed, wrong output).
A fixed-length export is numerically exact against the original PyTorch model (~4e-6 max abs
diff, float32 noise). The Rust side (music_key_model::KeyModel::detect) works around the fixed
length by windowing: several WINDOW_SAMPLES-long clips tiled across the track, aggregated by
majority vote -- see that function's doc comment.

Usage:
    git clone https://github.com/deezer/skey.git
    cd skey
    python -m venv .venv && .venv/Scripts/activate  (or source .venv/bin/activate)
    pip install torch torchaudio numpy einops soundfile nnAudio onnx onnxscript
    python export_skey_onnx.py
    # -> writes skey.onnx next to this script; ~512KB.

WINDOW_SAMPLES below (330750 = 15s * 22050Hz) must match `WINDOW_SAMPLES` in
crates/music-key-model/src/lib.rs -- if you change one, change both.
"""

import torch

from skey.key_detection import load_checkpoint, load_model_components

CHECKPOINT_PATH = "skey/models/skey.pt"
OUTPUT_PATH = "skey.onnx"
CROP_HEIGHT = 84  # matches key_detection.load_model_components's CropCQT(84)

ckpt = load_checkpoint(CHECKPOINT_PATH)
sample_rate = ckpt["audio"]["sr"]
assert sample_rate == 22050, f"expected 22050 Hz, checkpoint says {sample_rate} Hz"

device = torch.device("cpu")
hcqt, chromanet, _crop_fn = load_model_components(ckpt, device)

WINDOW_SAMPLES = 15 * sample_rate  # 330750
dummy_audio = torch.randn(1, WINDOW_SAMPLES)


class FullModel(torch.nn.Module):
    """hcqt -> static crop (real inference always crops from index 0 -- see
    key_detection.infer_key, which always passes torch.zeros(1) as CropCQT's start index) ->
    chromanet, fused into one graph for a single ONNX export rather than two, and to avoid
    CropCQT's own per-sample Python loop (which only exists for training-time pitch-shift
    augmentation, irrelevant at inference)."""

    def __init__(self, hcqt, chromanet, crop_height):
        super().__init__()
        self.hcqt = hcqt
        self.chromanet = chromanet
        self.crop_height = crop_height

    def forward(self, audio):
        spec = self.hcqt(audio)  # (batch, harmonics, n_bins, time)
        cropped = spec[:, :, : self.crop_height, :]
        return self.chromanet(cropped)  # (batch, 24)


model = FullModel(hcqt, chromanet, CROP_HEIGHT).eval()

with torch.no_grad():
    out = model(dummy_audio)
    assert out.shape == (1, 24), f"unexpected output shape: {out.shape}"

torch.onnx.export(
    model,
    (dummy_audio,),
    OUTPUT_PATH,
    input_names=["audio"],
    output_names=["key_probs"],
    opset_version=17,
)
print(f"Exported {OUTPUT_PATH} (window: {WINDOW_SAMPLES} samples @ {sample_rate} Hz)")
