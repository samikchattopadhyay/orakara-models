import types
import sys
import yaml
import torch
from einops import rearrange, pack, unpack, repeat

sys.path.insert(0, r"C:\tmp\roformer")
from models.bs_roformer.mel_band_roformer import MelBandRoformer, pack_one, unpack_one

with open("config_vocals_mel_band_roformer.yaml", "r") as f:
    cfg = yaml.unsafe_load(f)

model_cfg = dict(cfg["model"])
model_cfg["flash_attn"] = False
model = MelBandRoformer(**model_cfg)
model.eval()

ckpt = torch.load("MelBandRoformer.ckpt", map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
new_state = {}
for k, v in state_dict.items():
    nk = k
    for prefix in ("model.", "module."):
        if nk.startswith(prefix):
            nk = nk[len(prefix):]
    new_state[nk] = v
missing, unexpected = model.load_state_dict(new_state, strict=False)
assert not missing and not unexpected, (missing, unexpected)

chunk_size = cfg["inference"]["chunk_size"]


def export_forward(self, raw_audio):
    device = raw_audio.device
    if raw_audio.ndim == 2:
        raw_audio = rearrange(raw_audio, "b t -> b 1 t")
    batch, channels, raw_audio_length = raw_audio.shape

    raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")
    stft_window = self.stft_window_fn(device=device)

    # return_complex=False keeps this exportable (the ONNX stft symbolic rejects
    # return_complex=True) -- output is already the [...,2] real/imag layout that
    # torch.view_as_real would otherwise produce from the complex-True call.
    stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=False)
    stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
    stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

    batch_arange = torch.arange(batch, device=device)[..., None]
    x = stft_repr[batch_arange, self.freq_indices]
    x = rearrange(x, "b f t c -> b t (f c)")
    x = self.band_split(x)

    store = [None] * len(self.layers)
    for i, transformer_block in enumerate(self.layers):
        if len(transformer_block) == 3:
            linear_transformer, time_transformer, freq_transformer = transformer_block
            x, ft_ps = pack([x], "b * d")
            x = linear_transformer(x)
            (x,) = unpack(x, ft_ps, "b * d")
        else:
            time_transformer, freq_transformer = transformer_block

        if self.skip_connection:
            for j in range(i):
                x = x + store[j]

        x = rearrange(x, "b t f d -> b f t d")
        x, ps = pack([x], "* t d")
        x = time_transformer(x)
        (x,) = unpack(x, ps, "* t d")
        x = rearrange(x, "b f t d -> b t f d")
        x, ps = pack([x], "* f d")
        x = freq_transformer(x)
        (x,) = unpack(x, ps, "* f d")
        if self.skip_connection:
            store[i] = x

    heads = self.mask_estimators
    num_stems = len(heads)
    masks = torch.stack([fn(x) for fn in heads], dim=1)
    masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

    stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

    scatter_indices = repeat(
        self.freq_indices, "f -> b n f t c", b=batch, n=num_stems, t=stft_repr.shape[-2], c=2
    )
    stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
    masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

    denom = repeat(self.num_bands_per_freq, "f -> (f r) 1 1", r=channels)
    masks_averaged = masks_summed / denom.clamp(min=1e-8)

    # Manual complex multiply via real ops -- torch.view_as_complex has no ONNX symbolic
    # (and aten::complex, the alternative construction, has none either): (a+bi)(c+di) =
    # (ac-bd) + (ad+bc)i.
    a_re, a_im = stft_repr[..., 0], stft_repr[..., 1]
    m_re, m_im = masks_averaged[..., 0], masks_averaged[..., 1]
    out_re = a_re * m_re - a_im * m_im
    out_im = a_re * m_im + a_im * m_re
    masked = torch.stack([out_re, out_im], dim=-1)  # [b, n, F, t, 2]

    masked = rearrange(masked, "b n (f s) t c -> (b n s) f t c", s=self.audio_channels)

    if self.zero_dc:
        zero_dc_mask = torch.ones(masked.shape[1], device=device)
        zero_dc_mask = zero_dc_mask.clone()
        zero_dc_mask[0] = 0.0
        masked = masked * zero_dc_mask.view(1, -1, 1, 1)

    # ISTFT is done in Rust instead of here: torch.istft requires a genuine complex-dtype
    # tensor, and there is no ONNX-exportable path to construct one (`aten::complex` has
    # no symbolic; `torch.stft(..., return_complex=True)`'s own symbolic explicitly
    # rejects complex output) -- so the graph stops at the masked complex spectrogram
    # (real/imag stacked in the last dim) and Rust's `Stft::inverse` (an exact port of
    # `torch.istft`'s algorithm, already built and validated for the MDX-Net backend)
    # finishes the job.
    return masked


model.forward = types.MethodType(export_forward, model)

dummy = torch.randn(1, 2, chunk_size)
with torch.no_grad():
    out = model(dummy)
print("export-path output shape:", out.shape, out.dtype)

# Cross-check against the ORIGINAL (unpatched) forward's pre-istft state isn't directly
# comparable since the original collapses straight to audio -- instead sanity check by
# feeding this masked spectrogram through torch's own istft and comparing against the
# original model's full audio output, on the same input.
import importlib

orig_module = importlib.import_module("models.bs_roformer.mel_band_roformer")
orig_model = orig_module.MelBandRoformer(**model_cfg)
orig_model.load_state_dict(new_state, strict=False)
orig_model.eval()
with torch.no_grad():
    ref_audio = orig_model(dummy)
print("reference eager audio shape:", ref_audio.shape)

# out: [(b n s), f, t, 2] = [2, 1025, t, 2] (num_stems=1, stereo)
complex_spec = torch.complex(out[..., 0], out[..., 1])
stft_window = model.stft_window_fn(device="cpu")
recon = torch.istft(
    complex_spec,
    **model.stft_kwargs,
    window=stft_window,
    return_complex=False,
    length=chunk_size,
)
recon = recon.unsqueeze(0)  # [1, 2, T]
diff = (recon - ref_audio).abs()
print("mean abs diff (export-path istft'd vs reference):", diff.mean().item())
print("max abs diff:", diff.max().item())

print("exporting to ONNX (opset 18)...")
torch.onnx.export(
    model,
    (dummy,),
    "MelBandRoformer.onnx",
    input_names=["waveform"],
    output_names=["masked_spectrogram"],
    opset_version=18,
    dynamo=False,
)
print("export done")
