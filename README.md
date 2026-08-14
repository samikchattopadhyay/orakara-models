# orakara-models

Model weights hosted for the [OraKara](https://github.com/samikchattopadhyay) desktop app's
optional/required model downloads. Every file here is a redistributed or converted copy of
a third-party model, verbatim license terms preserved below.

## Files

| File | Original source | Code license | Weights license |
|---|---|---|---|
| `demucs_v4_two_stems.onnx` + `.onnx.data` | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (pretrained `htdemucs` weights), exported to ONNX offline | MIT (Meta Platforms, Inc.) | **CC-BY-NC-SA 4.0 — non-commercial** (trained on MUSDB18-HQ) |
| `aesthetic_head.onnx` | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)'s `sa_0_4_vit_b_32_linear.pth`, converted to ONNX offline | MIT (LAION AI) | MIT (LAION AI) |
| `MelBandRoformer.onnx` | [KimberleyJensen/Mel-Band-Roformer-Vocal-Model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)'s `MelBandRoformer.ckpt`, exported to ONNX offline | Apache-2.0 (ByteDance Mel-Band RoFormer reference architecture) | MIT, self-declared via the checkpoint author's own Hugging Face repo license tag (`huggingface.co/KimberleyJSN/melbandroformer`) — no separate long-form LICENSE text located |
| `BSRoformerViperx.onnx` | `TRvlvr/model_repo`'s `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (community checkpoint by "viperx"/`playdasegunda`), exported to ONNX offline | MIT (`lucidrains/BS-RoFormer` reference implementation) | **Unverified** — no license stated anywhere for this specific community-trained checkpoint as of 2026-08-13; treat as restricted/non-commercial until a primary-source statement of terms is found |
| `UVR-DeEcho-DeReverb.onnx` | [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) (Ultimate Vocal Remover, "UVR")'s `UVR-DeEcho-DeReverb.pth` (VR-arch "5.1" `CascadedNet`), exported to ONNX offline | MIT (UVR GUI code) | MIT, with a request to credit UVR (Anjok07 & aufr33) as trainers — see caveat below |
| `UVR-DeNoise.onnx` | [Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)'s `UVR-DeNoise.pth` (same VR-arch "5.1" family as DeEcho/DeReverb) | MIT (UVR GUI code) | MIT, with a request to credit UVR (Anjok07 & aufr33) as trainers — see caveat below |
| `skey.onnx` | [deezer/skey](https://github.com/deezer/skey)'s `skey.pt` (S-KEY / ChromaNet, ICASSP 2025), exported to ONNX offline | MIT (Deezer SA) | MIT (Deezer SA) — see caveat below |

**The Demucs weights are not MIT.** The Demucs *code* (and this ONNX export/graph) is MIT,
but the pretrained `htdemucs` *weights* were trained by Meta on the MUSDB18-HQ dataset, which
is licensed CC-BY-NC-SA 4.0 — a non-commercial share-alike license. That restriction
attaches to the trained weights, not just to the training script, so it carries over to
anyone redistributing or using this file. Consuming projects (e.g. OraKara) must be
distributed free/non-commercially, or must separately clear the weights, before bundling
`demucs_v4_two_stems.onnx.data`. See `NOTICE` for the full license text of both licenses.

The LAION aesthetic head's weights carry no license separate from its MIT code repo
(verified directly against the repo's `LICENSE` file, not assumed).

**`BSRoformerViperx.onnx`'s weights license is unverified — treat as non-commercial/
restricted.** Unlike the other files here, no primary-source license statement (a repo
LICENSE file, a Hugging Face license tag, anything) was found for this specific checkpoint
anywhere it's hosted (`TRvlvr/model_repo` and every third-party mirror checked). It's
mirrored here purely as a research/catalog entry alongside the others — OraKara's own
`license_tier: NonCommercial` classification for this model reflects the same caveat, not a
confirmed clearance for commercial use.

**`skey.onnx`'s weights license rests on a slightly indirect reading, flagged here rather
than asserted outright.** deezer/skey's repo-root `LICENSE` file is a standard MIT grant
("Copyright (c) 2019-present, Deezer SA") covering "this software" — and `skey.pt` ships
*inside* the package (`skey/models/skey.pt`) as the CLI's own default checkpoint, not
fetched from a separate, opaquely-licensed source. The README's feature list also states
outright: "🧠 A open-sourced pretrained model." Taken together this is a materially stronger
signal than `MelBandRoformer.onnx`'s case above (a Hugging Face metadata tag with no
long-form LICENSE at all) — but unlike `aesthetic_head.onnx`, deezer/skey's own README
License section phrases it as "the **code** of SKEY is MIT-licensed," which read in
isolation could be parsed as scoping MIT to the code only. No separate model-card or
weights-specific license statement was found to resolve that phrasing definitively either
way. See `NOTICE` for the full LICENSE text and this same reasoning reproduced in full.

## Why this repo exists

OraKara's main repository is private, so its own GitHub Release assets aren't publicly
downloadable without authentication. This repo exists solely to host these two files
publicly so the app can auto-download them with no user-configured URL.

## Reproducibility

[`scripts/`](scripts/) contains the exact, unmodified conversion scripts (copied from
OraKara's main repo) used to produce each release asset from its original upstream
weights, so the provenance here is checkable rather than asserted.

**`demucs_v4_two_stems.onnx` + `.onnx.data`** — via [`scripts/export_demucs_onnx.py`](scripts/export_demucs_onnx.py):
```
pip install torch demucs onnx onnxscript
python scripts/export_demucs_onnx.py --dynamo
python scripts/fix_scatternd_indices.py models/demucs_v4_two_stems.onnx
```
Downloads Meta's pretrained `htdemucs` weights via the `demucs` package (first run only)
and exports the model's native forward pass directly via `torch.onnx.export(dynamo=True)`.
The second script is a small post-export fixup for an ONNX exporter rough edge (some
`ScatterND` nodes need their indices cast to int64) — see the script's own docstring.

**`aesthetic_head.onnx`** — via [`scripts/export_aesthetic_head.py`](scripts/export_aesthetic_head.py):
```
pip install onnx numpy
curl -L -o sa_0_4_vit_b_32_linear.pth \
    https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth
python scripts/export_aesthetic_head.py sa_0_4_vit_b_32_linear.pth aesthetic_head.onnx
```
Reads LAION's single-`nn.Linear(512, 1)` checkpoint directly (no PyTorch needed) and
re-emits it as a 1-node ONNX `Gemm` graph.

**`MelBandRoformer.onnx`** — via [`scripts/export_mel_band_roformer.py`](scripts/export_mel_band_roformer.py):
```
pip install torch einops pyyaml onnx onnxscript
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training  # provides models/bs_roformer/mel_band_roformer.py
curl -L -o config_vocals_mel_band_roformer.yaml \
    https://raw.githubusercontent.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/main/configs/config_vocals_mel_band_roformer.yaml
curl -L -o MelBandRoformer.ckpt \
    https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt
python scripts/export_mel_band_roformer.py
```
Loads Kimberley Jensen's published checkpoint into `ZFTurbo/Music-Source-Separation-
Training`'s `MelBandRoformer` module, monkeypatches the forward pass to stop at the masked
complex spectrogram (ISTFT has no ONNX-exportable path for a genuine complex-dtype tensor —
finished in Rust instead, an exact port of `torch.istft`), and exports via
`torch.onnx.export` (opset 18). Validated against the original unpatched forward pass by
ISTFT-ing the exported path's output in PyTorch and diffing against the reference model's
full audio output on identical random input — bit-exact (mean/max abs diff 0.0).

**`BSRoformerViperx.onnx`** — via [`scripts/export_bs_roformer.py`](scripts/export_bs_roformer.py):
```
pip install torch einops pyyaml onnx onnxscript
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training  # provides models/bs_roformer/bs_roformer.py
curl -L -o config_bs_roformer.yaml \
    https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml
curl -L -o model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt
python scripts/export_bs_roformer.py
```
Same export/validation approach as `MelBandRoformer.onnx` above, against the "viperx"
community checkpoint instead — also bit-exact (mean/max abs diff 0.0) vs. the original
eager forward pass.

**`skey.onnx`** — via [`scripts/export_skey_onnx.py`](scripts/export_skey_onnx.py):
```
git clone https://github.com/deezer/skey.git
cd skey
pip install torch torchaudio numpy einops soundfile nnAudio onnx onnxscript
python export_skey_onnx.py
```
Loads the harmonic-VQT front end (`hcqt.py`'s `VQT`) and `ChromaNet` classifier directly from
`skey.pt`, fuses them into one `forward()` (the reference `CropCQT` module's per-sample crop
loop is only needed for training-time pitch-shift augmentation — real inference always crops
from index 0, so this export bakes that in as a static slice), and exports via
`torch.onnx.export` at a **fixed** input length (15s @ 22,050 Hz = 330,750 samples — S-KEY's
VQT front end does not trace correctly under a dynamic sample-count axis; a dynamic-shape
export attempt silently produced a malformed, wrong-shaped output, while this fixed-shape
export is numerically exact against the original PyTorch model, ~4e-6 max abs diff, float32
noise). The consuming Rust code windows a full track into several fixed-length clips and
aggregates by majority vote rather than needing one dynamic-length call — see
`crates/music-key-model` in the main OraKara repo.

## Verifying a downloaded file matches this repo

The OraKara app itself verifies every download's SHA-256 automatically (see
`src-tauri/src/models.rs`'s catalog in the main repo). To check by hand:
```
sha256sum demucs_v4_two_stems.onnx
# 05e57670601871543f91ac7c3c48cce3c108e3b215b20a631827a229fb67bf2c
sha256sum demucs_v4_two_stems.onnx.data
# fe0084e8279edd25c032e1c36e1c646a097b3b7835a0734ef467fa7577543cae
sha256sum aesthetic_head.onnx
# ed06657a2912fd5f6ac126e047c6b82a3dc1e0b8425b4dba71b1bd93ebb2703c
sha256sum MelBandRoformer.onnx
# 3caf6b9c2a76002de7f00c405a837a36431c0c98f9496cdca0cd081e08c6e95e
sha256sum BSRoformerViperx.onnx
# 590cb5715968f3052efd205491d513abcec3071fef4534e276649a582c886ce3
sha256sum skey.onnx
# 521f82a11c5b52cdf92cb9f1421b1e92234c5437c811bedd689adbe648e891b3
```
