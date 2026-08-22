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
| `projectm.wasm` + `projectm.js` | [projectM-visualizer/projectm](https://github.com/projectM-visualizer/projectm) (tag `v4.1.7`), compiled to WebAssembly via Emscripten, plus a small OraKara-authored Emscripten entry point ([`ork_visualizer_init`](scripts/projectm-wasm/entry.cpp)) | **LGPL-2.1** (projectM) | n/a — an engine, not a trained model |
| `visualizer-presets.zip` | [projectM-visualizer/presets-cream-of-the-crop](https://github.com/projectM-visualizer/presets-cream-of-the-crop), 10 curated category folders (Dancer, Drawing, Fractal, Geometric, Hypnotic, Particles, Reaction, Sparkle, Supernova, Waveform) out of that pack's full ~9,800 Milkdrop presets | n/a — content, not code | Public domain by longstanding community convention — see caveat below |

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

**`projectm.wasm`/`projectm.js` are never statically linked into the OraKara binary** — they're
downloaded from this repo at runtime into the app's own models directory and loaded from there,
which is what makes LGPL-2.1's "the user must be able to substitute a modified/updated version
of the library" condition trivial to satisfy: replacing the file here (or on a user's machine)
needs no rebuild of anything else. See OraKara's own `LEGAL.md` and
`docs/plans/music-visualizer-config-plan.md` §A.7 for the fuller reasoning.

**`visualizer-presets.zip`'s public-domain framing is a community convention, not a documented
legal fact.** Per `presets-cream-of-the-crop`'s own `LICENSE.md`: Milkdrop presets were, in
almost all cases, never released under any specific license by their original authors, who
theoretically hold full copyright; because they've been freely shared and reused across
countless Milkdrop/projectM packages for two decades, that upstream repo (and this one, in
turn) treats them as public domain by that longstanding convention — not because any preset
author formally dedicated their work to the public domain. See `NOTICE` for the exact upstream
wording, reproduced verbatim.

## Why this repo exists

OraKara's main repository is private, so its own GitHub Release assets aren't publicly
downloadable without authentication. This repo exists solely to host these files
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

**`projectm.wasm` + `projectm.js`** — via [`scripts/projectm-wasm/build.sh`](scripts/projectm-wasm/build.sh)
and [`scripts/projectm-wasm/entry.cpp`](scripts/projectm-wasm/entry.cpp):
```
emsdk install 3.1.53 && emsdk activate 3.1.53
git clone --branch v4.1.7 --depth 1 --recurse-submodules https://github.com/projectM-visualizer/projectm.git
emcmake cmake -G Ninja -S projectm -B projectm/cmake-build-emscripten \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DENABLE_PLAYLIST=ON -DENABLE_SDL_UI=OFF
emmake cmake --build projectm/cmake-build-emscripten --parallel
emcc entry.cpp -Lprojectm/cmake-build-emscripten/src/libprojectM \
    -Lprojectm/cmake-build-emscripten/src/playlist \
    -lprojectM-4 -lprojectM-4-playlist \
    -sMODULARIZE=1 -sEXPORT_NAME=createProjectMModule \
    -sEXPORTED_FUNCTIONS=[...see build.sh for the full list...] \
    -sEXPORTED_RUNTIME_METHODS=[ccall,cwrap] \
    -sUSE_WEBGL2=1 -sFULL_ES3=1 -sALLOW_MEMORY_GROWTH=1 -O2 \
    -o projectm.js
```
`ENABLE_SDL_UI=OFF` (and its own default, `OFF` — ignored under Emscripten regardless) means
no SDL2 dependency is needed at all: only libprojectM's core + playlist static libraries get
built, linked against `entry.cpp` (OraKara's own ~50-line Emscripten entry point — not
projectM's code — which creates the WebGL2 context `projectm_create()` needs before it's
called; projectM does not create its own window/context in library mode). Verified working:
smoke-tested in Node (module loads, `cwrap`/`HEAPF32`/`_malloc` all functional,
`projectm_pcm_get_max_samples()` returns a real value) — `ork_visualizer_init` itself needs a
real browser WebGL2 context to succeed, so it correctly returns null under Node, exercising the
intended graceful-failure path rather than a true end-to-end render (that needs the OraKara app
itself, in a real browser/webview).

**`visualizer-presets.zip`** — the 10 category folders named above, copied verbatim (recursive,
including every author subfolder within each category) from a shallow clone of
`presets-cream-of-the-crop` and zipped with each category folder as a top-level entry (so the
archive's own paths are `<Category>/<Author>/*.milk`, matching what OraKara's
`archive_extract_all_to` extraction (see its `models.rs`) reproduces verbatim under
`models_dir()/visualizer-presets/`):
```
git clone --depth 1 https://github.com/projectM-visualizer/presets-cream-of-the-crop.git
python3 -c "
import zipfile, os
categories = ['Dancer','Drawing','Fractal','Geometric','Hypnotic','Particles','Reaction','Sparkle','Supernova','Waveform']
with zipfile.ZipFile('visualizer-presets.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for cat in categories:
        for root, _, files in os.walk(f'presets-cream-of-the-crop/{cat}'):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, 'presets-cream-of-the-crop'))
"
```
9,791 files, ~143 MB uncompressed, ~30 MB zipped.

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
sha256sum projectm.wasm
# 30982d9d1e7893ba9eb88f06f047bac3fe86800b2475e8faf32d75544fe3e775
sha256sum projectm.js
# 0c06852fc9ea5c2874a185a161dba8248d276ab98894eaf6cc7fac0851b8325f
sha256sum visualizer-presets.zip
# d29ea94e319010aac0fc40f313386bc933b1fedbe88a299d05d8a6801407a7b8
```
The three hashes above were computed after re-downloading each file back down from this repo's
actual `v1` release (not from the local copy produced before upload) — the mandatory
verification step this repo's own `CONTRIBUTING.md` requires for every new hosted file.
