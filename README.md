# orakara-models

Model weights hosted for the [OraKara](https://github.com/samikchattopadhyay) desktop app's
optional/required model downloads. Every file here is a redistributed or converted copy of
a third-party model, verbatim license terms preserved below.

## Files

| File | Original source | Code license | Weights license |
|---|---|---|---|
| `demucs_v4_two_stems.onnx` + `.onnx.data` | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (pretrained `htdemucs` weights), exported to ONNX offline | MIT (Meta Platforms, Inc.) | **CC-BY-NC-SA 4.0 — non-commercial** (trained on MUSDB18-HQ) |
| `aesthetic_head.onnx` | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)'s `sa_0_4_vit_b_32_linear.pth`, converted to ONNX offline | MIT (LAION AI) | MIT (LAION AI) |

**The Demucs weights are not MIT.** The Demucs *code* (and this ONNX export/graph) is MIT,
but the pretrained `htdemucs` *weights* were trained by Meta on the MUSDB18-HQ dataset, which
is licensed CC-BY-NC-SA 4.0 — a non-commercial share-alike license. That restriction
attaches to the trained weights, not just to the training script, so it carries over to
anyone redistributing or using this file. Consuming projects (e.g. OraKara) must be
distributed free/non-commercially, or must separately clear the weights, before bundling
`demucs_v4_two_stems.onnx.data`. See `NOTICE` for the full license text of both licenses.

The LAION aesthetic head's weights carry no license separate from its MIT code repo
(verified directly against the repo's `LICENSE` file, not assumed).

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
```
