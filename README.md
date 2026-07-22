# orakara-models

Model weights hosted for the [OraKara](https://github.com/samikchattopadhyay) desktop app's
optional/required model downloads. Every file here is a redistributed or converted copy of
a third-party model, verbatim license terms preserved below.

## Files

| File | Original source | License |
|---|---|---|
| `demucs_v4_two_stems.onnx` + `.onnx.data` | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (pretrained `htdemucs` weights), exported to ONNX offline | MIT (Meta Platforms, Inc.) |
| `aesthetic_head.onnx` | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)'s `sa_0_4_vit_b_32_linear.pth`, converted to ONNX offline | MIT (LAION AI) |

Neither file's weights carry a license separate from the code repo's MIT license (verified
directly against each repo's `LICENSE` file, not assumed) — see `NOTICE` for the full
license text of each.

## Why this repo exists

OraKara's main repository is private, so its own GitHub Release assets aren't publicly
downloadable without authentication. This repo exists solely to host these two files
publicly so the app can auto-download them with no user-configured URL.
