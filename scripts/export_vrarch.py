"""PyTorch -> ONNX export for UVR's VR-arch "5.1" (`CascadedNet`) checkpoints.

Run this offline (needs a PyTorch environment this Rust app doesn't carry — same reasoning
as ``crates/voice-separator-roformer/scripts/export_*.py``) to turn a downloaded UVR ``.pth``
checkpoint into the ``.onnx`` graph ``voice-separator-vrarch`` actually loads at runtime.

The `CascadedNet`/`BaseNet`/layer classes below are a direct, unmodified-logic port of
``nomadkaraoke/python-audio-separator``'s ``audio_separator/separator/uvr_lib_v5/vr_network/
nets_new.py`` + ``layers_new.py`` (fetched/confirmed 2026-08-13) -- inlined here rather than
imported from a checked-out copy of that package so this script has no dependency beyond
plain ``torch``, matching how small the actual architecture is (~150 lines). If UVR/
python-audio-separator ever changes this architecture, re-diff against that repo's current
``vr_network/`` before assuming this script is still accurate.

Usage:
    python export_vrarch.py UVR-DeEcho-DeReverb.pth UVR-DeEcho-DeReverb.onnx --nout 64
    python export_vrarch.py UVR-DeNoise.pth UVR-DeNoise.onnx --nout 48

``--nout``/``--nout-lstm`` come from UVR's own ``vr_model_data/model_data.json`` per
checkpoint (looked up by the file's UVR-style last-10MB MD5 hash — see
``crates/voice-separator-vrarch/src/config.rs``'s doc comments for the two values already
confirmed for this crate's two shipped checkpoints: DeEcho-DeReverb is ``nout=64,
nout_lstm=128`` [triggered by this checkpoint's file size matching UVR's "VR 5.1" size table
rather than an explicit ``model_data.json`` entry -- see ``nets_new.py``'s
``nout = 64 if nn_arch_size == 218409 else nout``], DeNoise is ``nout=48, nout_lstm=128``
[explicit in ``model_data.json``]). Get this wrong and ``load_state_dict`` below will fail
loudly (shape mismatch) rather than silently producing a broken export.
"""

import argparse
import types

import torch
import torch.nn.functional as F
from torch import nn

# --- Port of layers_new.py -------------------------------------------------------------


def crop_center(h1, h2):
    """Ports `spec_utils.crop_center` -- the only helper `layers_new.Decoder` needs."""
    h1_shape = h1.size()
    h2_shape = h2.size()
    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    return h1[:, :, :, s_time:e_time]


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def __call__(self, x):
        return self.conv2(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv1(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super().__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ))
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode="bilinear", align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = self.bottleneck(torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1))
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LSTMModule(nn.Module):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super().__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True)
        self.dense = nn.Sequential(nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU())

    def forward(self, x):
        n, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]
        h = h.permute(2, 0, 1)
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size()[-1]))
        h = h.reshape(nframes, n, 1, nbins)
        h = h.permute(1, 2, 3, 0)
        return h


# --- Port of nets_new.py ----------------------------------------------------------------


class BaseNet(nn.Module):
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super().__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = Encoder(nout * 6, nout * 8, 3, 2, 1)
        self.aspp = ASPPModule(nout * 8, nout * 8, dilations, dropout=True)
        self.dec4 = Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        h = self.aspp(e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)
        return h


class CascadedNet(nn.Module):
    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super().__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm), Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(2, nout // 4, self.nin_lstm // 2, nout_lstm // 2)

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm), Conv2DBNActiv(nout, nout // 2, 1, 1, 0)
        )
        self.stg2_high_band_net = BaseNet(nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2)

        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, : self.max_bin]
        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)
        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)
        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)
        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(mask, pad=(0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")
        return mask


# --- Export ------------------------------------------------------------------------------

# Must match `crates/voice-separator-vrarch/src/config.rs`'s `TOTAL_BINS`/`WINDOW_SIZE`
# constants -- both sides hardcode UVR's own confirmed defaults for `4band_v3` checkpoints
# rather than reading them from a shared config file.
TOTAL_BINS = 673  # BINS (672) + 1, matches CascadedNet's `n_fft // 2 + 1` for n_fft=1344
N_FFT = 1344  # `mp.param["bins"] * 2` for `4band_v3.json` (`bins: 672`)
WINDOW_SIZE = 512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Input UVR .pth checkpoint")
    parser.add_argument("output", help="Output .onnx path")
    parser.add_argument("--nout", type=int, required=True, help="See this script's module docstring")
    parser.add_argument("--nout-lstm", type=int, default=128)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    model = CascadedNet(N_FFT, nout=args.nout, nout_lstm=args.nout_lstm)
    model.eval()

    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        # UVR's own .pth files are a plain state dict (no wrapping "state_dict"/"model" key,
        # confirmed against `architectures/vr_separator.py`'s own
        # `self.model_run.load_state_dict(torch.load(self.model_path, map_location="cpu"))`,
        # which passes no `weights_only` at all) -- fall back to the unrestricted loader only
        # if the strict/safe one rejects it.
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    # The traced entry point is `forward` (mask prediction, sigmoid + replicate-pad to
    # `output_bin` rows) plus the reference implementation's own offset-crop on the time
    # axis -- see `architectures/vr_separator.py::VRSeparator.inference_vr`'s `_execute`,
    # which calls exactly `predict_mask` (forward + this crop) per inference window. Baking
    # the crop into the exported graph means the Rust runtime never needs to know about
    # `offset` at all. Monkey-patched onto the instance (calling the *class*'s `forward` by
    # name, not `self.forward`, to avoid recursing into this same patched method) rather than
    # added as a real `predict_mask` method, so `torch.onnx.export`'s default trace-of-
    # `forward` picks it up with no extra `--dynamo`/method-selection plumbing -- same
    # pattern `voice-separator-roformer/scripts/export_mel_band_roformer.py` uses.
    def export_forward(self, x):
        mask = CascadedNet.forward(self, x)
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
        return mask

    model.forward = types.MethodType(export_forward, model)

    dummy = torch.randn(1, 2, TOTAL_BINS, WINDOW_SIZE)
    with torch.no_grad():
        out = model(dummy)
    expected_roi = WINDOW_SIZE - 2 * model.offset
    print("export-path output shape:", tuple(out.shape), "(expected (1, 2, 673, %d))" % expected_roi)
    assert out.shape == (1, 2, TOTAL_BINS, expected_roi), out.shape

    print(f"exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        input_names=["magnitude"],
        output_names=["mask"],
        opset_version=args.opset,
        dynamo=False,
    )
    print("export done ->", args.output)


if __name__ == "__main__":
    main()
