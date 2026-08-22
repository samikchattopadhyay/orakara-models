# Contributing to orakara-models

This repo hosts model weights (and other large downloadable assets) the OraKara desktop app
auto-downloads. It exists purely so OraKara's own private repo doesn't need public Release
assets for this — see `README.md`'s "Why this repo exists". This file is the checklist
OraKara's own `CONTRIBUTING.md` points at ("Adding a new hosted model file") — follow it in
full for every new file added here or every time an existing file is replaced.

## Why this exists

A real gap shipped once: the VR-arch export (`UVR-DeEcho-DeReverb.onnx`/`UVR-DeNoise.onnx`)
landed with the files uploaded to the `v1` release and catalogued in OraKara's `models.rs`,
but this repo's own README `Reproducibility`/`Verifying` sections were never updated to match
— and that went unnoticed for a while. Writing the code that produces a correct artifact is
not the same as verifying the artifact that actually ended up hosted. The checklist below
exists specifically to catch that class of gap.

## Checklist — every new or replaced hosted file

- [ ] **Export/conversion script** added to `scripts/` (or updated, if replacing a file) —
      the exact, unmodified script that produced the artifact, copied from OraKara's main
      repo if it lives there, so provenance here is checkable rather than asserted. Non-
      Rust/Python assets (e.g. a compiled binary/WASM module) still get a script here: the
      exact, reproducible build steps, not just the resulting file.
- [ ] **File uploaded** to the `v1` GitHub Release (`gh release upload v1 <file>` — this repo
      uses one long-lived release tag for every asset, not a new tag per file; do not create
      a new release/tag for a routine addition).
- [ ] **`README.md`'s Files table** gets a new row (or its existing row updated): original
      source, code license, weights/content license — matching the exact caveat-vs-confirmed
      phrasing convention already used there (e.g. "Unverified — treat as restricted" when a
      license genuinely can't be confirmed from a primary source; don't round an unclear case
      up to a clean license name).
- [ ] **`README.md`'s Reproducibility section** gets the new script's exact invocation (the
      literal commands to run, in order, with what each one downloads/produces) — same shape
      as every existing entry there.
- [ ] **`README.md`'s Verifying section** gets a new `sha256sum <file>` line with the real
      hash of the file that is actually sitting in the release, not a hash computed before
      upload (see the mandatory re-verification step below — this line must be filled in
      *after* that step, from its output, not from a locally-computed value).
- [ ] **`NOTICE`** gets a new `====`-delimited section: the full, verbatim license text for
      the file's source (code license, and separately the weights/content license if they
      differ — see the Demucs section in `NOTICE` for the shape when they do), sourced from
      the actual upstream LICENSE file, not paraphrased or assumed from a README blurb alone.
- [ ] **Mandatory re-verification — do this last, after the file is already uploaded:**
      download the file back down from its real public release URL (not the local copy you
      just uploaded) and run `sha256sum` on *that* downloaded copy. This must match both the
      hash you're about to write into `README.md`'s Verifying section and the hash you're
      about to write into OraKara's `models.rs` catalog entry. **Do not trust the hash you
      computed locally before uploading** — a corrupted upload, a wrong file, or an upload
      that silently picked up a stale local copy would all produce a locally-correct hash
      that doesn't match what's actually being served. This is the single step the VR-arch
      gap above would have caught immediately if it had existed at the time.
- [ ] **OraKara's own `models.rs` catalog entry** (main repo, `c:\Development\OraKara`) added
      or updated with the real `download_url` and the re-verified `sha256` from the step
      above — never a placeholder/estimated hash, never `None` once a real file is uploaded
      (a catalog entry with `sha256: None` skips verification entirely, which defeats the
      point of hosting it here in the first place).
- [ ] **OraKara's `LEGAL.md`** per-component table (and, if the file introduces a genuinely
      new license family this repo hasn't hosted before, `src-tauri/resources/legal/`'s own
      notice text) reflects the same license conclusion this repo's `NOTICE`/README do — the
      two repos' licensing claims for the same file must never diverge.

## What "done" looks like

Every item above checked, in one pass, before considering the task finished — not "code
merged, docs later." A hosted file with an unchecked box here is the exact failure mode this
checklist exists to prevent.
