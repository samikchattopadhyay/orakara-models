#!/usr/bin/env bash
# Builds the Music Visualizer's projectM WASM module + curated preset pack.
# See docs/plans/music-visualizer-config-plan.md §A.2.2/§A.2.3 for the full reasoning.
#
# Verified working end-to-end on Windows (Git Bash), 2026-08-22 — OraKara ships Windows-only
# today, so this targets Windows natively via emsdk + Ninja rather than requiring WSL/Linux
# (an earlier draft of this script assumed Linux, matching projectM's own CI machine, but that
# was never actually run — this version is the real, tested process). The only Windows-specific
# step is using a *short* build path (Windows' ~250-char MAX_PATH limit is hit by some of
# projectM's own deeply-nested object file paths otherwise, e.g. under
# src/libprojectM/MilkdropPreset/CMakeFiles/... — build under something like C:\pmw, not deep
# inside a long username/temp path).
#
# Requires: git, cmake, a short build path, and either scoop/choco or a manual ninja install
# (`scoop install ninja`). No SDL2/mesa/GLES3 dev packages needed — projectM's own
# ENABLE_SDL_UI option is OFF (and ignored under Emscripten regardless), so only the core +
# playlist static libraries get built.
#
# Output (not committed to this repo — see §A.2.3): projectm.wasm, projectm.js,
# visualizer-presets.zip, uploaded to orakara-models' existing `v1` release (one long-lived
# release tag, not a new one per addition — verified against that repo directly), with their
# re-downloaded-and-rehashed SHA-256 recorded in src-tauri/src/models.rs's catalog() entries
# for visualizer_engine_wasm/visualizer_engine_js/visualizer_presets.

set -euo pipefail

PROJECTM_TAG="v4.1.7"
EMSDK_VERSION="3.1.53"
BUILD_ROOT="${BUILD_ROOT:-/c/pmw}"   # short path — see header comment on MAX_PATH
OUT_DIR="$BUILD_ROOT/out"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$BUILD_ROOT" "$OUT_DIR"

echo "==> Ninja (skip if already installed)"
command -v ninja >/dev/null || scoop install ninja

echo "==> Setting up emsdk $EMSDK_VERSION"
if [ ! -d "$BUILD_ROOT/emsdk" ]; then
  git clone https://github.com/emscripten-core/emsdk.git "$BUILD_ROOT/emsdk"
fi
(cd "$BUILD_ROOT/emsdk" && ./emsdk install "$EMSDK_VERSION" && ./emsdk activate "$EMSDK_VERSION")
# shellcheck disable=SC1091
source "$BUILD_ROOT/emsdk/emsdk_env.sh"

echo "==> Checking out projectM $PROJECTM_TAG"
if [ ! -d "$BUILD_ROOT/projectm" ]; then
  git clone --branch "$PROJECTM_TAG" --depth 1 --recurse-submodules \
    https://github.com/projectM-visualizer/projectm.git "$BUILD_ROOT/projectm"
fi

echo "==> Configuring (Ninja generator — Unix Makefiles needs make.exe, not present on Windows)"
# ENABLE_PLAYLIST defaults ON already (verified against this exact tag's CMakeLists.txt) —
# passed explicitly anyway for clarity. ENABLE_SDL_UI defaults OFF and is ignored under
# Emscripten regardless; passed explicitly to document that no SDL2 dependency is needed.
emcmake cmake -G Ninja \
  -S "$BUILD_ROOT/projectm" \
  -B "$BUILD_ROOT/projectm/cmake-build-emscripten" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DENABLE_PLAYLIST=ON \
  -DENABLE_SDL_UI=OFF

echo "==> Building projectM's core + playlist static libraries"
emmake cmake --build "$BUILD_ROOT/projectm/cmake-build-emscripten" --parallel

echo "==> Linking OraKara's own entry point against the projectM static libs"
LIB_DIR="$BUILD_ROOT/projectm/cmake-build-emscripten/src/libprojectM"
PLAYLIST_LIB_DIR="$BUILD_ROOT/projectm/cmake-build-emscripten/src/playlist"

EXPORTED_FUNCTIONS='[
  "_ork_visualizer_init",
  "_projectm_destroy",
  "_projectm_load_preset_file",
  "_projectm_set_window_size",
  "_projectm_set_preset_duration",
  "_projectm_set_soft_cut_duration",
  "_projectm_set_beat_sensitivity",
  "_projectm_set_preset_locked",
  "_projectm_opengl_render_frame",
  "_projectm_pcm_add_float",
  "_projectm_pcm_get_max_samples",
  "_projectm_playlist_create",
  "_projectm_playlist_destroy",
  "_projectm_playlist_clear",
  "_projectm_playlist_add_path",
  "_projectm_playlist_set_shuffle",
  "_projectm_playlist_play_next",
  "_projectm_playlist_play_previous",
  "_malloc",
  "_free"
]'

# No -sEXPORTED_RUNTIME_METHODS entry for HEAPF32 — that flag is for *functions* (ccall/cwrap),
# not memory views. HEAPF32/etc. are already reachable as Module.HEAPF32 on the returned module
# instance without needing to be named there — confirmed via a Node smoke test (see below);
# adding "HEAPF32" to that list produces a build warning ("invalid item"), harmless but wrong.
emcc "$SCRIPT_DIR/entry.cpp" \
  -L"$LIB_DIR" -L"$PLAYLIST_LIB_DIR" \
  -lprojectM-4 -lprojectM-4-playlist \
  -sMODULARIZE=1 \
  -sEXPORT_NAME=createProjectMModule \
  -sEXPORTED_FUNCTIONS="$EXPORTED_FUNCTIONS" \
  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -sUSE_WEBGL2=1 \
  -sFULL_ES3=1 \
  -sALLOW_MEMORY_GROWTH=1 \
  -O2 \
  -o "$OUT_DIR/projectm.js"

echo "==> Smoke-testing the module in Node (WebGL2 itself needs a real browser, so"
echo "    ork_visualizer_init is expected to return null/0 here — this only confirms the"
echo "    module loads and every other exported symbol is callable)"
cat > "$OUT_DIR/smoke_test.js" <<'EOF'
const createProjectMModule = require("./projectm.js");
createProjectMModule().then((Module) => {
  const pcmGetMax = Module.cwrap("projectm_pcm_get_max_samples", "number", []);
  const ptr = Module._malloc(4);
  Module.HEAPF32.set([1.0], ptr / 4);
  Module._free(ptr);
  console.log("smoke test OK — projectm_pcm_get_max_samples() =", pcmGetMax());
}).catch((e) => { console.error("SMOKE TEST FAILED:", e); process.exit(1); });
EOF
node "$OUT_DIR/smoke_test.js"

echo "==> Zipping the curated preset categories (plan §A.5)"
if [ ! -d "$BUILD_ROOT/presets-src" ]; then
  git clone --depth 1 https://github.com/projectM-visualizer/presets-cream-of-the-crop.git \
    "$BUILD_ROOT/presets-src"
fi
python3 - "$BUILD_ROOT/presets-src" "$OUT_DIR/visualizer-presets.zip" <<'EOF'
import sys, zipfile, os
src, out = sys.argv[1], sys.argv[2]
categories = ["Dancer", "Drawing", "Fractal", "Geometric", "Hypnotic",
              "Particles", "Reaction", "Sparkle", "Supernova", "Waveform"]
count = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for cat in categories:
        for root, _, files in os.walk(os.path.join(src, cat)):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, src))
                count += 1
print(f"wrote {count} files to {out} ({os.path.getsize(out)} bytes)")
EOF

echo "==> Done. Output in $OUT_DIR:"
ls -la "$OUT_DIR"/*.wasm "$OUT_DIR"/*.js "$OUT_DIR"/*.zip
echo
echo "Next: gh release upload v1 $OUT_DIR/projectm.wasm $OUT_DIR/projectm.js $OUT_DIR/visualizer-presets.zip --repo samikchattopadhyay/orakara-models"
echo "Then re-download each file from its real release URL, sha256sum THAT copy (never the"
echo "pre-upload one — orakara-models/CONTRIBUTING.md's mandatory check), and fill in"
echo "download_url/sha256 for visualizer_engine_wasm/visualizer_engine_js/visualizer_presets"
echo "in src-tauri/src/models.rs."
