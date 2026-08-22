// OraKara's own Emscripten entry point for the Music Visualizer's projectM engine.
//
// Not part of projectM's own source tree — this is the ~50-line glue docs/plans/
// music-visualizer-config-plan.md's §A.3 calls for: projectM's C API (core.h/audio.h/
// parameters.h/render_opengl.h/playlist_*.h, all `extern "C"`) is exported directly via the
// build's `-sEXPORTED_FUNCTIONS` list (see build.sh in this directory), so no wrapper is needed
// for those calls themselves. The one thing genuinely missing is a WebGL2 context bound to a
// target canvas *before* `projectm_create()` runs — projectM does not create its own
// window/context in library mode (unlike the `projectM_SDL_emscripten` demo app projectM's own
// examples-emscripten repo builds, a full SDL application this project does not use). That's
// the entire job of this file.
//
// License: this file is OraKara's own code (not projectM's), covered by this repository's own
// license, not projectM's LGPL-2.1 — see LEGAL.md.

#include <emscripten.h>
#include <emscripten/html5.h>

#include <cstring>

extern "C" {
// Forward-declared, not included from a projectM header: this file is compiled standalone
// against the projectM static libraries (see build.sh), and pulling in the full projectM-4
// include tree here would be one more thing to keep in sync with whatever checkout produced
// those libraries. The signature is pinned by docs/plans/music-visualizer-config-plan.md §A.4,
// verified against the real v4.1.7 headers when that plan was written.
typedef void* projectm_handle;
projectm_handle projectm_create();

// Creates a WebGL2 context bound to `canvas_selector` (a CSS selector, e.g. "#player-visualizer-
// canvas") and makes it current, then creates and returns a projectM instance rendering into
// that context. Returns null (0 to the JS caller, since Emscripten's number return type maps a
// null pointer to 0) if either step fails — the JS side (frontend/visualizer.js's
// VisualizerEngine.init) treats that as the source requirements doc's §17 "unavailable" path,
// same as every other failure mode there, not a special case.
EMSCRIPTEN_KEEPALIVE
projectm_handle ork_visualizer_init(const char* canvas_selector) {
  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.majorVersion = 2;
  attrs.minorVersion = 0;
  attrs.alpha = false;
  attrs.depth = false;
  attrs.stencil = false;
  attrs.antialias = false;
  attrs.premultipliedAlpha = false;
  attrs.preserveDrawingBuffer = false;

  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx =
      emscripten_webgl_create_context(canvas_selector, &attrs);
  if (ctx <= 0) {
    return nullptr;
  }
  if (emscripten_webgl_make_context_current(ctx) != EMSCRIPTEN_RESULT_SUCCESS) {
    return nullptr;
  }

  return projectm_create();
}
}  // extern "C"
