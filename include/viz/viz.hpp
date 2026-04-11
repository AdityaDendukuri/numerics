/// @file include/viz/viz.hpp
/// @brief Lightweight real-time visualization layer for numerics apps.
///
/// Wraps raylib behind a single draw-callback interface.  Every app has the
/// same shape:
///
/// @code
///   #include "viz.hpp"
///
///   num::viz::run("My Sim", 900, 900, [&](num::viz::Frame& f) {
///       f.step([&]{ sim.step(dt); });          // SPACE pauses, +/- adjusts
///       substeps for (int i = 0; i < n; ++i)
///           f.dot(x[i], y[i], heat_color(T[i]));
///   });
/// @endcode
///
/// This header is NOT included by numerics.hpp -- it depends on raylib which
/// is an optional app-layer dependency.  Include it directly from your app.
///
/// ## Controls (built-in)
///   SPACE       pause / unpause
///   R           reset (check f.reset_pressed() in your callback)
///   +/-         increase / decrease substeps
///   ESC         quit
#pragma once

#include <raylib.h>
#include <raymath.h>
#include <algorithm>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <vector>

namespace num::viz {

// RGBA color  (identical byte layout to raylib ::Color -- cast-safe)
struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};

// Common presets
inline constexpr Color kBlack   = {0, 0, 0, 255};
inline constexpr Color kWhite   = {255, 255, 255, 255};
inline constexpr Color kRed     = {220, 40, 40, 255};
inline constexpr Color kGreen   = {50, 200, 50, 255};
inline constexpr Color kBlue    = {50, 100, 230, 255};
inline constexpr Color kYellow  = {240, 220, 30, 255};
inline constexpr Color kGray    = {120, 120, 120, 255};
inline constexpr Color kSkyBlue = {100, 180, 240, 255};

// Convert viz::Color <-> raylib ::Color
inline ::Color to_rl(Color c) {
    return {c.r, c.g, c.b, c.a};
}
inline Color from_rl(::Color c) {
    return {c.r, c.g, c.b, c.a};
}

// Unpack a packed ARGB uint32 (e.g. nbody Body::color) into viz::Color
inline Color unpack(uint32_t c) {
    return {uint8_t(c >> 24), uint8_t(c >> 16), uint8_t(c >> 8), uint8_t(c)};
}

// Colormaps (all return viz::Color; inputs are normalized unless noted)

/// t in [0,1]: blue -> cyan -> yellow -> red.
inline Color heat_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return from_rl(ColorFromHSV((1.0f - t) * 240.0f, 1.0f, 0.95f));
}

/// t in [-1,1]: deep blue -> black -> deep red.
inline Color diverging_color(float t) {
    t = std::clamp(t, -1.0f, 1.0f);
    if (t >= 0.0f) {
        auto v = uint8_t(255 * t * t);
        return {v, 0, uint8_t(v / 10), 255};
    } else {
        auto v = uint8_t(255 * t * t);
        return {uint8_t(v / 10), 0, v, 255};
    }
}

/// Quantum wavefunction: hue = phase, brightness = sqrt(|psi|^2 / max).
inline Color phase_hsv_color(double prob, double phase, double max_prob) {
    if (max_prob < 1e-20)
        return kBlack;
    float amp = std::min(1.0f, float(std::sqrt(prob / max_prob)));
    float hue = float((phase + 3.14159265) / (2.0 * 3.14159265)) * 360.0f;
    float h6  = hue / 60.0f;
    int   hi  = int(h6) % 6;
    float f   = h6 - int(h6);
    float s   = 0.95f;
    float p   = amp * (1.0f - s);
    float q   = amp * (1.0f - s * f);
    float u   = amp * (1.0f - s * (1.0f - f));
    float r, g, b;
    switch (hi) {
        case 0:
            r = amp;
            g = u;
            b = p;
            break;
        case 1:
            r = q;
            g = amp;
            b = p;
            break;
        case 2:
            r = p;
            g = amp;
            b = u;
            break;
        case 3:
            r = p;
            g = q;
            b = amp;
            break;
        case 4:
            r = u;
            g = p;
            b = amp;
            break;
        default:
            r = amp;
            g = p;
            b = q;
            break;
    }
    return {uint8_t(r * 255), uint8_t(g * 255), uint8_t(b * 255), 255};
}

/// Linear blend between two colors.
inline Color lerp_color(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return {
        uint8_t(a.r + t * (b.r - a.r)),
        uint8_t(a.g + t * (b.g - a.g)),
        uint8_t(a.b + t * (b.b - a.b)),
        uint8_t(a.a + t * (b.a - a.a)),
    };
}

// Internal state managed by run()
namespace detail {
struct FieldCanvas {
    Texture2D            tex{};
    std::vector<::Color> buf;
    int                  N     = 0;
    bool                 valid = false;

    void ensure(int n) {
        if (n == N && valid)
            return;
        if (valid)
            UnloadTexture(tex);
        N = n;
        buf.assign(n * n, ::BLACK);
        Image img = GenImageColor(n, n, ::BLACK);
        tex       = LoadTextureFromImage(img);
        UnloadImage(img);
        valid = true;
    }
    void unload() {
        if (valid) {
            UnloadTexture(tex);
            valid = false;
            N     = 0;
        }
    }
};

struct State {
    FieldCanvas canvas;
    bool        paused       = false;
    int         substeps     = 1;
    int         slider_count = 0; // reset each frame
};
} // namespace detail

// Frame -- passed to the draw callback on every tick.
struct Frame {
    int            width, height;
    bool&          paused;
    int&           substeps;
    detail::State& _s;

    // Advance the simulation fn() `substeps` times if not paused.
    // fn() -> void
    template<class Fn>
    void step(Fn fn) {
        if (!paused)
            for (int i = 0; i < substeps; ++i)
                fn();
    }

    // True on the frame R was pressed.
    bool reset_pressed() const {
        return IsKeyPressed(KEY_R);
    }

    // 2D primitives -- pixel coordinates, (0,0) = top-left.
    void dot(float x, float y, Color c, float r = 3.0f) {
        DrawCircleV({x, y}, r, to_rl(c));
    }
    void circle(float x, float y, float r, Color c) {
        DrawCircleLines(int(x), int(y), r, to_rl(c));
    }
    void line(float x0,
              float y0,
              float x1,
              float y1,
              Color c,
              float thick = 1.0f) {
        DrawLineEx({x0, y0}, {x1, y1}, thick, to_rl(c));
    }
    void rect(float x, float y, float w, float h, Color c) {
        DrawRectangleV({x, y}, {w, h}, to_rl(c));
    }
    void rect_outline(float x,
                      float y,
                      float w,
                      float h,
                      Color c,
                      float thick = 1.0f) {
        DrawRectangleLinesEx({x, y, w, h}, thick, to_rl(c));
    }
    void text(const char* s, float x, float y, int sz, Color c) {
        DrawText(s, int(x), int(y), sz, to_rl(c));
    }
    // printf-style text -- uses a 256-byte internal buffer (safe for HUD
    // strings)
    void textf(float x, float y, int sz, Color c, const char* fmt, ...) {
        char    buf[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        DrawText(buf, int(x), int(y), sz, to_rl(c));
    }

    // Pixel field -- fills the entire window with an N×N colored grid.
    // color_fn(col, row) -> Color    (col = x-axis, row = y-axis, top-left
    // origin)
    template<class Fn>
    void field(int N, Fn color_fn) {
        auto& cv = _s.canvas;
        cv.ensure(N);
        for (int row = 0; row < N; ++row)
            for (int col = 0; col < N; ++col)
                cv.buf[row * N + col] = to_rl(color_fn(col, row));
        UpdateTexture(cv.tex, cv.buf.data());
        DrawTexturePro(cv.tex,
                       {0, 0, float(N), float(N)},
                       {0, 0, float(width), float(height)},
                       {0, 0},
                       0.0f,
                       ::WHITE);
    }

    // GUI slider -- stacks vertically in the top-left corner.
    // Reads mouse drag to update val.
    void slider(const char* label, double lo, double hi, double& val) {
        constexpr int PAD    = 10;
        constexpr int SLOT_H = 38;
        constexpr int BAR_W  = 220;
        constexpr int BAR_H  = 8;
        const int     X      = PAD;
        const int     Y      = PAD + _s.slider_count * SLOT_H;
        ++_s.slider_count;

        DrawRectangle(X - 4, Y - 2, BAR_W + 8, SLOT_H - 4, {0, 0, 0, 170});
        DrawRectangle(X, Y + 18, BAR_W, BAR_H, {80, 80, 80, 220});

        float t  = std::clamp(float((val - lo) / (hi - lo)), 0.0f, 1.0f);
        int   tx = X + int(t * BAR_W);

        Vector2   mp  = GetMousePosition();
        Rectangle hit = {float(X), float(Y), float(BAR_W), float(SLOT_H)};
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)
            && CheckCollisionPointRec(mp, hit)) {
            t   = std::clamp((mp.x - X) / BAR_W, 0.0f, 1.0f);
            val = lo + t * (hi - lo);
            tx  = X + int(t * BAR_W);
        }

        DrawCircle(tx, Y + 18 + BAR_H / 2, 9, {200, 200, 200, 240});
        DrawText(TextFormat("%s: %.3g", label, val),
                 X,
                 Y + 2,
                 13,
                 {210, 210, 210, 230});
    }

    // 3D camera descriptor
    struct Cam3 {
        float px = 0, py = 0, pz = 0; // camera position
        float tx = 0, ty = 0, tz = 0; // look-at target
        float ux = 0, uy = 1, uz = 0; // up vector (default: world-Y)
        float fovy = 45.0f;
    };

    // Begin 3D rendering mode.  All 3D draw calls must sit between
    // begin3d/end3d.
    void begin3d(Cam3 cam) {
        Camera3D c{};
        c.position   = {cam.px, cam.py, cam.pz};
        c.target     = {cam.tx, cam.ty, cam.tz};
        c.up         = {cam.ux, cam.uy, cam.uz};
        c.fovy       = cam.fovy;
        c.projection = CAMERA_PERSPECTIVE;
        BeginMode3D(c);
    }
    void end3d() {
        EndMode3D();
    }

    void sphere3d(float x, float y, float z, float r, Color c) {
        DrawSphere({x, y, z}, r, to_rl(c));
    }
    void sphere3d_wire(float x, float y, float z, float r, Color c) {
        DrawSphereWires({x, y, z}, r, 6, 6, to_rl(c));
    }
    void line3d(float x0,
                float y0,
                float z0,
                float x1,
                float y1,
                float z1,
                Color c) {
        DrawLine3D({x0, y0, z0}, {x1, y1, z1}, to_rl(c));
    }
    void
    cube3d(float x, float y, float z, float sx, float sy, float sz, Color c) {
        DrawCube({x, y, z}, sx, sy, sz, to_rl(c));
    }
};

// run() -- open a window and call draw(Frame&) at 60 Hz until ESC / close.
//
// Built-in key bindings:
//   SPACE  pause / unpause
//   R      sets reset flag (check via f.reset_pressed())
//   +/-    increase / decrease substeps (capped at 16)
//   ESC    quit
template<class DrawFn>
void run(const char* title,
         int         w,
         int         h,
         DrawFn      draw,
         Color       bg = {15, 15, 15, 255}) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(w, h, title);
    SetTargetFPS(60);

    detail::State state;
    Frame         frame{w, h, state.paused, state.substeps, state};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE))
            state.paused = !state.paused;
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))
            state.substeps = std::min(state.substeps + 1, 16);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT))
            state.substeps = std::max(state.substeps - 1, 1);

        state.slider_count = 0;

        BeginDrawing();
        ClearBackground(to_rl(bg));

        draw(frame);

        if (state.paused) {
            DrawRectangle(w / 2 - 70, h / 2 - 18, 140, 36, {0, 0, 0, 160});
            DrawText("PAUSED", w / 2 - 46, h / 2 - 10, 24, ::YELLOW);
        }

        EndDrawing();
    }

    state.canvas.unload();
    CloseWindow();
}

} // namespace num::viz
