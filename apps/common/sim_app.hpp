/// @file apps/common/sim_app.hpp
/// @brief Minimal simulation app harness for raylib-based demos.
///
/// Eliminates the repeated window-lifecycle and pause/substep boilerplate
/// found in every simulation app:
///
///   bool paused = false; int substeps = 1;
///   InitWindow(W, H, title); SetTargetFPS(60);
///   while (!WindowShouldClose()) {
///       if (IsKeyPressed(KEY_SPACE)) paused = !paused;
///       if (IsKeyPressed(KEY_EQUAL)) substeps = min(substeps+1, 16);
///       if (IsKeyPressed(KEY_MINUS)) substeps = max(substeps-1, 1);
///       BeginDrawing(); ClearBackground(bg); ... EndDrawing();
///   }
///   CloseWindow();
///
/// Usage:
/// @code
///   run_sim("My Sim", 900, 900, 60, BLACK, [&](SimControls& ctrl) {
///       // App-specific input (ctrl.paused / ctrl.substeps already set)
///       if (ctrl.reset_pressed) my_solver.init();
///
///       // Step (no-op when paused)
///       ctrl.step([&]() { my_solver.step(); });
///
///       // Draw (already inside BeginDrawing / EndDrawing)
///       canvas.fill([&](int i, int j) { return my_color(i, j); });
///       canvas.draw();
///       ctrl.draw_paused_overlay(900, 900);
///   });
/// @endcode
#pragma once

#include <raylib.h>
#include <algorithm>

// SimControls

/// Per-frame simulation state managed by run_sim.
/// Passed to the user callback every frame.
struct SimControls {
    bool paused        = false;  ///< True when simulation is paused (SPACE)
    int  substeps      = 1;      ///< Substeps per frame (+/- to adjust)
    int  max_substeps  = 16;     ///< Upper limit for substeps
    bool reset_pressed = false;  ///< True only on the frame R was pressed

    /// Handle the standard keys: SPACE=pause, R=reset, +/-=substeps.
    /// Called automatically by run_sim before each frame.
    void handle_keys() {
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))
            substeps = std::min(substeps + 1, max_substeps);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT))
            substeps = std::max(substeps - 1, 1);
        reset_pressed = IsKeyPressed(KEY_R);
    }

    /// Advance the simulation fn() substeps times.
    /// Does nothing when paused.
    template<typename Fn>
    void step(Fn fn) {
        if (!paused)
            for (int s = 0; s < substeps; ++s) fn();
    }

    /// Draw a centred "PAUSED" banner.  Call at the end of your draw block.
    void draw_paused_overlay(int win_w, int win_h) const {
        if (paused) {
            DrawRectangle(win_w / 2 - 70, win_h / 2 - 18, 140, 36, {0, 0, 0, 160});
            DrawText("PAUSED", win_w / 2 - 46, win_h / 2 - 10, 24, YELLOW);
        }
    }

    /// Draw a one-line controls footer at the bottom of the window.
    /// @param extra  App-specific controls string appended after the defaults.
    void draw_footer(int win_w, int win_h, const char* extra = "") const {
        const char* base = "SPACE pause  R reset  +/- speed";
        DrawRectangle(0, win_h - 26, win_w, 26, {0, 0, 0, 160});
        DrawText(TextFormat("%s  %s", base, extra), 8, win_h - 19, 13, {160, 160, 160, 220});
    }
};

// run_sim

/// Open a window, run the simulation loop, close the window.
///
/// on_frame(SimControls&) is called every frame, already inside
/// BeginDrawing / ClearBackground / EndDrawing.  It should:
///   1. Handle app-specific input (SPACE / R / +/- are pre-handled)
///   2. Call ctrl.step([&]() { solver.step(); }) to advance the simulation
///   3. Issue raylib draw calls
///
/// @param title      Window title
/// @param win_w      Window width in pixels
/// @param win_h      Window height in pixels
/// @param fps        Target frame rate
/// @param bg         Background clear colour
/// @param on_frame   Per-frame callback  (SimControls& -> void)
template<typename Fn>
void run_sim(const char* title, int win_w, int win_h, int fps,
             Color bg, Fn on_frame)
{
    InitWindow(win_w, win_h, title);
    SetTargetFPS(fps);

    SimControls ctrl;
    while (!WindowShouldClose()) {
        ctrl.handle_keys();
        BeginDrawing();
        ClearBackground(bg);
        on_frame(ctrl);
        EndDrawing();
    }

    CloseWindow();
}

/// Variant with an on_init callback invoked after InitWindow but before the loop.
/// Use this when setup requires an active OpenGL context (e.g. creating PixelCanvas).
///
/// @param on_init    Called once after InitWindow()  (() -> void)
/// @param on_frame   Per-frame callback  (SimControls& -> void)
template<typename InitFn, typename FrameFn>
void run_sim(const char* title, int win_w, int win_h, int fps,
             Color bg, InitFn on_init, FrameFn on_frame)
{
    InitWindow(win_w, win_h, title);
    SetTargetFPS(fps);
    on_init();

    SimControls ctrl;
    while (!WindowShouldClose()) {
        ctrl.handle_keys();
        BeginDrawing();
        ClearBackground(bg);
        on_frame(ctrl);
        EndDrawing();
    }

    CloseWindow();
}
