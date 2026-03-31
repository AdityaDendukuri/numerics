/// @file main.cpp
/// @brief Quantum Circuit Demo  -- interactive statevector simulator
///
/// Visualises five canonical quantum circuits with step-by-step gate animation
/// and a live probability histogram.
///
/// Controls:
///   [right / L]    Apply next gate
///   [left  / H]    Undo last gate (step back)
///   [1-5]      Load preset circuit
///   [SPACE]    Auto-step (one gate per 0.5 s)
///   [R]        Reset to |0...0>
///
/// Presets:
///   [1]  Bell State          |Phi+> = (|00>+|11>)/sqrt(2)
///   [2]  GHZ State           (|000>+|111>)/sqrt(2)
///   [3]  Grover Search       2-qubit, finds |11> in one iteration
///   [4]  Quantum Teleportation   q0 -> q2
///   [5]  QFT_3               3-qubit Quantum Fourier Transform on |001>

#include "quantum/circuit.hpp"
#include "quantum/statevector.hpp"
#include <raylib.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace num;

// Layout constants
static constexpr int WIN_W = 1280;
static constexpr int WIN_H = 720;

// Circuit panel (left)
static constexpr int CIR_X = 10;
static constexpr int CIR_Y = 80;
static constexpr int CIR_W = 720;
static constexpr int CIR_H = 500;

// Histogram panel (right)
static constexpr int HIS_X = 750;
static constexpr int HIS_Y = 80;
static constexpr int HIS_W = 510;
static constexpr int HIS_H = 500;

// Gate cell sizes (pixels)
static constexpr int GATE_W     = 64; // column pitch
static constexpr int GATE_H     = 36; // gate box height
static constexpr int WIRE_PITCH = 80; // vertical distance between qubit wires

// UCSB-dark palette
static constexpr Color COL_BG     = {0, 26, 48, 255};    // #001a30
static constexpr Color COL_PANEL  = {0, 30, 56, 255};    // slightly lighter
static constexpr Color COL_WIRE   = {80, 120, 160, 255};
static constexpr Color COL_GOLD   = {254, 188, 17, 255}; // UCSB gold
static constexpr Color COL_TEXT   = {232, 238, 244, 255};
static constexpr Color COL_DIM    = {120, 150, 180, 255};
static constexpr Color COL_ACTIVE = {255, 255, 255, 255}; // active gate border

// Gate category colours
static Color gate_color(int kind, const std::string& label) {
    if (kind == 2)
        return {180, 60, 180, 255}; // SWAP: purple
    if (kind == 3)
        return {40, 160, 160, 255}; // Toffoli/Fredkin: teal
    if (kind == 1)
        return {40, 160, 160, 255}; // controlled: teal
    // Single-qubit by name
    if (label == "H")
        return {254, 188, 17, 255}; // gold
    if (label == "X" || label == "(+)")
        return {210, 60, 60, 255};  // red
    if (label == "Y")
        return {60, 180, 60, 255};  // green
    if (label == "Z")
        return {60, 100, 210, 255}; // blue
    if (label == "S" || label == "Sdg")
        return {200, 120, 40, 255}; // orange
    if (label == "T" || label == "Tdg")
        return {200, 120, 40, 255}; // orange
    if (label[0] == 'R')
        return {140, 60, 200, 255}; // purple
    return {100, 140, 180, 255};    // default
}

// Preset circuits
struct Preset {
    std::string name;
    Circuit     circuit;
};

static std::vector<Preset> make_presets() {
    const real          pi = 3.141592653589793;
    std::vector<Preset> p;

    // 1. Bell State
    {
        Circuit c(2);
        c.h(0).cx(0, 1);
        p.push_back({"Bell State  |Phi+>", std::move(c)});
    }
    // 2. GHZ State
    {
        Circuit c(3);
        c.h(0).cx(0, 1).cx(0, 2);
        p.push_back({"GHZ State", std::move(c)});
    }
    // 3. Grover Search  -- 2 qubits, oracle marks |11>
    {
        Circuit c(2);
        c.h(0)
            .h(1)     // uniform superposition
            .cz(0, 1) // oracle: phase flip |11>
            .h(0)
            .h(1)
            .x(0)
            .x(1)
            .cz(0, 1)
            .x(0)
            .x(1)
            .h(0)
            .h(1); // diffusion
        p.push_back({"Grover Search  |11>", std::move(c)});
    }
    // 4. Quantum Teleportation  -- teleports RY(pi/3)|0> from q0 to q2
    {
        Circuit c(3);
        c.ry(pi / 3, 0) // prepare state on q0
            .h(1)
            .cx(1, 2)   // Bell pair between q1 and q2
            .cx(0, 1)
            .h(0)       // Alice's encoding
            .cx(1, 2)
            .cz(0, 2);  // Bob's corrections
        p.push_back({"Quantum Teleportation", std::move(c)});
    }
    // 5. QFT_3 on |001>
    {
        Circuit c(3);
        c.x(0) // prepare |001> (q0=1)
            .h(2)
            .cp(pi / 2, 1, 2)
            .cp(pi / 4, 0, 2)
            .h(1)
            .cp(pi / 2, 0, 1)
            .h(0)
            .swap(0, 2);
        p.push_back({"QFT\xe2\x82\x83  on |001\xe2\x9f\xa9", std::move(c)});
    }

    return p;
}

// Drawing helpers

static void draw_panel(int x, int y, int w, int h, const char* title) {
    DrawRectangle(x, y, w, h, COL_PANEL);
    DrawRectangleLinesEx({(float)x, (float)y, (float)w, (float)h}, 1, COL_WIRE);
    if (title && title[0])
        DrawText(title, x + 8, y + 6, 16, COL_DIM);
}

static void draw_gate_box(int         cx,
                          int         cy,
                          int         w,
                          int         h,
                          Color       fill,
                          const char* label,
                          bool        active) {
    int x = cx - w / 2, y = cy - h / 2;
    DrawRectangle(x, y, w, h, fill);
    if (active)
        DrawRectangleLinesEx(
            {(float)x - 1, (float)y - 1, (float)(w + 2), (float)(h + 2)},
            2,
            COL_ACTIVE);
    else
        DrawRectangleLinesEx({(float)x, (float)y, (float)w, (float)h},
                             1,
                             {0, 0, 0, 80});

    int font = (int(strlen(label)) > 2) ? 12 : 16;
    int tw   = MeasureText(label, font);
    DrawText(label, cx - tw / 2, cy - font / 2, font, COL_TEXT);
}

// Draw the circuit for the given gate views up to step `current_step`.
// active_gate is highlighted with a white border.
static void draw_circuit(const std::vector<GateView>& views,
                         int                          n_qubits,
                         int                          current_step,
                         int                          active_gate,
                         int                          origin_x,
                         int                          origin_y,
                         int                          panel_w) {
    if (views.empty())
        return;

    // Compute total columns
    int total_cols = 0;
    for (const auto& v : views)
        total_cols = std::max(total_cols, v.col + 1);

    // Wire y-positions
    auto wire_y = [&](int q) {
        return origin_y + 40 + q * WIRE_PITCH;
    };
    // Gate x-position
    auto gate_x = [&](int col) {
        return origin_x + 50 + col * GATE_W;
    };

    // Draw qubit labels and wires
    for (int q = 0; q < n_qubits; ++q) {
        int  wy = wire_y(q);
        char label[16];
        std::snprintf(label, sizeof(label), "q[%d]", q);
        DrawText(label, origin_x + 4, wy - 8, 16, COL_DIM);
        // Wire extends across all columns
        int wire_end_x = gate_x(total_cols);
        DrawLine(origin_x + 46,
                 wy,
                 std::min(wire_end_x + 10, origin_x + panel_w - 10),
                 wy,
                 COL_WIRE);
    }

    // Draw gates
    for (int i = 0; i < (int)views.size(); ++i) {
        const auto& v       = views[i];
        bool        applied = (i < current_step);
        bool        active  = (i == active_gate);
        Color       base    = gate_color(v.kind, v.label);
        if (!applied) {
            // Dim unplayed gates
            base.r = (unsigned char)(base.r * 0.4f);
            base.g = (unsigned char)(base.g * 0.4f);
            base.b = (unsigned char)(base.b * 0.4f);
        }
        int gx = gate_x(v.col);

        if (v.kind == 0) {
            // Single-qubit gate box
            draw_gate_box(gx,
                          wire_y(v.q0),
                          GATE_W - 8,
                          GATE_H,
                          base,
                          v.label.c_str(),
                          active);
        } else if (v.kind == 1) {
            // Controlled gate: control dot + vertical line + target symbol
            int cy_ctrl = wire_y(v.q0);
            int cy_tgt  = wire_y(v.q1);
            // Vertical line
            DrawLine(gx,
                     std::min(cy_ctrl, cy_tgt),
                     gx,
                     std::max(cy_ctrl, cy_tgt),
                     applied ? COL_WIRE : COL_DIM);
            // Control dot
            DrawCircle(gx, cy_ctrl, 7, applied ? base : COL_DIM);
            if (active)
                DrawCircleLines(gx, cy_ctrl, 9, COL_ACTIVE);
            // Target symbol ((+) for CX, box for CZ/CP/CY)
            const char* tgt_sym = (v.label == "CX") ? "\xe2\x8a\x95" : // (+)
                                      (v.label == "CZ") ? "Z"
                                  : (v.label == "CY")   ? "Y"
                                                        : "P";
            draw_gate_box(gx,
                          cy_tgt,
                          GATE_W - 8,
                          GATE_H,
                          base,
                          tgt_sym,
                          active);
        } else if (v.kind == 2) {
            // SWAP: two x symbols + vertical line
            int cy0 = wire_y(v.q0), cy1 = wire_y(v.q1);
            DrawLine(gx, cy0, gx, cy1, applied ? COL_WIRE : COL_DIM);
            draw_gate_box(gx,
                          cy0,
                          GATE_W - 8,
                          GATE_H,
                          base,
                          "\xc3\x97",
                          active); // x
            draw_gate_box(gx,
                          cy1,
                          GATE_W - 8,
                          GATE_H,
                          base,
                          "\xc3\x97",
                          active);
        } else {
            // 3-qubit gate (Toffoli / Fredkin)
            int c0 = wire_y(v.q0), c1 = wire_y(v.q1), c2 = wire_y(v.q2);
            int y_lo = std::min({c0, c1, c2}), y_hi = std::max({c0, c1, c2});
            DrawLine(gx, y_lo, gx, y_hi, applied ? COL_WIRE : COL_DIM);
            DrawCircle(gx, c0, 7, applied ? base : COL_DIM);
            DrawCircle(gx, c1, 7, applied ? base : COL_DIM);
            draw_gate_box(gx,
                          c2,
                          GATE_W - 8,
                          GATE_H,
                          base,
                          "\xe2\x8a\x95",
                          active);
            if (active) {
                DrawCircleLines(gx, c0, 9, COL_ACTIVE);
                DrawCircleLines(gx, c1, 9, COL_ACTIVE);
            }
        }
    }
}

// Draw probability histogram
static void draw_histogram(const quantum::Statevector& sv,
                           int                         n_qubits,
                           int                         ox,
                           int                         oy,
                           int                         panel_w,
                           int                         panel_h) {
    int  N     = 1 << n_qubits;
    auto probs = quantum::probabilities(sv);

    // Usable area
    int ax = ox + 10, aw = panel_w - 20;
    int ay = oy + 30, ah = panel_h - 80;

    int bar_w   = std::max(4, (aw - N * 2) / N);
    int spacing = aw / N;

    for (int k = 0; k < N; ++k) {
        float p     = (float)probs[k];
        int   bar_h = (int)(p * ah);
        int   bx    = ax + k * spacing;
        int   by    = ay + ah - bar_h;

        // Bar colour: gold if dominant, teal otherwise
        Color col = (p > 0.4f) ? COL_GOLD : Color{40, 160, 160, 255};
        if (p < 0.001f)
            col = {40, 60, 80, 255};

        DrawRectangle(bx, by, bar_w, bar_h, col);
        DrawRectangleLinesEx({(float)bx, (float)(ay), (float)bar_w, (float)ah},
                             1,
                             {60, 80, 100, 120});

        // Label below
        std::string lbl(n_qubits, '0');
        for (int b = 0; b < n_qubits; ++b)
            lbl[n_qubits - 1 - b] = ((k >> b) & 1) ? '1' : '0';
        int font = (n_qubits <= 3) ? 14 : 11;
        DrawText(("|" + lbl + "\xe2\x9f\xa9").c_str(),
                 bx,
                 ay + ah + 4,
                 font,
                 COL_DIM);

        // Probability text above bar
        if (p > 0.02f) {
            char pct[8];
            std::snprintf(pct, sizeof(pct), "%3.0f%%", p * 100.f);
            DrawText(pct, bx, by - 18, 12, COL_TEXT);
        }
    }
}

// main

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(WIN_W, WIN_H, "Quantum Circuit Demo  -- numerics");
    SetTargetFPS(60);

    auto         presets       = make_presets();
    int          preset_idx    = 0;
    int          step          = 0; // gates applied so far
    bool         auto_step     = false;
    double       auto_timer    = 0.0;
    const double AUTO_INTERVAL = 0.55;

    auto load_preset = [&](int idx) {
        preset_idx = idx;
        step       = 0;
        auto_step  = false;
    };

    load_preset(0);

    while (!WindowShouldClose()) {
        // Input
        const Circuit& circ     = presets[preset_idx].circuit;
        int            n_gates  = circ.n_gates();
        int            n_qubits = circ.n_qubits();

        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_L))
            step = std::min(step + 1, n_gates);
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_H))
            step = std::max(step - 1, 0);
        if (IsKeyPressed(KEY_R))
            step = 0;
        if (IsKeyPressed(KEY_SPACE))
            auto_step = !auto_step;

        for (int i = 0; i < 5; ++i)
            if (IsKeyPressed(KEY_ONE + i))
                load_preset(i);

        if (auto_step) {
            auto_timer += GetFrameTime();
            if (auto_timer >= AUTO_INTERVAL) {
                auto_timer = 0.0;
                if (step < n_gates)
                    ++step;
                else
                    auto_step = false;
            }
        }

        // State
        auto sv     = circ.statevector_at(step);
        auto views  = circ.views();
        int  active = (step > 0 && step <= n_gates) ? step - 1 : -1;

        // Draw
        BeginDrawing();
        ClearBackground(COL_BG);

        // Title bar
        DrawText("QUANTUM CIRCUIT DEMO", 12, 12, 22, COL_GOLD);
        DrawText(presets[preset_idx].name.c_str(), 12, 40, 18, COL_TEXT);

        // Step counter (top right)
        char step_buf[40];
        std::snprintf(step_buf,
                      sizeof(step_buf),
                      "STEP %d / %d",
                      step,
                      n_gates);
        int sw = MeasureText(step_buf, 18);
        DrawText(step_buf,
                 WIN_W - sw - 12,
                 40,
                 18,
                 (step == n_gates) ? COL_GOLD : COL_DIM);

        // Auto indicator
        if (auto_step)
            DrawText("[AUTO]", WIN_W - 90, 12, 16, {100, 210, 100, 255});

        // Circuit panel
        draw_panel(CIR_X, CIR_Y, CIR_W, CIR_H, "Circuit");
        draw_circuit(views,
                     n_qubits,
                     step,
                     active,
                     CIR_X + 4,
                     CIR_Y + 20,
                     CIR_W - 8);

        // Histogram panel
        draw_panel(HIS_X, HIS_Y, HIS_W, HIS_H, "Measurement Probabilities");
        draw_histogram(sv,
                       n_qubits,
                       HIS_X + 4,
                       HIS_Y + 20,
                       HIS_W - 8,
                       HIS_H - 30);

        // Stats strip (below circuit panel)
        int sy = CIR_Y + CIR_H + 10;
        DrawRectangle(CIR_X, sy, CIR_W, 80, COL_PANEL);
        DrawRectangleLinesEx({(float)CIR_X, (float)sy, (float)CIR_W, 80},
                             1,
                             COL_WIRE);

        // Most likely state
        auto        probs  = quantum::probabilities(sv);
        int         best_k = (int)(std::max_element(probs.begin(), probs.end())
                           - probs.begin());
        std::string best_lbl(n_qubits, '0');
        for (int b = 0; b < n_qubits; ++b)
            best_lbl[n_qubits - 1 - b] = ((best_k >> b) & 1) ? '1' : '0';
        char stat_buf[64];
        std::snprintf(stat_buf,
                      sizeof(stat_buf),
                      "Most likely: |%s\xe2\x9f\xa9  (%.1f%%)", // >
                      best_lbl.c_str(),
                      probs[best_k] * 100.0);
        DrawText(stat_buf, CIR_X + 10, sy + 8, 16, COL_TEXT);

        // Entanglement entropy of qubit 0
        real S = quantum::entanglement_entropy(sv, n_qubits, 0);
        char ent_buf[48];
        std::snprintf(ent_buf,
                      sizeof(ent_buf),
                      "S(q[0]) = %.3f ebit%s",
                      S,
                      S > 0.999 ? "  (max entangled)" : "");
        DrawText(ent_buf,
                 CIR_X + 10,
                 sy + 30,
                 16,
                 S > 0.5 ? COL_GOLD : COL_DIM);

        // Norm (sanity check)
        real nrm = quantum::norm(sv);
        char nrm_buf[32];
        std::snprintf(nrm_buf, sizeof(nrm_buf), "|\u03c8| = %.6f", nrm);
        DrawText(nrm_buf, CIR_X + 10, sy + 52, 14, COL_DIM);

        // Stats strip (below histogram panel)
        DrawRectangle(HIS_X, sy, HIS_W, 80, COL_PANEL);
        DrawRectangleLinesEx({(float)HIS_X, (float)sy, (float)HIS_W, 80},
                             1,
                             COL_WIRE);
        DrawText("All amplitudes", HIS_X + 10, sy + 8, 14, COL_DIM);

        int N   = 1 << n_qubits;
        int row = 0;
        for (int k = 0; k < N && row < 3; ++k) {
            if (std::abs(sv[k]) < 0.005)
                continue;
            std::string lbl(n_qubits, '0');
            for (int b = 0; b < n_qubits; ++b)
                lbl[n_qubits - 1 - b] = ((k >> b) & 1) ? '1' : '0';
            char amp[64];
            std::snprintf(amp,
                          sizeof(amp),
                          "|%s\xe2\x9f\xa9 %+.3f%+.3fi",
                          lbl.c_str(),
                          (float)sv[k].real(),
                          (float)sv[k].imag());
            DrawText(amp, HIS_X + 10, sy + 24 + row * 18, 14, COL_TEXT);
            ++row;
        }
        if (row == 3 && N > 4)
            DrawText("...", HIS_X + 10, sy + 62, 14, COL_DIM);

        // Controls bar (bottom)
        int ctrl_y = WIN_H - 38;
        DrawRectangle(0, ctrl_y, WIN_W, 38, {0, 14, 28, 255});
        DrawLine(0, ctrl_y, WIN_W, ctrl_y, COL_WIRE);
        DrawText("[<-/->] Step   [SPACE] Auto   [R] Reset   [1-5] Load Circuit",
                 12,
                 ctrl_y + 10,
                 16,
                 COL_DIM);

        // Preset selector highlights
        for (int i = 0; i < 5; ++i) {
            char key[4];
            std::snprintf(key, sizeof(key), "[%d]", i + 1);
            int kx = WIN_W - 300 + i * 56;
            DrawText(key,
                     kx,
                     ctrl_y + 10,
                     16,
                     (i == preset_idx) ? COL_GOLD : COL_DIM);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
