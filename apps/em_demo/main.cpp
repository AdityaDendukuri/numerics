/// @file apps/em_demo/main.cpp
/// @brief Electromagnetic field demo -- aluminum rod + permanent magnet.
///
/// Physics solved once at startup (CG Poisson solvers):
///   Electric: div(sigma * grad(phi)) = 0  ->  J = sigma * (-grad phi)
///   Magnetic: Laplacian(A) = -mu0 * J     ->  B = curl(A)
/// Magnet dipole field added analytically each frame.
/// Camera orbits the scene; ESC quits.

#include "em_demo/sim.hpp"
#include "viz/viz.hpp"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <vector>

// Compile-time alias (zero runtime overhead):
//   static_cast<float>(x)  →  cast<float>(x)
template<class To, class From>
constexpr To cast(From x) {
    return static_cast<To>(x);
}

static constexpr int   NX = 32, NY = 32, NZ = 32;
static constexpr float DX         = 0.05f;
static constexpr float SIM_DOMAIN = NX * DX; // 1.6 m

static constexpr int   ROD_CX  = NX / 2;
static constexpr int   ROD_CZ  = NZ / 2;
static constexpr int   ROD_R   = 4;
static constexpr float ROD_SIG = 1.0f;
static constexpr float BG_SIG  = 1e-6f;
static constexpr float V_TOP   = 1.0f;
static constexpr float V_BOT   = 0.0f;

static constexpr float MAG_M0    = 0.01f; // dipole moment [A*m^2]
static constexpr float MAG_VIS_R = 0.07f; // drawn sphere radius
static constexpr int   STRIDE    = 4;     // sample every N cells
static constexpr float MAX_LEN   = DX * 2.2f;
static constexpr int   WIN       = 900;

static bool inside_rod(int i, int k) {
    int di = i - ROD_CX, dk = k - ROD_CZ;
    return di * di + dk * dk <= ROD_R * ROD_R;
}
static int flat(int i, int j, int k) {
    return k * NY * NX + j * NX + i;
}

// Dipole field (m = M0 * y_hat)
static void dipole_B(float  wx,
                     float  wy,
                     float  wz,
                     float  mx,
                     float  my,
                     float  mz,
                     float& bx,
                     float& by,
                     float& bz) {
    float rx = wx - mx, ry = wy - my, rz = wz - mz;
    float r2 = rx * rx + ry * ry + rz * rz;
    if (r2 < 1e-5f) {
        bx = by = bz = 0.0f;
        return;
    }
    float           r5  = r2 * r2 * sqrtf(r2);
    float           mdr = MAG_M0 * ry;
    constexpr float C   = 1e-7f;
    bx                  = C * 3.0f * mdr * rx / r5;
    by                  = C * MAG_M0 * (3.0f * ry * ry - r2) / r5;
    bz                  = C * 3.0f * mdr * rz / r5;
}

int main() {
    // Solve EM fields once (CPU, no raylib needed yet)
    physics::ScalarField3D sigma(NX, NY, NZ, DX);
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                sigma.grid().set(i, j, k, inside_rod(i, k) ? ROD_SIG : BG_SIG);

    std::vector<physics::ElectrodeBC> bcs;
    for (int k = 0; k < NZ; ++k)
        for (int i = 0; i < NX; ++i) {
            if (!inside_rod(i, k))
                continue;
            bcs.push_back({flat(i, 0, k), V_BOT});
            bcs.push_back({flat(i, NY - 1, k), V_TOP});
        }

    printf("Solving EM fields... ");
    fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();

    physics::ScalarField3D phi(NX, NY, NZ, DX);
    auto r_phi = physics::ElectricSolver::solve_potential(phi, sigma, bcs);

    physics::VectorField3D J = physics::MagneticSolver::current_density(sigma,
                                                                        phi);
    physics::VectorField3D B = physics::MagneticSolver::solve_magnetic_field(J);

    double solve_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    printf("done in %.0f ms\n", solve_ms);

    // Magnet position
    const float mag_x = ROD_CX * DX + 0.45f;
    const float mag_y = SIM_DOMAIN * 0.5f;
    const float mag_z = ROD_CZ * DX;

    // Orbiting camera angle
    float azimuth = 0.785f;

    num::viz::run(
        "EM: Aluminum Rod + Magnet",
        WIN,
        WIN,
        [&](num::viz::Frame& f) {
            azimuth += 2.0f * 3.14159265f
                       / (60.0f * 12.0f); // full orbit in ~12 s

            const float elev = 0.524f;    // 30 degrees
            const float dist = SIM_DOMAIN * 2.6f;
            float cx = SIM_DOMAIN * 0.5f + dist * cosf(elev) * sinf(azimuth);
            float cy = SIM_DOMAIN * 0.5f + dist * sinf(elev);
            float cz = SIM_DOMAIN * 0.5f + dist * cosf(elev) * cosf(azimuth);

            // B_max this frame (rod + dipole)
            float B_max = 1e-30f;
            for (int j = 0; j < NY; j += STRIDE)
                for (int k = 0; k < NZ; k += STRIDE)
                    for (int i = 0; i < NX; i += STRIDE) {
                        float bx = cast<float>(B.x.grid()(i, j, k));
                        float by = cast<float>(B.y.grid()(i, j, k));
                        float bz = cast<float>(B.z.grid()(i, j, k));
                        float dbx, dby, dbz;
                        dipole_B(i * DX,
                                 j * DX,
                                 k * DX,
                                 mag_x,
                                 mag_y,
                                 mag_z,
                                 dbx,
                                 dby,
                                 dbz);
                        float mag = sqrtf((bx + dbx) * (bx + dbx)
                                          + (by + dby) * (by + dby)
                                          + (bz + dbz) * (bz + dbz));
                        if (mag > B_max)
                            B_max = mag;
                    }

            f.begin3d({cx,
                       cy,
                       cz,
                       SIM_DOMAIN * 0.5f,
                       SIM_DOMAIN * 0.5f,
                       SIM_DOMAIN * 0.5f});

            // Domain wireframe
            DrawCubeWires(
                {SIM_DOMAIN * 0.5f, SIM_DOMAIN * 0.5f, SIM_DOMAIN * 0.5f},
                SIM_DOMAIN,
                SIM_DOMAIN,
                SIM_DOMAIN,
                {60, 60, 80, 255});

            // Aluminum rod
            float rod_wr = ROD_R * DX;
            DrawCylinder(
                {cast<float>(ROD_CX) * DX, 0.0f, cast<float>(ROD_CZ) * DX},
                rod_wr,
                rod_wr,
                SIM_DOMAIN,
                24,
                {180, 185, 195, 220});
            DrawCylinderWires(
                {cast<float>(ROD_CX) * DX, 0.0f, cast<float>(ROD_CZ) * DX},
                rod_wr,
                rod_wr,
                SIM_DOMAIN,
                24,
                {100, 110, 130, 255});
            DrawCylinder({cast<float>(ROD_CX) * DX,
                          SIM_DOMAIN - DX * 0.5f,
                          cast<float>(ROD_CZ) * DX},
                         rod_wr,
                         rod_wr,
                         DX * 0.5f,
                         24,
                         {220, 60, 60, 255});
            DrawCylinder(
                {cast<float>(ROD_CX) * DX, 0.0f, cast<float>(ROD_CZ) * DX},
                rod_wr,
                rod_wr,
                DX * 0.5f,
                24,
                {60, 80, 220, 255});

            // Magnet (N/S poles)
            f.sphere3d(mag_x,
                       mag_y + MAG_VIS_R * 0.5f,
                       mag_z,
                       MAG_VIS_R,
                       {210, 60, 60, 255});
            f.sphere3d(mag_x,
                       mag_y - MAG_VIS_R * 0.5f,
                       mag_z,
                       MAG_VIS_R,
                       {60, 80, 210, 255});
            f.line3d(mag_x,
                     mag_y - MAG_VIS_R * 1.6f,
                     mag_z,
                     mag_x,
                     mag_y + MAG_VIS_R * 1.6f,
                     mag_z,
                     num::viz::kWhite);

            // B field arrows
            constexpr float B_THRESH = 0.002f;
            for (int j = 0; j < NY; j += STRIDE)
                for (int k = 0; k < NZ; k += STRIDE)
                    for (int i = 0; i < NX; i += STRIDE) {
                        float wx = i * DX, wy = j * DX, wz = k * DX;
                        float bx = cast<float>(B.x.grid()(i, j, k));
                        float by = cast<float>(B.y.grid()(i, j, k));
                        float bz = cast<float>(B.z.grid()(i, j, k));
                        float dbx, dby, dbz;
                        dipole_B(wx,
                                 wy,
                                 wz,
                                 mag_x,
                                 mag_y,
                                 mag_z,
                                 dbx,
                                 dby,
                                 dbz);
                        bx += dbx;
                        by += dby;
                        bz += dbz;
                        float mag = sqrtf(bx * bx + by * by + bz * bz);
                        float t   = mag / B_max;
                        if (t < B_THRESH)
                            continue;
                        float t_log = logf(1.0f + (t / B_THRESH - 1.0f))
                                      / logf(1.0f + (1.0f / B_THRESH - 1.0f));
                        float len = MAX_LEN * (0.15f + 0.85f * t_log);
                        auto  col = num::viz::heat_color(t);
                        f.line3d(wx,
                                 wy,
                                 wz,
                                 wx + bx / mag * len,
                                 wy + by / mag * len,
                                 wz + bz / mag * len,
                                 col);
                        f.sphere3d(wx + bx / mag * len,
                                   wy + by / mag * len,
                                   wz + bz / mag * len,
                                   len * 0.12f,
                                   col);
                    }

            f.end3d();

            // HUD
            f.rect(8, 8, 380, 62, {0, 0, 0, 160});
            f.text("Aluminum Rod + Permanent Magnet",
                   16,
                   14,
                   16,
                   num::viz::kWhite);
            f.textf(16,
                    36,
                    13,
                    num::viz::kGray,
                    "solve %.0f ms | iters=%zu res=%.1e",
                    solve_ms,
                    r_phi.iterations,
                    r_phi.residual);
            f.textf(16,
                    52,
                    13,
                    num::viz::kGray,
                    "B_max = %.3e T",
                    cast<double>(B_max));
        },
        {15, 15, 20, 255});
}
