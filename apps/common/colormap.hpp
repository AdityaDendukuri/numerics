/// @file apps/common/colormap.hpp
/// @brief Colour-mapping utilities shared across simulation apps.
///
/// All functions take normalised float inputs and return a raylib Color.
/// No state, no allocation -- every function is a pure inline mapping.
///
/// Available maps:
///   heat_color(t)          -- t in [0,1]:  blue -> cyan -> yellow -> red
///   diverging_color(t)     -- t in [-1,1]: blue -> black -> red  (vorticity)
///   phase_hsv_color(...)   -- quantum phase with amplitude brightness
///   density_color(...)     -- probability density: black -> cyan -> white
///   lerp_color(a, b, t)    -- linear blend between two Colors
#pragma once

#include <raylib.h>
#include <cmath>
#include <algorithm>

// Heat map

/// t in [0,1]: blue (240 deg) -> cyan -> green -> yellow -> red (0 deg).
/// Widely applicable: temperature fields, magnetic field magnitude, speed.
inline Color heat_color(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return ColorFromHSV((1.0f - t) * 240.0f, 1.0f, 0.95f);
}

// Diverging map

/// t in [-1,1]: deep blue -> black -> deep red.
/// Designed for signed fields (vorticity, charge density, magnetisation).
inline Color diverging_color(float t) {
    t = std::clamp(t, -1.0f, 1.0f);
    if (t >= 0.0f) {
        auto v = static_cast<unsigned char>(255 * t * t);
        return {v, 0, static_cast<unsigned char>(v / 10), 255};
    } else {
        auto v = static_cast<unsigned char>(255 * t * t);
        return {static_cast<unsigned char>(v / 10), 0, v, 255};
    }
}

// Quantum phase / amplitude

/// HSV colour for a complex wavefunction: hue = phase, brightness = amplitude.
///
/// @param prob      |psi|^2 at this point
/// @param phase     arg(psi) in [-pi, pi]
/// @param max_prob  maximum |psi|^2 in the domain (for normalisation)
inline Color phase_hsv_color(double prob, double phase, double max_prob) {
    if (max_prob < 1e-20)
        return {0, 0, 0, 255};
    float amp = std::min(1.0f, static_cast<float>(std::sqrt(prob / max_prob)));
    float hue = static_cast<float>((phase + M_PI) / (2.0 * M_PI)) * 360.0f;

    // Manual HSV -> RGB (avoids calling ColorFromHSV twice for the same math)
    float h6 = hue / 60.0f;
    int   hi = static_cast<int>(h6) % 6;
    float f  = h6 - static_cast<int>(h6);
    float s  = 0.95f;
    float p  = amp * (1.0f - s);
    float q  = amp * (1.0f - s * f);
    float t_ = amp * (1.0f - s * (1.0f - f));
    float r, g, b;
    switch (hi) {
        case 0:
            r = amp;
            g = t_;
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
            b = t_;
            break;
        case 3:
            r = p;
            g = q;
            b = amp;
            break;
        case 4:
            r = t_;
            g = p;
            b = amp;
            break;
        default:
            r = amp;
            g = p;
            b = q;
            break;
    }
    return {static_cast<unsigned char>(r * 255),
            static_cast<unsigned char>(g * 255),
            static_cast<unsigned char>(b * 255),
            255};
}

// Probability density

/// t = prob / max_prob in [0,1]: black -> teal -> white.
/// Gamma-corrected for better perceptual range on low-amplitude tails.
inline Color density_color(double prob, double max_prob) {
    if (max_prob < 1e-20)
        return {0, 0, 0, 255};
    float t = std::min(1.0f,
                       std::pow(static_cast<float>(prob / max_prob), 0.45f));
    if (t < 0.5f) {
        auto v = static_cast<unsigned char>(t * 2.0f * 255);
        return {0, v, v, 255};
    } else {
        auto v = static_cast<unsigned char>((t - 0.5f) * 2.0f * 255);
        return {v, 255, 255, 255};
    }
}

// Utility

/// Linear interpolation between two Colors.  t in [0,1].
inline Color lerp_color(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return {
        static_cast<unsigned char>(a.r + t * (b.r - a.r)),
        static_cast<unsigned char>(a.g + t * (b.g - a.g)),
        static_cast<unsigned char>(a.b + t * (b.b - a.b)),
        static_cast<unsigned char>(a.a + t * (b.a - a.a)),
    };
}
