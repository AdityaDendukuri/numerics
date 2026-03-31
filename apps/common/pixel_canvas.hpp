/// @file apps/common/pixel_canvas.hpp
/// @brief RAII wrapper around a raylib Texture2D + CPU pixel buffer.
///
/// Replaces the repeated boilerplate in 2D-field apps:
///   std::vector<Color> pixels(N*N);
///   Image img = {pixels.data(), N, N, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
///   Texture2D tex = LoadTextureFromImage(img);
///   // ... per frame:
///   for (...) pixels[...] = some_color(...);
///   UpdateTexture(tex, pixels.data());
///   DrawTexturePro(tex, src, dst, {0,0}, 0.0f, WHITE);
///   UnloadTexture(tex);
///
/// Usage:
/// @code
///   PixelCanvas canvas(N, N, {0, 0, (float)WIN, (float)WIN});
///
///   // Physical convention: col=x, row=y with row 0 at the bottom.
///   canvas.fill([&](int col, int row) -> Color {
///       return heat_color(field[row * N + col]);
///   });
///   canvas.draw();
/// @endcode
///
/// @note Must be constructed after InitWindow() (Texture2D requires an active
///       OpenGL context).  Not copyable -- owns GPU texture memory.
#pragma once

#include <raylib.h>
#include <vector>

// PixelCanvas

struct PixelCanvas {
    int cols; ///< Grid width  (number of columns = x cells)
    int rows; ///< Grid height (number of rows    = y cells)

    /// @param cols_   Grid width in cells
    /// @param rows_   Grid height in cells
    /// @param dst     On-screen destination rectangle (pixels)
    PixelCanvas(int cols_, int rows_, Rectangle dst)
        : cols(cols_)
        , rows(rows_)
        , pixels_(cols_ * rows_, {0, 0, 0, 255})
        , dst_(dst) {
        Image img = {pixels_.data(),
                     cols_,
                     rows_,
                     1,
                     PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        tex_      = LoadTextureFromImage(img);
    }

    ~PixelCanvas() {
        UnloadTexture(tex_);
    }

    PixelCanvas(const PixelCanvas&)            = delete;
    PixelCanvas& operator=(const PixelCanvas&) = delete;

    /// Fill using fn(col, row) -> Color.
    /// Physical convention: row 0 = bottom of screen (y-axis points up).
    /// The canvas flips the y-axis internally so the texture renders correctly.
    template<typename Fn>
    void fill(Fn fn) {
        for (int row = 0; row < rows; ++row) {
            int tex_row = rows - 1
                          - row; // flip: row 0 (bottom) -> last texture row
            for (int col = 0; col < cols; ++col)
                pixels_[tex_row * cols + col] = fn(col, row);
        }
        UpdateTexture(tex_, pixels_.data());
    }

    /// Fill using fn(col, row) -> Color, row 0 = top of screen (no y-flip).
    /// Use this when your data is already in top-down order.
    template<typename Fn>
    void fill_topdown(Fn fn) {
        for (int row = 0; row < rows; ++row)
            for (int col = 0; col < cols; ++col)
                pixels_[row * cols + col] = fn(col, row);
        UpdateTexture(tex_, pixels_.data());
    }

    /// Upload pixels to GPU and draw to the destination rectangle.
    void draw(Color tint = WHITE) {
        Rectangle src = {0,
                         0,
                         static_cast<float>(cols),
                         static_cast<float>(rows)};
        DrawTexturePro(tex_, src, dst_, {0, 0}, 0.0f, tint);
    }

  private:
    std::vector<Color> pixels_;
    Texture2D          tex_;
    Rectangle          dst_;
};
