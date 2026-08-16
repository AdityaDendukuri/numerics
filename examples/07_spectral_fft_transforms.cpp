/// @file 07_spectral_fft_transforms.cpp
/// @brief 1D/2D FFT, iFFT, Real rFFT/irfft, and Precomputed FFTPlan.
#include <numerics.hpp>
#include <iostream>

int main() {
    using namespace num;
    using namespace num::spectral;

    idx n = 8;
    CVector in(n, cplx{0, 0});
    for (idx i = 0; i < n; ++i)
        in[i] = {static_cast<double>(i + 1), 0.0};

    // 1. Complex-to-Complex Forward & Inverse FFT
    CVector out(n, cplx{0, 0});
    fft(in, out, FFTBackend::seq);

    CVector recovered(n, cplx{0, 0});
    ifft(out, recovered, FFTBackend::seq);
    for (idx i = 0; i < n; ++i)
        recovered[i] /= static_cast<double>(n);

    std::cout << "FFT Round-Trip Error = " << std::abs(recovered[0] - in[0]) << "\n";

    // 2. Real-to-Complex rFFT and Inverse irfft
    Vector r_in{1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0};
    CVector r_out(n / 2 + 1, cplx{0, 0});
    rfft(r_in, r_out, FFTBackend::seq);

    Vector r_rec(n, 0.0);
    irfft(r_out, static_cast<int>(n), r_rec, FFTBackend::seq);
    for (idx i = 0; i < n; ++i)
        r_rec[i] /= static_cast<double>(n);

    std::cout << "rFFT Real DC Component = " << r_out[0].real() << "\n";

    // 3. Precomputed Amortized FFTPlan
    FFTPlan plan(static_cast<int>(n), false, FFTBackend::seq);
    CVector plan_out(n, cplx{0, 0});
    plan.execute(in, plan_out);
    std::cout << "FFTPlan Executed successfully.\n";

    return 0;
}
