/// @file spectral/backends/fftw/impl.hpp
/// @brief FFTW3 backend wrappers.
/// Only included by src/spectral/fft.cpp.
#pragma once

#ifdef NUMERICS_HAS_FFTW
#include "spectral/fft.hpp"
#include <fftw3.h>
#include <stdexcept>

namespace backends {
namespace fftw {

// std::complex<double> is layout-compatible with fftw_complex (double[2])
inline fftw_complex* fc(num::CVector& v) {
    return reinterpret_cast<fftw_complex*>(v.data());
}
inline const fftw_complex* fc(const num::CVector& v) {
    return reinterpret_cast<const fftw_complex*>(v.data());
}

inline void fft(const num::CVector& in, num::CVector& out) {
    fftw_plan p = fftw_plan_dft_1d(static_cast<int>(in.size()),
                                    const_cast<fftw_complex*>(fc(in)), fc(out),
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

inline void ifft(const num::CVector& in, num::CVector& out) {
    fftw_plan p = fftw_plan_dft_1d(static_cast<int>(in.size()),
                                    const_cast<fftw_complex*>(fc(in)), fc(out),
                                    FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

inline void rfft(const num::Vector& in, num::CVector& out) {
    fftw_plan p = fftw_plan_dft_r2c_1d(static_cast<int>(in.size()),
                                        const_cast<double*>(in.data()), fc(out),
                                        FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

inline void irfft(const num::CVector& in, int n, num::Vector& out) {
    fftw_plan p = fftw_plan_dft_c2r_1d(n,
                                        const_cast<fftw_complex*>(fc(in)), out.data(),
                                        FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

struct FFTPlanImpl {
    fftw_plan plan;
    int n;

    FFTPlanImpl(int n_, bool forward) : n(n_) {
        // Allocate dummy arrays -- FFTW_MEASURE overwrites them during planning
        num::CVector tmp_in(n_, num::cplx{0, 0}), tmp_out(n_, num::cplx{0, 0});
        plan = fftw_plan_dft_1d(n_, fc(tmp_in), fc(tmp_out),
                                 forward ? FFTW_FORWARD : FFTW_BACKWARD,
                                 FFTW_MEASURE);
    }

    ~FFTPlanImpl() { fftw_destroy_plan(plan); }

    void execute(const num::CVector& in, num::CVector& out) const {
        fftw_execute_dft(plan,
                         const_cast<fftw_complex*>(fc(in)),
                         const_cast<fftw_complex*>(fc(out)));
    }
};

} // namespace fftw
} // namespace backends

#endif // NUMERICS_HAS_FFTW
