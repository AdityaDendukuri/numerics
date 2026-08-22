#include "linalg/solvers/resolvent.hpp"

namespace num {
namespace {

[[nodiscard]] std::vector<cplx> complex_copy(const Vector &source) {
    std::vector<cplx> result(source.size());
    for (idx index = 0; index < source.size(); ++index) {
        result[index] = source[index];
    }
    return result;
}

[[nodiscard]] std::vector<std::vector<cplx>> complex_copy(const std::vector<Vector> &sources) {
    std::vector<std::vector<cplx>> result;
    result.reserve(sources.size());
    for (const Vector &source : sources) {
        result.push_back(complex_copy(source));
    }
    return result;
}

} // namespace

ResolventFactor::ResolventFactor(cplx shift, const Matrix &matrix) : solver_(matrix) {
    solver_.factorize(shift);
}

std::vector<cplx> ResolventFactor::solve(const std::vector<cplx> &rhs) const {
    return solver_.solve(rhs);
}

std::vector<std::vector<cplx>>
ResolventFactor::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    return solver_.solve(right_hand_sides);
}

std::vector<cplx> resolvent_solve(cplx shift, const Matrix &matrix, const Vector &right_hand_side) {
    ResolventFactor factor(shift, matrix);
    return factor.solve(complex_copy(right_hand_side));
}

std::vector<std::vector<cplx>>
resolvent_solve_rhs_batch(cplx shift, const Matrix &matrix,
                          const std::vector<Vector> &right_hand_sides) {
    ResolventFactor factor(shift, matrix);
    return factor.solve(complex_copy(right_hand_sides));
}

std::vector<std::vector<cplx>> resolvent_solve_batch(const std::vector<cplx> &shifts,
                                                     const Matrix &matrix,
                                                     const Vector &right_hand_side) {
    std::vector<std::vector<cplx>> result(shifts.size());
#if defined(_OPENMP)
#pragma omp parallel for if (shifts.size() > 4)
#endif
    for (std::size_t index = 0; index < shifts.size(); ++index) {
        result[index] = resolvent_solve(shifts[index], matrix, right_hand_side);
    }
    return result;
}

std::vector<std::vector<std::vector<cplx>>>
resolvent_solve_batch(const std::vector<cplx> &shifts, const Matrix &matrix,
                      const std::vector<Vector> &right_hand_sides) {
    std::vector<std::vector<std::vector<cplx>>> result(shifts.size());
#if defined(_OPENMP)
#pragma omp parallel for if (shifts.size() > 4)
#endif
    for (std::size_t index = 0; index < shifts.size(); ++index) {
        result[index] = resolvent_solve_rhs_batch(shifts[index], matrix, right_hand_sides);
    }
    return result;
}

} // namespace num
