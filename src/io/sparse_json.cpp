#include "io/sparse_json.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace num::io {

SparseMatrix sparse_matrix(const json &value) {
    if (!value.is_object()) {
        throw std::invalid_argument("sparse matrix JSON must be an object");
    }
    if (!value.contains("shape") || !value.contains("values")) {
        throw std::invalid_argument("sparse matrix JSON missing required fields");
    }
    const auto shape = value["shape"].get<std::vector<idx>>();
    if (shape.size() != 2) {
        throw std::invalid_argument("sparse matrix shape must have two entries");
    }
    const auto values = value["values"].get<std::vector<real>>();
    const std::string storage = value.value("format", "csc");

    if (storage == "csr") {
        if (!value.contains("col_indices") || !value.contains("row_ptrs")) {
            throw std::invalid_argument("CSR sparse matrix JSON missing col_indices or row_ptrs");
        }
        const auto columns = value["col_indices"].get<std::vector<idx>>();
        const auto pointers = value["row_ptrs"].get<std::vector<idx>>();
        if (pointers.size() != shape[0] + 1 || pointers.back() != values.size() ||
            columns.size() != values.size()) {
            throw std::invalid_argument("invalid sparse matrix CSR arrays");
        }
        return SparseMatrix(shape[0], shape[1], values, columns, pointers);
    }
    if (storage != "csc") {
        throw std::invalid_argument("sparse matrix format must be 'csr' or 'csc'");
    }
    if (!value.contains("row_indices") || !value.contains("col_ptrs")) {
        throw std::invalid_argument("CSC sparse matrix JSON missing row_indices or col_ptrs");
    }
    const auto rows = value["row_indices"].get<std::vector<idx>>();
    const auto pointers = value["col_ptrs"].get<std::vector<idx>>();
    if (pointers.empty() || pointers.back() > values.size() || pointers.back() > rows.size()) {
        throw std::invalid_argument("invalid sparse matrix CSC pointers");
    }
    return SparseMatrix::from_csc(shape[0], shape[1], values, rows, pointers);
}

json sparse_matrix_json(const SparseMatrix &matrix) {
    json result;
    result["shape"] = {matrix.n_rows(), matrix.n_cols()};
    result["format"] = "csr";

    std::vector<real> values;
    std::vector<idx> columns;
    std::vector<idx> pointers;
    pointers.reserve(matrix.n_rows() + 1);
    pointers.push_back(0);

    for (idx row = 0; row < matrix.n_rows(); ++row) {
        for (idx k = matrix.row_ptr()[row]; k < matrix.row_ptr()[row + 1]; ++k) {
            values.push_back(matrix.values()[k]);
            columns.push_back(matrix.col_idx()[k]);
        }
        pointers.push_back(values.size());
    }

    result["values"] = std::move(values);
    result["col_indices"] = std::move(columns);
    result["row_ptrs"] = std::move(pointers);
    return result;
}

} // namespace num::io
