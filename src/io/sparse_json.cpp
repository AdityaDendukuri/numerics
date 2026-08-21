#include "io/sparse_json.hpp"

#include <boost/json/object.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace num::io {
namespace {

const boost::json::value& field(const boost::json::object& object, const char* name) {
  const auto* value = object.if_contains(name);
  if (value == nullptr) {
    throw std::invalid_argument(std::string("sparse matrix JSON is missing '") + name + "'");
  }
  return *value;
}

template <class T>
std::vector<T> array(const boost::json::value& value, const char* name) {
  if (!value.is_array()) {
    throw std::invalid_argument(std::string("sparse matrix field '") + name + "' must be an array");
  }
  std::vector<T> result;
  result.reserve(value.as_array().size());
  for (const auto& item : value.as_array()) {
    if constexpr (std::is_same_v<T, real>) {
      if (!item.is_double() && !item.is_int64() && !item.is_uint64()) {
        throw std::invalid_argument("sparse matrix values must be numeric");
      }
      result.push_back(item.to_number<real>());
    } else {
      if (!item.is_int64() && !item.is_uint64()) {
        throw std::invalid_argument("sparse matrix indices must be integers");
      }
      result.push_back(item.to_number<T>());
    }
  }
  return result;
}

} // namespace

SparseMatrix sparse_matrix(const boost::json::value& value) {
  if (!value.is_object()) {
    throw std::invalid_argument("sparse matrix JSON must be an object");
  }
  const auto& object = value.as_object();
  const auto shape = array<idx>(field(object, "shape"), "shape");
  if (shape.size() != 2) {
    throw std::invalid_argument("sparse matrix shape must have two entries");
  }
  const auto values = array<real>(field(object, "values"), "values");
  const auto* format = object.if_contains("format");
  const std::string storage = format != nullptr && format->is_string()
                                   ? std::string(format->as_string())
                                   : "csc";
  if (storage == "csr") {
    const auto columns = array<idx>(field(object, "col_indices"), "col_indices");
    const auto pointers = array<idx>(field(object, "row_ptrs"), "row_ptrs");
    if (pointers.size() != shape[0] + 1 || pointers.back() != values.size() ||
        columns.size() != values.size()) {
      throw std::invalid_argument("invalid sparse matrix CSR arrays");
    }
    return SparseMatrix(shape[0], shape[1], values, columns, pointers);
  }
  if (storage != "csc")
    throw std::invalid_argument("sparse matrix format must be 'csr' or 'csc'");
  const auto rows = array<idx>(field(object, "row_indices"), "row_indices");
  const auto pointers = array<idx>(field(object, "col_ptrs"), "col_ptrs");
  if (pointers.empty() || pointers.back() > values.size() || pointers.back() > rows.size()) {
    throw std::invalid_argument("invalid sparse matrix CSC pointers");
  }
  return SparseMatrix::from_csc(shape[0], shape[1], values, rows, pointers);
}

boost::json::value sparse_matrix_json(const SparseMatrix& matrix) {
  // Generic output is emitted as CSR, matching Numerics' native storage.
  boost::json::array values;
  boost::json::array columns;
  boost::json::array pointers;
  pointers.emplace_back(0);
  for (idx row = 0; row < matrix.n_rows(); ++row) {
    for (idx k = matrix.row_ptr()[row]; k < matrix.row_ptr()[row + 1]; ++k) {
      values.emplace_back(matrix.values()[k]);
      columns.emplace_back(matrix.col_idx()[k]);
    }
    pointers.emplace_back(values.size());
  }
  boost::json::object result;
  result["shape"] = boost::json::array{matrix.n_rows(), matrix.n_cols()};
  result["values"] = std::move(values);
  result["col_indices"] = std::move(columns);
  result["row_ptrs"] = std::move(pointers);
  result["format"] = "csr";
  return result;
}

} // namespace num::io
