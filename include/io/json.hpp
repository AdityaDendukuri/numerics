/// @file json.hpp
/// @brief Small JSON file helpers used by optional data-I/O modules.
#pragma once

#include <boost/json/value.hpp>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num::io {

/// Parse a complete JSON document from disk.
[[nodiscard]] boost::json::value read_json(const std::filesystem::path& path);
/// Serialize a JSON value to disk.
void write_json(const boost::json::value& value, const std::filesystem::path& path);

template<typename T>
/// Convert a JSON numeric array to an arithmetic vector.
[[nodiscard]] std::vector<T> json_vector(const boost::json::value& value) {
  static_assert(std::is_arithmetic_v<T>);
  if (!value.is_array()) {
    throw std::invalid_argument("JSON value must be an array");
  }
  std::vector<T> result;
  result.reserve(value.as_array().size());
  for (const auto& item : value.as_array()) {
    const bool valid = [](const boost::json::value& number) {
      if constexpr (std::is_integral_v<T>) {
        return number.is_int64() || number.is_uint64();
      }
      return number.is_double() || number.is_int64() || number.is_uint64();
    }(item);
    if (!valid) {
      throw std::invalid_argument("JSON array entries must be numeric");
    }
    result.push_back(item.to_number<T>());
  }
  return result;
}

template<typename T>
/// Convert a JSON array of numeric arrays to a nested vector.
[[nodiscard]] std::vector<std::vector<T>> json_matrix(const boost::json::value& value) {
  if (!value.is_array()) {
    throw std::invalid_argument("JSON matrix must be an array");
  }
  std::vector<std::vector<T>> result;
  result.reserve(value.as_array().size());
  for (const auto& row : value.as_array()) {
    result.push_back(json_vector<T>(row));
  }
  return result;
}

} // namespace num::io
