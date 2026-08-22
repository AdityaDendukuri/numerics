/// @file json.hpp
/// @brief Small JSON file helpers using nlohmann::json.
#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num::io {

using json = nlohmann::json;

/// Parse a complete JSON document from disk.
[[nodiscard]] json read_json(const std::filesystem::path &path);
/// Serialize a JSON value to disk.
void write_json(const json &value, const std::filesystem::path &path);

template <typename T>
/// Convert a JSON numeric array to an arithmetic vector.
[[nodiscard]] inline std::vector<T> json_vector(const json &value) {
    static_assert(std::is_arithmetic_v<T>);
    if (!value.is_array()) {
        throw std::invalid_argument("JSON value must be an array");
    }
    try {
        return value.get<std::vector<T>>();
    } catch (const json::exception &) {
        throw std::invalid_argument("JSON array entries must be numeric");
    }
}

template <typename T>
/// Convert a JSON array of numeric arrays to a nested vector.
[[nodiscard]] inline std::vector<std::vector<T>> json_matrix(const json &value) {
    if (!value.is_array()) {
        throw std::invalid_argument("JSON matrix must be an array");
    }
    try {
        return value.get<std::vector<std::vector<T>>>();
    } catch (const json::exception &) {
        throw std::invalid_argument("JSON matrix entries must be numeric arrays");
    }
}

} // namespace num::io
