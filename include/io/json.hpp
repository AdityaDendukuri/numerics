/// @file io/json.hpp
/// @brief JSON file helpers.
#pragma once

#include "core/types.hpp"
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num::io {

/// @brief JSON value: `nlohmann::json`.
using json = nlohmann::json;

/// @brief Read a JSON file.
[[nodiscard]] json read_json(const std::filesystem::path &path);
/// @brief Write a JSON file.
void write_json(const json &value, const std::filesystem::path &path);

template <typename T>
/// @brief JSON numeric array to an array.
[[nodiscard]] inline array<T> json_vector(const json &value) {
    static_assert(std::is_arithmetic_v<T>);
    if (!value.is_array()) {
        throw std::invalid_argument("JSON value must be an array");
    }
    try {
        return value.get<array<T>>();
    } catch (const json::exception &) {
        throw std::invalid_argument("JSON array entries must be numeric");
    }
}

template <typename T>
/// @brief JSON array of numeric arrays to a nested array.
[[nodiscard]] inline array<array<T>> json_matrix(const json &value) {
    if (!value.is_array()) {
        throw std::invalid_argument("JSON matrix must be an array");
    }
    try {
        return value.get<array<array<T>>>();
    } catch (const json::exception &) {
        throw std::invalid_argument("JSON matrix entries must be numeric arrays");
    }
}



inline json read_json(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open JSON file: " + path.string());
    }
    json j;
    try {
        input >> j;
    } catch (const json::exception &error) {
        throw std::runtime_error("invalid JSON in " + path.string() + ": " + error.what());
    }
    return j;
}

inline void write_json(const json &value, const std::filesystem::path &path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open JSON file for writing: " + path.string());
    }
    output << value.dump(2) << '\n';
    if (!output) {
        throw std::runtime_error("failed writing JSON file: " + path.string());
    }
}

} // namespace num::io
