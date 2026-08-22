#include "io/json.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace num::io {

json read_json(const std::filesystem::path &path) {
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

void write_json(const json &value, const std::filesystem::path &path) {
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
