/// @file json.hpp
/// @brief Small JSON file helpers used by optional data-I/O modules.
#pragma once

#include <boost/json/value.hpp>
#include <filesystem>

namespace num::io {

[[nodiscard]] boost::json::value read_json(const std::filesystem::path& path);
void write_json(const boost::json::value& value, const std::filesystem::path& path);

} // namespace num::io
