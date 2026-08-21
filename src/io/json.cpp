#include "io/json.hpp"

#include <boost/json/parse.hpp>
#include <boost/json/serialize.hpp>
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace num::io {

boost::json::value read_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open JSON file: " + path.string());
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  try {
    return boost::json::parse(buffer.str());
  } catch (const boost::system::system_error& error) {
    throw std::runtime_error("invalid JSON in " + path.string() + ": " + error.what());
  }
}

void write_json(const boost::json::value& value, const std::filesystem::path& path) {
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot open JSON file for writing: " + path.string());
  output << boost::json::serialize(value) << '\n';
  if (!output)
    throw std::runtime_error("failed writing JSON file: " + path.string());
}

} // namespace num::io
