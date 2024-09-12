#include "nigiri/common/parse_time.h"

#include <sstream>

namespace nigiri {

unixtime_t parse_time(std::string const& str) {
  unixtime_t parsed;
  auto ss = std::stringstream{str};
  ss.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

}  // namespace nigiri