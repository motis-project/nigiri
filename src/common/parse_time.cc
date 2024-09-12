#include "nigiri/common/parse_time.h"

#include <ios>
#include <strstream>

namespace nigiri {

unixtime_t parse_time(std::string const& sv) {
  unixtime_t parsed;
  auto ss = std::stringstream{sv};
  ss.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

}  // namespace nigiri