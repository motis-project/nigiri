#include "nigiri/common/parse_time.h"

#include <strstream>

namespace nigiri {

unixtime_t parse_time(std::string const& sv) {
  unixtime_t parsed;
  auto ss = std::stringstream{sv};
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

}  // namespace nigiri