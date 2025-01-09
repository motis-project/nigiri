#include "nigiri/common/parse_time.h"

#include <sstream>

namespace nigiri {

unixtime_t parse_time_tz(std::string_view s, char const* format) {
  std::stringstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in << s;

  date::local_seconds ls;
  std::string tz;
  in >> date::parse(format, ls, tz);

  return std::chrono::time_point_cast<unixtime_t::duration>(
      date::make_zoned(tz, ls).get_sys_time());
}

unixtime_t parse_time(std::string_view s, char const* format) {
  std::stringstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in << s;

  unixtime_t u;
  in >> date::parse(format, u);

  return u;
}

unixtime_t parse_time_no_tz(std::string_view s) {
  return parse_time(s, "%FT%T");
}

}  // namespace nigiri