#pragma once

#include "utl/verify.h"

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri {

unixtime_t parse_time_tz(std::string_view s, char const* format);

unixtime_t parse_time_no_tz(std::string_view);

template <typename FirstFormat, typename... RestFormat>
unixtime_t parse_time(std::string_view s,
                      FirstFormat format,
                      RestFormat... rest) {
  auto in = std::istringstream{std::string{s}};

  auto t = date::sys_time<std::chrono::milliseconds>{};
  auto offset = std::chrono::minutes{};
  in.clear();
  in.str(std::string{s});
  in >> date::parse(format, t, offset);

  if (!in.fail()) {
    return std::chrono::time_point_cast<unixtime_t::duration>(t);
  }

  if constexpr (sizeof...(RestFormat) == 0U) {
    throw utl::fail("unable to parse time {:?}", s);
  } else {
    return parse_time(s, rest...);
  }
}

}  // namespace nigiri