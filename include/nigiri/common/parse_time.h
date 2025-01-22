#pragma once

#include "nigiri/types.h"

namespace nigiri {

unixtime_t parse_time_tz(std::string_view s, char const* format);

unixtime_t parse_time(std::string_view s, char const* format);

unixtime_t parse_time_no_tz(std::string_view);

}  // namespace nigiri