#pragma once

#include <bitset>
#include <string_view>

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct calendar {
  std::bitset<7> week_days_;
  date::sys_days first_day_, last_day_;
};

hash_map<std::string, calendar> read_calendar(std::string_view e);

}  // namespace nigiri::loader::gtfs
