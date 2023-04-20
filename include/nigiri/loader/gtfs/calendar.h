#pragma once

#include <bitset>
#include <string_view>

#include "date/date.h"

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct calendar {
  std::bitset<7> week_days_;
  interval<date::sys_days> interval_;
};

hash_map<std::string, calendar> read_calendar(std::string_view file_content);

}  // namespace nigiri::loader::gtfs
