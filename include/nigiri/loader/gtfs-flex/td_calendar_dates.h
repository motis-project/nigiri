#pragma once

#include "date/date.h"

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

struct calendar_date {
  enum { kAdd, kRemove } type_{kAdd};
  date::sys_days day_;
};

using td_calendar_date_map_t = hash_map<std::string, std::vector<calendar_date>>;

td_calendar_date_map_t read_td_calendar_date(
    std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex