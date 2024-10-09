#pragma once
#include <nigiri/types.h>
#include <bitset>

namespace nigiri::loader::gtfs_flex {
  struct td_calendar {
    std::bitset<7> week_days_;
    interval<date::sys_days> interval_;
  };

  using td_calendar_map_t = hash_map<std::string, std::unique_ptr<td_calendar>>;

  td_calendar_map_t read_td_calendar(std::string_view file_content);
}