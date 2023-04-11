#pragma once

#include "date/date.h"

namespace nigiri::loader::gtfs {

inline date::sys_days parse_date(unsigned const yyyymmdd) {
  return date::year_month_day{date::year{static_cast<int>(yyyymmdd) / 10000},
                              date::month{(yyyymmdd % 10000) / 100},
                              date::day{yyyymmdd % 100}};
}

}  // namespace nigiri::loader::gtfs
