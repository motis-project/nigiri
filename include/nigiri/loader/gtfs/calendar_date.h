#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct calendar_date {
  enum { kAdd, kRemove } type_{kAdd};
  date::sys_days day_;
};

hash_map<std::string, std::vector<calendar_date>> read_calendar_date(
    std::string_view file_content);

}  // namespace nigiri::loader::gtfs
