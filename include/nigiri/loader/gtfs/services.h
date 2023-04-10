#pragma once

#include "date/date.h"

#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct traffic_days {
  interval<date::sys_days> interval_;
  hash_map<std::string, std::unique_ptr<bitfield>> traffic_days_;
};

traffic_days merge_traffic_days(
    hash_map<std::string, calendar> const&,
    hash_map<std::string, std::vector<calendar_date>> const&);

}  // namespace nigiri::loader::gtfs
