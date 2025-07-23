#pragma once

#include "date/date.h"

#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

using traffic_days_t = hash_map<std::string, std::unique_ptr<bitfield>>;

traffic_days_t merge_traffic_days(
    interval<date::sys_days> const& tt_interval,
    hash_map<std::string, calendar> const&,
    hash_map<std::string, std::vector<calendar_date>> const&,
    std::optional<date::sys_days> const& feed_end_date = std::nullopt);

interval<date::sys_days> max_service_range(
    hash_map<std::string, calendar> const&,
    hash_map<std::string, std::vector<calendar_date>> const&,
    std::optional<date::sys_days> const& feed_end_date);

}  // namespace nigiri::loader::gtfs
