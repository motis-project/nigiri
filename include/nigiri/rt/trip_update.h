#pragma once

#include <string>

#include "nigiri/common/delta_t.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;
struct rt_timetable;

struct trip_info {
  std::basic_string<stop::value_type> stop_seq_;
  std::vector<unixtime_t> event_times_;
  std::basic_string<clasz> section_clasz_;
};

struct trip_update {
  bool is_cancel() const { return info_.stop_seq_.empty(); }

  trip_id id_;
  date::year_month_day day_;
  std::optional<unixtime_t> start_time_;
  bool is_rerouting_{false};
  bool is_additional_{false};
  trip_info info_;
};

void update(timetable const&, rt_timetable&, trip_update const&);

}  // namespace nigiri