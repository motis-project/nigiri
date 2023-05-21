#pragma once

#include <string>

#include "nigiri/common/delta_t.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;
struct rt_timetable;

struct trip_update {
  bool is_cancel() const { return stop_seq_.empty(); }

  trip_id id_;
  day_idx_t day_;
  bool is_rerouting_{false};
  std::basic_string<stop::value_type> stop_seq_;
  std::basic_string<delta_t> event_times_;
  std::basic_string<clasz> section_clasz_;
};

void update(timetable const&, rt_timetable&, trip_update const&);

}  // namespace nigiri