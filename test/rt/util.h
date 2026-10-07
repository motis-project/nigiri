#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "fmt/core.h"

#include "date/date.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::test {

struct trip {
  struct delay {
    std::optional<std::string> stop_id_{};
    std::optional<unsigned> seq_{};
    event_type ev_type_;
    int delay_minutes_{0};
  };
  std::string trip_id_;
  std::vector<delay> delays_;
};

template <typename T>
std::uint64_t to_unix(T&& x) {
  return static_cast<std::uint64_t>(
      std::chrono::time_point_cast<std::chrono::seconds>(x)
          .time_since_epoch()
          .count());
};

// One stop time update, stated the way real feeds do it.
struct stu {
  unsigned seq_;
  std::string stop_id_;  // the scheduled stop id
  std::optional<std::string> assigned_{};  // track change
  std::optional<int> arr_delay_{};  // minutes
  std::optional<int> dep_delay_{};  // minutes
  bool has_seq_{true};
  bool has_stop_id_{true};
  // Track change stated the old way: stop_id differs from the schedule and
  // there is no stop_time_properties.
  bool is_assigned_as_stop_id_{false};
  bool is_skipped_{false};
};

struct trip_update {
  std::string trip_id_;
  std::vector<stu> stus_;
};

// One trip update per entry; start_date goes into every trip descriptor.
transit_realtime::FeedMessage to_msg(
    std::vector<trip_update> const&,
    date::sys_seconds msg_time,
    std::optional<std::string> const& start_date = std::nullopt);

transit_realtime::FeedMessage to_feed_msg(std::vector<trip> const& trip_delays,
                                          date::sys_seconds const msg_time);

void with_rt_trips(
    timetable const& tt,
    date::sys_days base_day,
    std::vector<std::string> const& trip_ids,
    std::function<void(rt_timetable*, std::string_view)> const& fn);

}  // namespace nigiri::test
