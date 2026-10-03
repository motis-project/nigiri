#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../transfer_rules_util.h"
#include "./util.h"

// GTFS-RT messages for the transfers.txt rule tests: trip updates of kDay,
// source 0.

namespace nigiri::test {

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

inline transit_realtime::FeedMessage to_msg(
    std::vector<trip_update> const& updates) {
  auto msg = transit_realtime::FeedMessage{};
  auto* const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(kDay + std::chrono::hours{8}));

  auto id = 0U;
  for (auto const& u : updates) {
    auto* const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));
    auto* const tu = e->mutable_trip_update();
    tu->mutable_trip()->set_trip_id(u.trip_id_);
    tu->mutable_trip()->set_start_date("20190501");
    for (auto const& s : u.stus_) {
      auto* const x = tu->add_stop_time_update();
      if (s.has_seq_) {
        x->set_stop_sequence(s.seq_);
      }
      if (s.is_assigned_as_stop_id_) {
        x->set_stop_id(*s.assigned_);
      } else {
        if (s.has_stop_id_) {
          x->set_stop_id(s.stop_id_);
        }
        if (s.assigned_.has_value()) {
          x->mutable_stop_time_properties()->set_assigned_stop_id(*s.assigned_);
        }
      }
      if (s.arr_delay_.has_value()) {
        x->mutable_arrival()->set_delay(*s.arr_delay_ * 60);
      }
      if (s.dep_delay_.has_value()) {
        x->mutable_departure()->set_delay(*s.dep_delay_ * 60);
      }
      if (s.is_skipped_) {
        x->set_schedule_relationship(
            transit_realtime::
                TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
      }
    }
  }
  return msg;
}

// Applies the updates; the lower bounds follow, as motis does after every
// update.
inline void update(timetable const& tt,
                   rt_timetable& rtt,
                   std::vector<trip_update> const& updates) {
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", to_msg(updates));
  rtt.update_lbs(tt);
}

}  // namespace nigiri::test
