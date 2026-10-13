#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/raptor/pong.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

// One trip per day, S0 -> S1, 08:00 -> 09:00 local time (= 07:00 -> 08:00 UTC,
// Europe/London is UTC+1 in June). Runs for the whole of June 2026, so pong
// could crawl through the entire month to satisfy `min_connection_count_` if
// it were not capped by `kMaxLookAhead`.
constexpr auto kTimetable = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DAILY,1,1,1,1,1,1,1,20260601,20260630

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,DAILY,R0_DAILY,R0_DAILY,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_DAILY,08:00:00,08:00:00,S0,0,0,0
R0_DAILY,09:00:00,09:00:00,S1,1,0,0
)";

timetable get_tt() {
  static auto const files = mem_dir::read(kTimetable);
  timetable tt;
  tt.date_range_ = {sys_days{2026_y / June / 01}, sys_days{2026_y / July / 01}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, files, tt);
  finalize(tt);
  return tt;
}

// Enough connections requested that the search would keep crawling through the
// timetable day by day if `kMaxLookAhead` did not stop it.
constexpr auto kMinConnectionCount = 10U;

routing::query make_query(timetable const& tt,
                          interval<unixtime_t> const start_time,
                          bool const fwd) {
  auto const S0 = tt.find(location_id{"S0", source_idx_t{0}}).value();
  auto const S1 = tt.find(location_id{"S1", source_idx_t{0}}).value();
  return routing::query{
      .start_time_ = start_time,
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = false,
      .start_ = {{fwd ? S0 : S1, 0min, 0U}},
      .destination_ = {{fwd ? S1 : S0, 0min, 0U}},
      .min_connection_count_ = kMinConnectionCount,
      .extend_interval_earlier_ = !fwd,
      .extend_interval_later_ = fwd};
}

std::vector<unixtime_t> departures(routing::routing_result const& r) {
  auto times = std::vector<unixtime_t>{};
  for (auto const& j : *r.journeys_) {
    times.emplace_back(j.departure_time());
  }
  return times;
}

std::vector<unixtime_t> arrivals(routing::routing_result const& r) {
  auto times = std::vector<unixtime_t>{};
  for (auto const& j : *r.journeys_) {
    times.emplace_back(j.arrival_time());
  }
  return times;
}

}  // namespace

// Query on Monday 2026-06-01: the six days of lookahead reach up to (but not
// including) Sunday 2026-06-07 07:00 - in particular no journey of the
// following Monday is reported.
TEST(routing, pong_max_lookahead_forward) {
  auto const tt = get_tt();
  auto s_state = routing::search_state{};
  auto r_state = routing::raptor_state{};

  auto const query_start = unixtime_t{sys_days{2026_y / June / 01}};
  auto const result = routing::pong_search(
      tt, nullptr, s_state, r_state,
      make_query(tt, {query_start, query_start + 1h}, true),
      direction::kForward);

  EXPECT_EQ(
      (std::vector<unixtime_t>{unixtime_t{sys_days{2026_y / June / 01}} + 7h,
                               unixtime_t{sys_days{2026_y / June / 02}} + 7h,
                               unixtime_t{sys_days{2026_y / June / 03}} + 7h,
                               unixtime_t{sys_days{2026_y / June / 04}} + 7h,
                               unixtime_t{sys_days{2026_y / June / 05}} + 7h,
                               unixtime_t{sys_days{2026_y / June / 06}} + 7h}),
      departures(result));

  for (auto const t : departures(result)) {
    EXPECT_LT(t, query_start + 6_days);
  }
  EXPECT_LE(result.interval_.to_, query_start + 6_days);
}

// Same in the other direction: arriving on Monday 2026-06-15, no journey of the
// previous Monday is reported.
TEST(routing, pong_max_lookahead_backward) {
  auto const tt = get_tt();
  auto s_state = routing::search_state{};
  auto r_state = routing::raptor_state{};

  auto const query_end = unixtime_t{sys_days{2026_y / June / 16}};
  auto const result = routing::pong_search(
      tt, nullptr, s_state, r_state,
      make_query(tt, {query_end - 1h, query_end}, false), direction::kBackward);

  EXPECT_EQ(
      (std::vector<unixtime_t>{unixtime_t{sys_days{2026_y / June / 10}} + 8h,
                               unixtime_t{sys_days{2026_y / June / 11}} + 8h,
                               unixtime_t{sys_days{2026_y / June / 12}} + 8h,
                               unixtime_t{sys_days{2026_y / June / 13}} + 8h,
                               unixtime_t{sys_days{2026_y / June / 14}} + 8h,
                               unixtime_t{sys_days{2026_y / June / 15}} + 8h}),
      arrivals(result));

  for (auto const t : arrivals(result)) {
    EXPECT_GT(t, query_end - 6_days);
  }
  EXPECT_GE(result.interval_.from_, query_end - 6_days);
}
