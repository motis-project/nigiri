#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/rt/create_rt_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;
using namespace std::chrono_literals;

namespace {

// Europe/Berlin on 2019-05-01 => UTC+2
//   T1: A 06:00 -> B 07:00 (= 04:00 -> 05:00 UTC), daily 05-01 .. 05-05
//
// The timetable's date range is much wider than the service so that the
// driver level tests can place a query months away from the real-time data
// while still inside the timetable.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S1,1,1,1,1,1,1,1,20190501,20190505

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,06:00:00,06:00:00,A,1,0,0
T1,07:00:00,07:00:00,B,2,0,0
)");
}

constexpr auto const kDay = 2019_y / May / 1;

// `x` relative to the base day midnight UTC.
unixtime_t t(auto&& x) { return unixtime_t{sys_days{kDay} + x}; }

// Start time on `d`, 08:00 UTC. 08:00 keeps `std::chrono::round<days>()` (used
// by the interval estimator to derive the maximum interval extension) away
// from the half-day tie.
unixtime_t at_8(year_month_day const d) { return unixtime_t{sys_days{d} + 8h}; }

timetable load_tt() {
  auto tt = timetable{};
  register_special_stations(tt);
  tt.date_range_ = {sys_days{2019_y / March / 25},
                    sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  return tt;
}

}  // namespace

// ===========================================================================
// Driver level: `raptor_search_interval()` / `pong_search_interval()`
// ===========================================================================

namespace {

query q_at(start_time_t const start,
           duration_t const max_travel_time = kMaxTravelTime) {
  auto q = query{};
  q.start_time_ = start;
  q.max_travel_time_ = max_travel_time;
  return q;
}

// The two driver-level decisions, spelled out: build the interval the search
// can reach, then ask the one predicate. Production does this inline at its
// single call site each (`raptor_search.cc` / `pong.cc`); these exist so the
// tests below read as one assertion per case.
template <direction SearchDir>
bool needs_rt(timetable const& tt, rt_timetable const& rtt, query const& q) {
  return rtt.affects(raptor_search_interval(SearchDir, tt, q), q.prf_idx_);
}

template <direction SearchDir>
bool pong_needs_rt(timetable const& tt,
                   rt_timetable const& rtt,
                   query const& q,
                   duration_t const min_look_ahead) {
  return rtt.affects(pong_search_interval(SearchDir, tt, q, min_look_ahead),
                     q.prf_idx_);
}

}  // namespace

TEST(routing, needs_rt_no_coverage) {
  auto const tt = load_tt();
  auto const rtt = rt::create_rt_timetable(tt, sys_days{kDay});

  ASSERT_TRUE(rtt.coverage_.empty());

  // Without any real-time data the query can always be answered statically -
  // regardless of where it is located in time.
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q_at(t(8h))));
  EXPECT_FALSE(needs_rt<direction::kBackward>(tt, rtt, q_at(t(8h))));
}

TEST(routing, needs_rt_overlap) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt.extend_coverage(interval{t(8h), t(9h + 10min)});

  // Query right in the covered window.
  EXPECT_TRUE(needs_rt<direction::kForward>(tt, rtt, q_at(t(8h))));
  EXPECT_TRUE(needs_rt<direction::kBackward>(tt, rtt, q_at(t(9h))));

  // Query interval instead of a single start time.
  EXPECT_TRUE(
      needs_rt<direction::kForward>(tt, rtt, q_at(interval{t(6h), t(12h)})));
}

TEST(routing, needs_rt_far_away) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt.extend_coverage(interval{t(8h), t(9h + 10min)});

  // A month later / earlier: even with the maximum interval extension (+-2
  // days) and the maximum travel time (5 days) the search cannot reach the
  // covered window.
  EXPECT_FALSE(
      needs_rt<direction::kForward>(tt, rtt, q_at(at_8(2019_y / June / 1))));
  EXPECT_FALSE(
      needs_rt<direction::kBackward>(tt, rtt, q_at(at_8(2019_y / June / 1))));
  EXPECT_FALSE(
      needs_rt<direction::kForward>(tt, rtt, q_at(at_8(2019_y / April / 1))));
  EXPECT_FALSE(
      needs_rt<direction::kBackward>(tt, rtt, q_at(at_8(2019_y / April / 1))));
}

// The relevant window is widened by the maximum travel time in search
// direction only - so the same query can need real-time data backwards but
// not forwards.
TEST(routing, needs_rt_direction) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt.extend_coverage(interval{t(8h), t(9h + 10min)});

  auto const start = at_8(2019_y / May / 5);

  // Forward the window is [05-05 08:00, 05-10 08:01): 05-01 is not in it.
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q_at(start)));

  // Backward it is [04-30 08:00, 05-05 08:01), which contains 05-01.
  EXPECT_TRUE(needs_rt<direction::kBackward>(tt, rtt, q_at(start)));

  // ... unless the journey may not be that long.
  EXPECT_FALSE(needs_rt<direction::kBackward>(tt, rtt, q_at(start, 60min)));
}

// The start interval is widened only in the directions the query allows, and
// only if it asks for a minimum number of connections. The common case: a
// query for the next day while the real-time data is for today.
TEST(routing, needs_rt_next_day) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt.extend_coverage(interval{t(8h), t(9h + 10min)});

  auto const next_day = at_8(2019_y / May / 2);

  // Forward: nothing before the start time is ever searched.
  auto q = q_at(next_day);
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q));

  // ... unless the interval may be extended backwards.
  q.min_connection_count_ = 3U;
  q.extend_interval_earlier_ = true;
  EXPECT_TRUE(needs_rt<direction::kForward>(tt, rtt, q));

  // The extension only happens for a minimum connection count.
  q.min_connection_count_ = 0U;
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q));

  // Backward (`arriveBy`): with the maximum travel time 05-01 is reachable
  // from the next day, with one hour it is not.
  EXPECT_TRUE(needs_rt<direction::kBackward>(tt, rtt, q_at(next_day)));
  EXPECT_FALSE(needs_rt<direction::kBackward>(tt, rtt, q_at(next_day, 60min)));
}

// Time dependent footpaths are not tracked by the coverage: a profile that has
// any keeps the real-time timetable, regardless of the query's position in
// time. A profile without is treated like the default profile.
TEST(routing, needs_rt_profile) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});

  ASSERT_TRUE(rtt.coverage_.empty());

  auto q = q_at(at_8(2019_y / June / 1));
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q));

  q.prf_idx_ = 1U;
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q));

  // The bitvecs are only sized for the timetable's profiles; the test
  // timetable has just the default one.
  rtt.has_td_footpaths_out_[1U].resize(tt.n_locations());
  rtt.has_td_footpaths_in_[1U].resize(tt.n_locations());
  rtt.has_td_footpaths_out_[1U].set(location_idx_t{0U}, true);
  EXPECT_TRUE(needs_rt<direction::kForward>(tt, rtt, q));
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q_at(q.start_time_)));

  rtt.has_td_footpaths_out_[1U].set(location_idx_t{0U}, false);
  rtt.has_td_footpaths_in_[1U].set(location_idx_t{0U}, true);
  EXPECT_TRUE(needs_rt<direction::kForward>(tt, rtt, q));
}

// PONG does not use the interval estimator: it walks the start time in search
// direction until it leaves the timetable's external interval. Its window is
// therefore much wider than the range RAPTOR one.
TEST(routing, pong_needs_rt_interval_extension) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt.extend_coverage(interval{t(8h), t(9h + 10min)});

  // Range RAPTOR cannot reach 05-01 from a month before / after.
  EXPECT_FALSE(
      needs_rt<direction::kForward>(tt, rtt, q_at(at_8(2019_y / April / 1))));
  EXPECT_FALSE(
      needs_rt<direction::kBackward>(tt, rtt, q_at(at_8(2019_y / June / 1))));

  // PONG walks forward from 04-01 up to the end of the timetable (11-01) and
  // backward from 06-01 down to its start (03-25) - both pass 05-01.
  EXPECT_TRUE(pong_needs_rt<direction::kForward>(
      tt, rtt, q_at(at_8(2019_y / April / 1)), kMinLookAhead));
  EXPECT_TRUE(pong_needs_rt<direction::kBackward>(
      tt, rtt, q_at(at_8(2019_y / June / 1)), kMinLookAhead));

  // Against the search direction it is still bounded by the query itself.
  EXPECT_FALSE(pong_needs_rt<direction::kForward>(
      tt, rtt, q_at(at_8(2019_y / June / 1)), kMinLookAhead));
  EXPECT_FALSE(pong_needs_rt<direction::kBackward>(
      tt, rtt, q_at(at_8(2019_y / April / 1)), kMinLookAhead));
}

// PONG looks `kMinLookAhead` further than `max_travel_time_`.
TEST(routing, pong_needs_rt_min_look_ahead) {
  auto const tt = load_tt();

  // Real-time data right after the end of the timetable (11-01 00:00).
  auto rtt_after = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt_after.extend_coverage(
      interval{unixtime_t{sys_days{2019_y / November / 1} + 12h},
               unixtime_t{sys_days{2019_y / November / 1} + 13h}});

  // The last start time PONG tries is just before 11-01 00:00. With one hour
  // of travel time it stops looking at 01:00 ...
  auto const q_fwd = q_at(at_8(2019_y / October / 31), 60min);
  EXPECT_FALSE(
      pong_needs_rt<direction::kForward>(tt, rtt_after, q_fwd, duration_t{0}));
  // ... but with the additional look ahead it reaches 11-02 01:00.
  EXPECT_TRUE(
      pong_needs_rt<direction::kForward>(tt, rtt_after, q_fwd, kMinLookAhead));

  // Same for the earliest start time backwards (03-25 00:00).
  auto rtt_before = rt::create_rt_timetable(tt, sys_days{kDay});
  rtt_before.extend_coverage(
      interval{unixtime_t{sys_days{2019_y / March / 24} + 11h},
               unixtime_t{sys_days{2019_y / March / 24} + 12h}});

  auto const q_bwd = q_at(at_8(2019_y / March / 26), 60min);
  EXPECT_FALSE(pong_needs_rt<direction::kBackward>(tt, rtt_before, q_bwd,
                                                   duration_t{0}));
  EXPECT_TRUE(pong_needs_rt<direction::kBackward>(tt, rtt_before, q_bwd,
                                                  kMinLookAhead));
}

TEST(routing, pong_needs_rt_no_coverage_and_profile) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});

  ASSERT_TRUE(rtt.coverage_.empty());

  auto q = q_at(at_8(kDay));
  EXPECT_FALSE(pong_needs_rt<direction::kForward>(tt, rtt, q, kMinLookAhead));

  // A profile without time dependent footpaths does not need it either ...
  q.prf_idx_ = 1U;
  EXPECT_FALSE(pong_needs_rt<direction::kForward>(tt, rtt, q, kMinLookAhead));

  // ... one with does: they are not tracked by the coverage.
  rtt.has_td_footpaths_in_[1U].resize(tt.n_locations());
  rtt.has_td_footpaths_in_[1U].set(location_idx_t{0U}, true);
  EXPECT_TRUE(pong_needs_rt<direction::kForward>(tt, rtt, q, kMinLookAhead));
}
