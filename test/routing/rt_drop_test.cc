#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"

#include "results_to_string.h"

#include "../util.h"

// Dropping the real-time timetable for a query it cannot influence. There are
// three layers, this file covers the two routing ones:
//
//   1. `rt_timetable::coverage_` - the interval real-time data is known for,
//      maintained by every function that writes to the real-time timetable.
//      Tested in `test/rt/rt_coverage_test.cc` (GTFS-RT) and, for the VDV
//      update path, in `test/rt/vdv_aus_test.cc`.
//
//   2. Driver level: each driver builds the interval its search can reach
//      (`raptor_search_interval()` / `pong_search_interval()`) and asks
//      `affects()` up front. Dropping the pointer here also strips real-time
//      from the lower bound graph, the start labels and the reconstruction, and
//      selects the `Rt = false` RAPTOR.
//
//   3. Routing core: `raptor<>::execute()` decides again per start time, from
//      the window that one execution can actually produce a journey in. This
//      is the correctness-complete layer - every driver profits from it - and
//      it adapts within a query: PONG walks its start time and can leave (or
//      enter) the coverage in the middle of a search.

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;
using namespace std::chrono_literals;
using nigiri::test::to_unix;

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
// Layer 2 - driver level: `raptor_search_interval()` /
// `pong_search_interval()`
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

  // Max start window is [05-03, 05-07). Forward it grows to [05-03, 05-12):
  // 05-01 is not in it.
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q_at(start)));

  // Backward it grows to [04-28, 05-07), which contains 05-01.
  EXPECT_TRUE(needs_rt<direction::kBackward>(tt, rtt, q_at(start)));

  // ... unless the journey may not be that long.
  EXPECT_FALSE(needs_rt<direction::kBackward>(tt, rtt, q_at(start, 60min)));
}

// Time dependent footpaths are not tracked by the coverage.
TEST(routing, needs_rt_profile) {
  auto const tt = load_tt();
  auto const rtt = rt::create_rt_timetable(tt, sys_days{kDay});

  ASSERT_TRUE(rtt.coverage_.empty());

  auto q = q_at(at_8(2019_y / June / 1));
  EXPECT_FALSE(needs_rt<direction::kForward>(tt, rtt, q));

  q.prf_idx_ = 1U;
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
  auto const rtt = rt::create_rt_timetable(tt, sys_days{kDay});

  ASSERT_TRUE(rtt.coverage_.empty());

  auto q = q_at(at_8(kDay));
  EXPECT_FALSE(pong_needs_rt<direction::kForward>(tt, rtt, q, kMinLookAhead));

  // Time dependent footpaths are not tracked by the coverage.
  q.prf_idx_ = 1U;
  EXPECT_TRUE(pong_needs_rt<direction::kForward>(tt, rtt, q, kMinLookAhead));
}

// ===========================================================================
// Layer 3 - routing core: `raptor<>::execute()`
//
// Driven through PONG because it is the driver whose start time moves: the
// answer can change from one execution to the next within a single query.
// ===========================================================================

namespace {

// Delays T1 on 2019-05-01 by 30 minutes. That is the only real-time data in
// the timetable, so the coverage stays inside 05-01.
void delay_first_day(timetable const& tt, rt_timetable& rtt) {
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(sys_days{kDay} + 3h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);
  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_trip_id("T1");
  td->set_start_date("20190501");
  td->set_start_time("06:00:00");
  auto const stu = e->mutable_trip_update()->add_stop_time_update();
  stu->set_stop_sequence(2U);
  stu->mutable_arrival()->set_delay(30 * 60);

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  ASSERT_EQ(1U, stats.total_entities_success_);
}

// Same real-time data, but the coverage is widened to the whole timetable so
// the gate never fires - the reference for what the search must return.
rt_timetable never_gated(rt_timetable const& rtt) {
  auto copy = rtt;
  copy.extend_coverage(interval{unixtime_t{sys_days{2019_y / April / 1}},
                                unixtime_t{sys_days{2019_y / June / 1}}});
  return copy;
}

// `dir == kBackward` is an `arriveBy=true` query: `start_time_` is the time at
// B and the search runs from B back to A (hence the swapped start /
// destination).
query pong_query(timetable const& tt,
                 direction const dir,
                 unixtime_t const start_time,
                 std::uint8_t const min_connection_count) {
  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const B = tt.find(location_id{"B", source_idx_t{0}}).value();
  auto const kFwd = dir == direction::kForward;
  return query{.start_time_ = start_time,
               .start_match_mode_ = location_match_mode::kEquivalent,
               .dest_match_mode_ = location_match_mode::kEquivalent,
               .use_start_footpaths_ = false,
               .start_ = {{kFwd ? A : B, 0min, 0U}},
               .destination_ = {{kFwd ? B : A, 0min, 0U}},
               .max_travel_time_ = 3h,
               .min_connection_count_ = min_connection_count,
               .extend_interval_earlier_ = !kFwd,
               .extend_interval_later_ = kFwd};
}

// One PONG search against the real coverage and one against `never_gated()`,
// asserting that the gate changed nothing about the result.
struct gated_vs_reference {
  gated_vs_reference(timetable const& tt,
                     rt_timetable const& rtt,
                     direction const dir,
                     unixtime_t const start_time,
                     std::uint8_t const min_connection_count)
      : rtt_always_{never_gated(rtt)} {
    auto s_state = search_state{};
    auto r_state = raptor_state{};

    auto const gated =
        pong_search(tt, &rtt, s_state, r_state,
                    pong_query(tt, dir, start_time, min_connection_count), dir);
    journeys_ = *gated.journeys_;
    stats_ = gated.algo_stats_;

    auto const reference =
        pong_search(tt, &rtt_always_, s_state, r_state,
                    pong_query(tt, dir, start_time, min_connection_count), dir);

    EXPECT_EQ(to_string(tt, &rtt_always_, *reference.journeys_),
              to_string(tt, &rtt, journeys_));
    EXPECT_EQ(0U, reference.algo_stats_.at("n_executes_without_rt"));
  }

  std::uint64_t with_rt() const { return stats_.at("n_executes_with_rt"); }
  std::uint64_t without_rt() const {
    return stats_.at("n_executes_without_rt");
  }

  rt_timetable rtt_always_;
  pareto_set<journey> journeys_;
  std::map<std::string, std::uint64_t> stats_;
};

}  // namespace

// PONG walks its start time forward. The first start times can reach the
// real-time data on 05-01, the later ones cannot - so the same query uses the
// real-time timetable for the first executions and drops it for the rest.
TEST(routing, rt_gate_deactivates_within_query) {
  auto const tt = load_tt();

  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  delay_first_day(tt, rtt);
  ASSERT_FALSE(rtt.coverage_.empty());

  auto const r = gated_vs_reference{tt, rtt, direction::kForward,
                                    unixtime_t{sys_days{kDay} + 2h}, 3U};

  EXPECT_GT(r.with_rt(), 0U);
  EXPECT_GT(r.without_rt(), 0U);
}

// The other way round: an `arriveBy=true` query that arrives on 05-05 starts
// far behind the real-time data on 05-01. PONG walks its start time backwards
// and eventually reaches it - the gate has to switch the real-time timetable
// back on mid-query, otherwise the delayed 05-01 trip is lost: it was taken
// off the static traffic days when its rt transport was created, so the
// static scan alone does not find it at all.
TEST(routing, rt_gate_activates_within_query) {
  auto const tt = load_tt();

  auto rtt = rt::create_rt_timetable(tt, sys_days{kDay});
  delay_first_day(tt, rtt);

  auto const r =
      gated_vs_reference{tt, rtt, direction::kBackward,
                         unixtime_t{sys_days{2019_y / May / 5} + 8h}, 10U};

  // It starts outside the coverage and walks into it.
  EXPECT_GT(r.without_rt(), 0U);
  EXPECT_GT(r.with_rt(), 0U);

  // One journey per day, and the 05-01 one is only reachable through the
  // real-time timetable - it carries the 30 minute delay (07:00 + 30min local
  // = 05:30 UTC).
  EXPECT_EQ(5U, r.journeys_.size());
  EXPECT_EQ(1U, utl::count_if(r.journeys_, [](journey const& j) {
              return j.arrival_time() == t(5h + 30min);
            }));
}
