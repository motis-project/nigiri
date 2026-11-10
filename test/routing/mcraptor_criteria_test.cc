#include "gtest/gtest.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using namespace std::chrono_literals;

// Expected pareto sets on timetables built around each criterion's trade-off,
// in both search directions. test::raptor_search adds what the other engines
// have to agree on: plain, pong and, in CUDA builds, the GPU.
namespace {

// All trips depart from their first stop at 10:00 unless noted; times are local
// (Europe/Berlin, UTC+2 on this day).
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240619,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
W0,W0,,,,,,
W1,W1,,,,,,
W2,W2,,,,,,
W3,W3,,,,,,
W9,W9,,,,,,
F0,F0,,,,,,
F9,F9,,,,,,
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S9,S9,,,,,,
G0,G0,,,,,,
G1,G1,,,,,,
G9,G9,,,,,,
A0,A0,,,,,,
A1,A1,,,,,,
A2,A2,,,,,,
A3,A3,,,,,,
A4,A4,,,,,,
A9,A9,,,,,,
L0,L0,,,,,,
L1,L1,,,,,,
L2,L2,,,,,,
L9,L9,,,,,,
O0,O0,,,,,,
O1,O1,,,,,,
O2,O2,,,,,,
O9,O9,,,,,,
T1,T1,,,,,,
T9,T9,,,,,,
M0,M0,,,,,,
M9,M9,,,,,,
U0,U0,,,,,,
U1,U1,,,,,,
U2,U2,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RAIL,MTA,RAIL,,,2
BUS,MTA,BUS,,,3
AIR,MTA,AIR,,,1100
COACH,MTA,COACH,,,202

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
BUS,D,W_FAST1,,
BUS,D,W_FAST2,,
BUS,D,W_DRY1,,
BUS,D,W_DRY2,,
AIR,D,F_AIR,,
RAIL,D,F_RAIL,,
RAIL,D,S_MIX1,,
BUS,D,S_MIX2,,
BUS,D,S_BUS1,,
BUS,D,S_BUS2,,
RAIL,D,G_RAIL,,
BUS,D,G_BUS,,
BUS,D,G_LAST,,
AIR,D,A_AIR,,
RAIL,D,A_WALK1,,
RAIL,D,A_WALK2,,
RAIL,D,A_MIX1,,
BUS,D,A_MIX2,,
BUS,D,A_BUS1,,
BUS,D,A_BUS2,,
BUS,D,L_EARLY,,
BUS,D,L_LATE,,
BUS,D,L_WALK1,,
BUS,D,L_WALK2,,
BUS,D,O_NEAR,,
BUS,D,O_FAR,,
BUS,D,O_EGRESS1,,
BUS,D,O_EGRESS2,,
BUS,D,T_A,,
BUS,D,T_B,,
BUS,D,T_C,,
AIR,D,M_AIR,,
COACH,D,M_COACH,,
RAIL,D,M_RAIL,,
AIR,D,U_AIR,,
RAIL,D,U_RAIL,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
W_FAST1,10:00:00,10:00:00,W0,0,0,0
W_FAST1,10:20:00,10:20:00,W1,1,0,0
W_FAST2,10:35:00,10:35:00,W2,0,0,0
W_FAST2,11:00:00,11:00:00,W9,1,0,0
W_DRY1,10:00:00,10:00:00,W0,0,0,0
W_DRY1,10:20:00,10:20:00,W3,1,0,0
W_DRY2,10:35:00,10:35:00,W3,0,0,0
W_DRY2,11:10:00,11:10:00,W9,1,0,0
F_AIR,10:00:00,10:00:00,F0,0,0,0
F_AIR,10:30:00,10:30:00,F9,1,0,0
F_RAIL,10:00:00,10:00:00,F0,0,0,0
F_RAIL,11:30:00,11:30:00,F9,1,0,0
S_MIX1,10:00:00,10:00:00,S0,0,0,0
S_MIX1,10:20:00,10:20:00,S1,1,0,0
S_MIX2,10:35:00,10:35:00,S1,0,0,0
S_MIX2,11:00:00,11:00:00,S9,1,0,0
S_BUS1,10:00:00,10:00:00,S0,0,0,0
S_BUS1,10:20:00,10:20:00,S2,1,0,0
S_BUS2,10:35:00,10:35:00,S2,0,0,0
S_BUS2,11:10:00,11:10:00,S9,1,0,0
G_RAIL,10:00:00,10:00:00,G0,0,0,0
G_RAIL,10:10:00,10:10:00,G1,1,0,0
G_BUS,10:00:00,10:00:00,G0,0,0,0
G_BUS,10:20:00,10:20:00,G1,1,0,0
G_LAST,10:30:00,10:30:00,G1,0,0,0
G_LAST,11:00:00,11:00:00,G9,1,0,0
A_AIR,10:00:00,10:00:00,A0,0,0,0
A_AIR,10:30:00,10:30:00,A9,1,0,0
A_WALK1,10:00:00,10:00:00,A0,0,0,0
A_WALK1,10:20:00,10:20:00,A1,1,0,0
A_WALK2,10:35:00,10:35:00,A2,0,0,0
A_WALK2,11:00:00,11:00:00,A9,1,0,0
A_MIX1,10:00:00,10:00:00,A0,0,0,0
A_MIX1,10:20:00,10:20:00,A3,1,0,0
A_MIX2,10:35:00,10:35:00,A3,0,0,0
A_MIX2,11:10:00,11:10:00,A9,1,0,0
A_BUS1,10:00:00,10:00:00,A0,0,0,0
A_BUS1,10:20:00,10:20:00,A4,1,0,0
A_BUS2,10:35:00,10:35:00,A4,0,0,0
A_BUS2,11:20:00,11:20:00,A9,1,0,0
L_EARLY,09:00:00,09:00:00,L0,0,0,0
L_EARLY,10:30:00,10:30:00,L9,1,0,0
L_LATE,10:30:00,10:30:00,L0,0,0,0
L_LATE,11:10:00,11:10:00,L9,1,0,0
L_WALK1,09:30:00,09:30:00,L0,0,0,0
L_WALK1,09:50:00,09:50:00,L1,1,0,0
L_WALK2,10:05:00,10:05:00,L2,0,0,0
L_WALK2,11:10:00,11:10:00,L9,1,0,0
O_NEAR,10:00:00,10:00:00,O1,0,0,0
O_NEAR,10:30:00,10:30:00,O9,1,0,0
O_FAR,10:00:00,10:00:00,O2,0,0,0
O_FAR,11:00:00,11:00:00,O9,1,0,0
O_EGRESS1,10:00:00,10:00:00,O0,0,0,0
O_EGRESS1,10:20:00,10:20:00,O1,1,0,0
O_EGRESS2,10:00:00,10:00:00,O0,0,0,0
O_EGRESS2,11:00:00,11:00:00,O2,1,0,0
T_A,10:00:00,10:00:00,T1,0,0,0
T_A,11:00:00,11:00:00,T9,1,0,0
T_B,10:30:00,10:30:00,T1,0,0,0
T_B,11:30:00,11:30:00,T9,1,0,0
T_C,11:00:00,11:00:00,T1,0,0,0
T_C,12:00:00,12:00:00,T9,1,0,0
M_AIR,10:00:00,10:00:00,M0,0,0,0
M_AIR,10:30:00,10:30:00,M9,1,0,0
M_COACH,10:00:00,10:00:00,M0,0,0,0
M_COACH,10:50:00,10:50:00,M9,1,0,0
M_RAIL,10:00:00,10:00:00,M0,0,0,0
M_RAIL,11:30:00,11:30:00,M9,1,0,0
U_AIR,10:00:00,10:00:00,U0,0,0,0
U_AIR,10:15:00,10:15:00,U1,1,0,0
U_AIR,10:30:00,10:30:00,U2,2,0,0
U_RAIL,10:00:00,10:00:00,U0,0,0,0
U_RAIL,11:30:00,11:30:00,U2,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
W1,W2,2,600
A1,A2,2,600
L1,L2,2,600
)");
}

timetable const& test_tt() {
  static auto const tt =
      loader::load(test_files(), {2024_y / June / 18, 2024_y / June / 20});
  return tt;
}

constexpr auto const kUtcOffset = 2h;

unixtime_t at(int const hour, int const minute) {
  return unixtime_t{sys_days{2024_y / June / 19}} + std::chrono::hours{hour} +
         std::chrono::minutes{minute} - kUtcOffset;
}

std::string hhmm(unixtime_t const t) {
  auto const m =
      (t + kUtcOffset - unixtime_t{sys_days{2024_y / June / 19}}).count();
  return fmt::format("{:02}:{:02}", m / 60, m % 60);
}

std::vector<std::string> rows(pareto_set<journey> const& js) {
  auto v = std::vector<std::string>{};
  for (auto const& j : js) {
    v.push_back(fmt::format("{} {} t{} c{} f{} s{}", hhmm(j.departure_time()),
                            hhmm(j.arrival_time()), j.transfers_,
                            j.criteria_cost_, j.criteria_mode_filter_ ? 1 : 0,
                            j.criteria_mode_switches_));
  }
  std::sort(begin(v), end(v));
  return v;
}

location_idx_t stop_loc(std::string_view const id) {
  return test_tt().find({id, source_idx_t{0U}}).value();
}

// A query in both directions. Backward searches start from the destination and
// anchor the window on the arrival.
struct scenario {
  query make(direction const dir) const {
    auto q = query{};
    q.start_match_mode_ = intermodal_ ? location_match_mode::kIntermodal
                                      : location_match_mode::kExact;
    q.dest_match_mode_ = q.start_match_mode_;
    q.start_ = from_;
    q.destination_ = to_;
    q.td_start_ = td_from_;
    q.td_dest_ = td_to_;
    if (dir == direction::kBackward) {
      q.flip_dir();
    }
    q.start_time_ = dir == direction::kForward ? deps_ : arrs_;
    return q;
  }

  std::vector<offset> from_{}, to_{};
  td_offsets_t td_from_{}, td_to_{};
  interval<unixtime_t> deps_{}, arrs_{};
  bool intermodal_{false};
};

// `duration` from `from` on, until the next entry. valid_from is when the walk
// begins, whichever way the search runs.
td_offset td(unixtime_t const from, duration_t const duration) {
  return {.valid_from_ = from,
          .duration_ = duration,
          .transport_mode_payload_ = 0U};
}

// reachable for walks beginning in the minute at `from` only
std::vector<td_offset> one_minute(unixtime_t const from,
                                  duration_t const duration) {
  return {td(unixtime_t{0h}, footpath::kMaxDuration), td(from, duration),
          td(from + 1min, footpath::kMaxDuration)};
}

using dirs_t = std::initializer_list<direction>;

template <typename State>
void expect_rows(scenario const& s,
                 std::vector<std::string> const& expected,
                 dirs_t const dirs = {direction::kForward,
                                      direction::kBackward},
                 rt_timetable const* rtt = nullptr) {
  for (auto const dir : dirs) {
    auto ss = search_state{};
    auto state = State{};
    auto const r =
        routing::raptor_search(test_tt(), rtt, ss, state, s.make(dir), dir);
    EXPECT_EQ(expected, rows(*r.journeys_))
        << (dir == direction::kForward ? "forward" : "backward");
  }
}

// plain raptor's result, with pong, mcraptor and the GPU compared against it
void expect_engines(scenario const& s,
                    std::vector<std::string> const& expected,
                    dirs_t const dirs = {direction::kForward,
                                         direction::kBackward},
                    rt_timetable const* rtt = nullptr) {
  for (auto const dir : dirs) {
    EXPECT_EQ(expected,
              rows(test::raptor_search(test_tt(), rtt, s.make(dir), dir)))
        << (dir == direction::kForward ? "forward" : "backward");
  }
}

scenario between(std::string_view const from,
                 std::string_view const to,
                 interval<unixtime_t> const deps,
                 interval<unixtime_t> const arrs) {
  return {.from_ = {{stop_loc(from), 0min, 0U}},
          .to_ = {{stop_loc(to), 0min, 0U}},
          .deps_ = deps,
          .arrs_ = arrs};
}

}  // namespace

// footpath W1 -> W2: 11:00 with 10 minutes of walking, 11:10 without
TEST(mcraptor_criteria, non_transit_trades_walking_for_arrival) {
  auto const s =
      between("W0", "W9", {at(9, 30), at(10, 30)}, {at(10, 45), at(11, 30)});
  expect_engines(s, {"10:00 11:00 t1 c0 f0 s0"});
  expect_rows<mcraptor_state>(s, {"10:00 11:00 t1 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(
      s, {"10:00 11:00 t1 c10 f0 s0", "10:00 11:10 t1 c0 f0 s0"});
}

// the flight is much faster, the train is the best without an avoided class
TEST(mcraptor_criteria, mode_filter_keeps_best_without_avoided_class) {
  auto const s =
      between("F0", "F9", {at(9, 30), at(10, 30)}, {at(10, 15), at(11, 45)});
  expect_engines(s, {"10:00 10:30 t0 c0 f0 s0"});
  expect_rows<mcraptor_mode_filter_state>(
      s, {"10:00 10:30 t0 c0 f1 s0", "10:00 11:30 t0 c0 f0 s0"});
}

// air 10:30, coach 10:50, rail 11:30: each mask motis offers (minimizeWithout:
// AIR, COACH, both), on the CPU and, through expect_engines, the GPU
TEST(mcraptor_criteria, mode_filter_avoids_each_mask) {
  auto const s =
      between("M0", "M9", {at(9, 30), at(10, 30)}, {at(10, 15), at(11, 45)});
  auto const check = [&](clasz_mask_t const avoided,
                         std::vector<std::string> const& expected) {
    mode_filter_dim::set_avoided(avoided);
    expect_engines(s, {"10:00 10:30 t0 c0 f0 s0"});
    expect_rows<mcraptor_mode_filter_state>(s, expected);
    expect_rows<mcraptor_non_transit_mode_filter_mode_switches_state>(s,
                                                                      expected);
  };
  check(to_mask(clasz::kAir),
        {"10:00 10:30 t0 c0 f1 s0", "10:00 10:50 t0 c0 f0 s0"});
  check(to_mask(clasz::kCoach), {"10:00 10:30 t0 c0 f0 s0"});
  check(to_mask(clasz::kAir) | to_mask(clasz::kCoach),
        {"10:00 10:30 t0 c0 f1 s0", "10:00 11:30 t0 c0 f0 s0"});
  mode_filter_dim::set_avoided(to_mask(clasz::kAir));  // the default
}

// A ride has its route's class (route_clasz_, as raptor's clasz filter), also
// when alighted behind the first section (U0 -> U2 over U1) and as an rt run
TEST(mcraptor_criteria, ride_has_the_route_class) {
  auto const s =
      between("U0", "U2", {at(9, 30), at(10, 30)}, {at(10, 15), at(11, 45)});
  auto const check = [&](rt_timetable const* rtt,
                         std::string const& flight_plain,
                         std::string const& flight) {
    auto const both = dirs_t{direction::kForward, direction::kBackward};
    expect_engines(s, {flight_plain}, both, rtt);
    auto expected = std::vector<std::string>{flight, "10:00 11:30 t0 c0 f0 s0"};
    std::sort(begin(expected), end(expected));  // as rows()
    expect_rows<mcraptor_mode_filter_state>(s, expected, both, rtt);
    expect_rows<mcraptor_non_transit_mode_filter_mode_switches_state>(
        s, expected, both, rtt);
  };
  check(nullptr, "10:00 10:30 t0 c0 f0 s0", "10:00 10:30 t0 c0 f1 s0");

  auto rtt = rt::create_rt_timetable(test_tt(), sys_days{2024_y / June / 19});
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          at(9, 0).time_since_epoch())
          .count()));
  auto const e = msg.add_entity();
  e->set_id("1");
  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_start_time("10:00:00");
  td->set_start_date("20240619");
  td->set_trip_id("U_AIR");
  auto const stop_update = e->mutable_trip_update()->add_stop_time_update();
  stop_update->set_stop_sequence(0U);
  stop_update->mutable_departure()->set_delay(10 * 60);
  ASSERT_EQ(rt::gtfsrt_update_msg(test_tt(), rtt, source_idx_t{0}, "tag", msg)
                .total_entities_success_,
            1U);
  check(&rtt, "10:10 10:40 t0 c0 f0 s0", "10:10 10:40 t0 c0 f1 s0");
}

// rail + bus arrives at 11:00, bus + bus at 11:10 without changing group
TEST(mcraptor_criteria, mode_switches_trades_switch_for_arrival) {
  auto const s =
      between("S0", "S9", {at(9, 30), at(10, 30)}, {at(10, 45), at(11, 30)});
  expect_engines(s, {"10:00 11:00 t1 c0 f0 s0"});
  expect_rows<mcraptor_mode_switches_state>(
      s, {"10:00 11:00 t1 c0 f0 s1", "10:00 11:10 t1 c0 f0 s0"});
}

// At G1 the rail arrival is earlier with as few switches as the bus arrival.
// Only the bus arrival avoids a switch on the last ride, so it must survive.
TEST(mcraptor_criteria, mode_switches_keeps_later_label_of_other_group) {
  auto const s =
      between("G0", "G9", {at(9, 30), at(10, 30)}, {at(10, 45), at(11, 30)});
  expect_rows<mcraptor_mode_switches_state>(s, {"10:00 11:00 t1 c0 f0 s0"});
  expect_engines(s, {"10:00 11:00 t1 c0 f0 s0"});
}

// A0 -> A9: flight 10:30 | rail + 10 min walk + rail 11:00 | rail + bus 11:10 |
// bus + bus 11:20. Each criteria set keeps the trade-offs it can tell apart.
TEST(mcraptor_criteria, criteria_sets_combine) {
  auto const s =
      between("A0", "A9", {at(9, 30), at(10, 30)}, {at(10, 15), at(11, 30)});
  auto const flight_plain = "10:00 10:30 t0 c0 f0 s0";
  auto const flight = "10:00 10:30 t0 c0 f1 s0";
  expect_engines(s, {flight_plain});
  expect_rows<mcraptor_non_transit_state>(s, {flight_plain});
  expect_rows<mcraptor_mode_switches_state>(s, {flight_plain});
  expect_rows<mcraptor_mode_filter_state>(s,
                                          {flight, "10:00 11:00 t1 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_mode_filter_state>(
      s, {flight, "10:00 11:00 t1 c10 f0 s0", "10:00 11:10 t1 c0 f0 s0"});
  expect_rows<mcraptor_mode_filter_mode_switches_state>(
      s, {flight, "10:00 11:00 t1 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_mode_filter_mode_switches_state>(
      s, {flight, "10:00 11:00 t1 c10 f0 s0", "10:00 11:10 t1 c0 f0 s1",
          "10:00 11:20 t1 c0 f0 s0"});
}

// The walking journey (09:30, arrives 11:10) departs earlier than the direct
// 10:30 for the same arrival, so it loses across start times.
TEST(mcraptor_criteria, later_departure_dominates_across_the_range) {
  auto const s =
      between("L0", "L9", {at(8, 30), at(11, 0)}, {at(10, 0), at(11, 30)});
  auto const expected = std::vector<std::string>{"09:00 10:30 t0 c0 f0 s0",
                                                 "10:30 11:10 t0 c0 f0 s0"};
  expect_engines(s, expected);
  expect_rows<mcraptor_non_transit_state>(s, expected);
  expect_rows<mcraptor_non_transit_mode_filter_mode_switches_state>(s,
                                                                    expected);
}

// 10 minute walk to O1 (arrives 10:30) or none from O2 (arrives 11:00)
TEST(mcraptor_criteria, start_offset_counts_as_walking) {
  auto const s = scenario{
      .from_ = {{stop_loc("O1"), 10min, 0U}, {stop_loc("O2"), 0min, 0U}},
      .to_ = {{stop_loc("O9"), 0min, 0U}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 15), at(11, 30)},
      .intermodal_ = true,
  };
  auto const expected = std::vector<std::string>{"09:50 10:30 t0 c0 f0 s0",
                                                 "10:00 11:00 t0 c0 f0 s0"};
  expect_engines(s, expected);
  expect_rows<mcraptor_non_transit_state>(
      s, {"09:50 10:30 t0 c10 f0 s0", "10:00 11:00 t0 c0 f0 s0"});
}

// leave O0 for O1 + 10 minute walk (arrives 10:30) or O2 (arrives 11:00)
TEST(mcraptor_criteria, destination_offset_counts_as_walking) {
  auto const s = scenario{
      .from_ = {{stop_loc("O0"), 0min, 0U}},
      .to_ = {{stop_loc("O1"), 10min, 0U}, {stop_loc("O2"), 0min, 0U}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 15), at(11, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:00 10:30 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(
      s, {"10:00 10:30 t0 c10 f0 s0", "10:00 11:00 t0 c0 f0 s0"});
}

// ---- time-dependent offsets -------------------------------------------------

// T1 -> T9 runs at 10:00, 10:30 and 11:00, one hour each. The walk takes 60
// minutes from 09:50 on: 09:49 still gets the 10:00, and the 10:30 is out of
// reach for every later start, so the next journey leaves at 10:00 for the
// 11:00.
TEST(td_offset_engines, start_gets_slower_between_departures) {
  auto const s = scenario{
      .to_ = {{stop_loc("T9"), 0min, 0U}},
      .td_from_ = {{stop_loc("T1"),
                    {td(unixtime_t{0h}, 5min), td(at(9, 50), 60min)}}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 45), at(12, 15)},
      .intermodal_ = true,
  };
  expect_engines(s, {"09:49 11:00 t0 c0 f0 s0", "10:00 12:00 t0 c0 f0 s0"});
}

// The egress closes at 11:10 and reopens at 12:00. The 10:30 trip arrives while
// it is closed and waits, which the 11:00 trip catches with a later departure.
TEST(td_offset_engines, destination_closes_and_reopens) {
  auto const s = scenario{
      .from_ = {{stop_loc("T1"), 0min, 0U}},
      .td_to_ = {{stop_loc("T9"),
                  {td(unixtime_t{0h}, 5min),
                   td(at(11, 10), footpath::kMaxDuration),
                   td(at(12, 0), 5min)}}},
      .deps_ = {at(9, 30), at(11, 30)},
      .arrs_ = {at(10, 45), at(12, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:00 11:05 t0 c0 f0 s0", "11:00 12:05 t0 c0 f0 s0"});
}

TEST(td_offset_engines, both_ends) {
  auto const s = scenario{
      .td_from_ = {{stop_loc("T1"), one_minute(at(9, 40), 5min)}},
      .td_to_ = {{stop_loc("T9"),
                  {td(unixtime_t{0h}, 5min),
                   td(at(11, 10), footpath::kMaxDuration)}}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 45), at(11, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"09:40 11:05 t0 c0 f0 s0"});
}

// as start_offset_counts_as_walking, with the walk to O1 time-dependent
TEST(td_offset_engines, start_counts_as_walking) {
  auto const s = scenario{
      .from_ = {{stop_loc("O2"), 0min, 0U}},
      .to_ = {{stop_loc("O9"), 0min, 0U}},
      .td_from_ = {{stop_loc("O1"), {td(unixtime_t{0h}, 10min)}}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 15), at(11, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"09:50 10:30 t0 c0 f0 s0", "10:00 11:00 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(
      s, {"09:50 10:30 t0 c10 f0 s0", "10:00 11:00 t0 c0 f0 s0"});
}

// as destination_offset_counts_as_walking, with the walk from O1 time-dependent
TEST(td_offset_engines, destination_counts_as_walking) {
  auto const s = scenario{
      .from_ = {{stop_loc("O0"), 0min, 0U}},
      .to_ = {{stop_loc("O2"), 0min, 0U}},
      .td_to_ = {{stop_loc("O1"), {td(unixtime_t{0h}, 10min)}}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 15), at(11, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:00 10:30 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(
      s, {"10:00 10:30 t0 c10 f0 s0", "10:00 11:00 t0 c0 f0 s0"});
}

// Walk 5 minutes at 09:40, then wait 15 minutes for the 10:00.
TEST(td_offset_engines, waiting_after_the_start_walk_is_not_walking) {
  auto const s = scenario{
      .to_ = {{stop_loc("O9"), 0min, 0U}},
      .td_from_ = {{stop_loc("O1"), one_minute(at(9, 40), 5min)}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 15), at(10, 45)},
      .intermodal_ = true,
  };
  expect_engines(s, {"09:40 10:30 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(s, {"09:40 10:30 t0 c5 f0 s0"});
}

// Arrive at 10:20, wait 30 minutes, walk 5 minutes at 10:50.
TEST(td_offset_engines, waiting_before_the_destination_walk_is_not_walking) {
  auto const s = scenario{
      .from_ = {{stop_loc("O0"), 0min, 0U}},
      .td_to_ = {{stop_loc("O1"), one_minute(at(10, 50), 5min)}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 40), at(11, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:00 10:55 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(s, {"10:00 10:55 t0 c5 f0 s0"});
}

// O1 arrives 10:20 + 10 minute walk. O2 arrives 11:00, waits, walks 5 minutes
// at 11:10. Counted as walking, the wait would make the second one dominated.
TEST(td_offset_engines, waiting_keeps_a_shorter_walk_alive) {
  auto const s = scenario{
      .from_ = {{stop_loc("O0"), 0min, 0U}},
      .td_to_ = {{stop_loc("O1"), {td(unixtime_t{0h}, 10min)}},
                 {stop_loc("O2"), one_minute(at(11, 10), 5min)}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(10, 15), at(11, 30)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:00 10:30 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(
      s, {"10:00 10:30 t0 c10 f0 s0", "10:00 11:15 t0 c5 f0 s0"});
}

// The egress gets slower at 11:20. A backward search only sees the arrivals in
// its window: 12:30, by the 10:30 trip and its 60 minute walk.
TEST(td_offset_engines, backward_window_selects_by_arrival) {
  auto const egress =
      std::vector<td_offset>{td(unixtime_t{0h}, 5min), td(at(11, 20), 60min)};
  auto const all = scenario{
      .from_ = {{stop_loc("T1"), 0min, 0U}},
      .td_to_ = {{stop_loc("T9"), egress}},
      .deps_ = {at(9, 30), at(11, 30)},
      .arrs_ = {at(10, 45), at(13, 15)},
      .intermodal_ = true,
  };
  expect_engines(all, {"10:00 11:05 t0 c0 f0 s0", "10:30 12:30 t0 c0 f0 s0",
                       "11:00 13:00 t0 c0 f0 s0"});

  auto narrow = all;
  narrow.arrs_ = {at(12, 15), at(12, 45)};
  expect_engines(narrow, {"10:30 12:30 t0 c0 f0 s0"}, {direction::kBackward});
  expect_rows<mcraptor_non_transit_state>(narrow, {"10:30 12:30 t0 c60 f0 s0"},
                                          {direction::kBackward});
}

// The egress is open for walks at 11:50 only: the 10:00 and the 10:30 trip wait
// for it, the 11:00 arrives too late. The later departure wins.
TEST(td_offset_engines, trips_wait_for_a_late_egress) {
  auto const s = scenario{
      .from_ = {{stop_loc("T1"), 0min, 0U}},
      .td_to_ = {{stop_loc("T9"), one_minute(at(11, 50), 5min)}},
      .deps_ = {at(9, 30), at(11, 30)},
      .arrs_ = {at(11, 45), at(12, 15)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:30 11:55 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(s, {"10:30 11:55 t0 c5 f0 s0"});
}

// The access is open for walks at 10:10 only, in time for the 10:30 and the
// 11:00 trip. The earlier arrival wins.
TEST(td_offset_engines, later_trips_after_an_early_access) {
  auto const s = scenario{
      .to_ = {{stop_loc("T9"), 0min, 0U}},
      .td_from_ = {{stop_loc("T1"), one_minute(at(10, 10), 5min)}},
      .deps_ = {at(9, 30), at(10, 30)},
      .arrs_ = {at(11, 15), at(12, 15)},
      .intermodal_ = true,
  };
  expect_engines(s, {"10:10 11:30 t0 c0 f0 s0"});
  expect_rows<mcraptor_non_transit_state>(s, {"10:10 11:30 t0 c5 f0 s0"});
}

// Fixed start time, 5 minutes before there is anything to walk to in time: the
// wait is part of the journey but not of the walk.
TEST(td_offset_engines, fixed_departure_time) {
  auto const s = scenario{
      .to_ = {{stop_loc("T9"), 0min, 0U}},
      .td_from_ = {{stop_loc("T1"), {td(unixtime_t{0h}, 5min)}}},
      .intermodal_ = true,
  };
  auto q = s.make(direction::kForward);
  q.start_time_ = at(9, 50);
  EXPECT_EQ(
      std::vector<std::string>{"09:50 11:00 t0 c0 f0 s0"},
      rows(test::raptor_search(test_tt(), nullptr, q, direction::kForward)));
  auto ss = search_state{};
  auto state = mcraptor_non_transit_state{};
  EXPECT_EQ(std::vector<std::string>{"09:50 11:00 t0 c5 f0 s0"},
            rows(*routing::raptor_search(test_tt(), nullptr, ss, state, q,
                                         direction::kForward)
                      .journeys_));
}

// mirrored: the 11:00 trip arrives at 12:00, 5 minute walk, arrive by 12:10
TEST(td_offset_engines, fixed_arrival_time) {
  auto const s = scenario{
      .from_ = {{stop_loc("T1"), 0min, 0U}},
      .td_to_ = {{stop_loc("T9"), {td(unixtime_t{0h}, 5min)}}},
      .intermodal_ = true,
  };
  auto q = s.make(direction::kBackward);
  q.start_time_ = at(12, 10);
  EXPECT_EQ(
      std::vector<std::string>{"11:00 12:10 t0 c0 f0 s0"},
      rows(test::raptor_search(test_tt(), nullptr, q, direction::kBackward)));
  auto ss = search_state{};
  auto state = mcraptor_non_transit_state{};
  EXPECT_EQ(std::vector<std::string>{"11:00 12:10 t0 c5 f0 s0"},
            rows(*routing::raptor_search(test_tt(), nullptr, ss, state, q,
                                         direction::kBackward)
                      .journeys_));
}
