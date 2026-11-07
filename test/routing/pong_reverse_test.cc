#include "gtest/gtest.h"

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using namespace nigiri::test;
using namespace std::chrono_literals;

// A ---local every 10 min---> B -> C -> D   (10 min per hop)
// A ---express every 30 min-> D             (15 min)
// C ---feeder every 20 min--> E             (10 min)
// D ---feeder every 25 min--> E             (10 min)
// E ---shuttle every 15 min-> F             (10 min)
// -> A->F has journeys with 1..3 transfers, A->D with 0 and 2 transfers
std::string pong_reverse_timetable() {
  auto s = std::string{R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,0.0,,,
B,B,,0.0,0.1,,,
C,C,,0.0,0.2,,,
D,D,,0.0,0.3,,,
E,E,,0.1,0.2,,,
F,F,,0.2,0.2,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DAILY,1,1,1,1,1,1,1,20260601,20260630

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
L,DTA,L,local,,3
X,DTA,X,express,,2
CE,DTA,CE,feeder C,,3
DE,DTA,DE,feeder D,,3
EF,DTA,EF,shuttle,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
)"};
  auto st = std::string{R"(
# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
)"};
  auto const hhmm = [](unsigned const m) {
    return fmt::format("{:02}:{:02}:00", m / 60, m % 60);
  };
  auto const line = [&](std::string const& route, unsigned const freq,
                        unsigned const first, unsigned const last,
                        std::vector<std::pair<char, unsigned>> const& stops) {
    for (auto t = first, i = 0U; t <= last; t += freq, ++i) {
      auto const trip = fmt::format("{}_{}", route, i);
      s += fmt::format("{},DAILY,{},{},\n", route, trip, trip);
      auto seq = 0U;
      for (auto const& [stop, offset] : stops) {
        st += fmt::format("{},{},{},{},{},0,0\n", trip, hhmm(t + offset),
                          hhmm(t + offset), stop, seq++);
      }
    }
  };
  line("L", 10, 5 * 60, 22 * 60, {{'A', 0}, {'B', 10}, {'C', 20}, {'D', 30}});
  line("X", 30, 5 * 60 + 7, 22 * 60, {{'A', 0}, {'D', 15}});
  line("CE", 20, 5 * 60 + 3, 22 * 60, {{'C', 0}, {'E', 10}});
  line("DE", 25, 5 * 60 + 9, 22 * 60, {{'D', 0}, {'E', 10}});
  line("EF", 15, 5 * 60 + 1, 23 * 60, {{'E', 0}, {'F', 10}});
  return s + st;
}

timetable pong_reverse_tt() {
  static auto const files = mem_dir::read(pong_reverse_timetable());
  return loader::load(
      files, {sys_days{2026_y / June / 01}, sys_days{2026_y / July / 01}});
}

query pong_reverse_query(timetable const& tt,
                         std::string_view from,
                         std::string_view to,
                         interval<unixtime_t> const iv,
                         bool const extend_earlier,
                         bool const extend_later,
                         unsigned const min_connection_count) {
  auto q = query{
      .start_time_ = iv,
      .start_match_mode_ = location_match_mode::kEquivalent,
      .dest_match_mode_ = location_match_mode::kEquivalent,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, source_idx_t{0}}),
                  0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {to, source_idx_t{0}}),
                        0_minutes, 0U}},
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_earlier,
      .extend_interval_later_ = extend_later};
  return q;
}

std::vector<std::tuple<unixtime_t, unixtime_t, std::uint8_t>> keys_in(
    pareto_set<journey> const& js, interval<unixtime_t> const iv) {
  auto v = std::vector<std::tuple<unixtime_t, unixtime_t, std::uint8_t>>{};
  for (auto const& j : js) {
    if (iv.contains(j.start_time_)) {
      v.emplace_back(j.start_time_, j.dest_time_, j.transfers_);
    }
  }
  std::sort(begin(v), end(v));
  return v;
}

// reference and reverse page agree on (departure, arrival, transfers) inside
// the window on both engines; the legs may differ between equal-time paths
void expect_same_keys(timetable const& tt,
                      query const& ref_q,
                      query const& rev_q,
                      direction const dir,
                      interval<unixtime_t> const iv,
                      std::string const& what) {
  auto ss = routing::search_state{};
  auto rs = routing::raptor_state{};
  auto const ref =
      keys_in(*pong_search(tt, nullptr, ss, rs, ref_q, dir).journeys_, iv);
  EXPECT_EQ(
      ref, keys_in(*pong_search(tt, nullptr, ss, rs, rev_q, dir).journeys_, iv))
      << what << " (cpu)";
#if defined(NIGIRI_CUDA)
  if (gpu::gpu_supported(rev_q, nullptr)) {
    auto gpu_tt = gpu::gpu_timetable{tt};
    auto gpu_rs = gpu::gpu_raptor_state{gpu_tt};
    EXPECT_EQ(
        ref,
        keys_in(*pong_search(tt, nullptr, ss, gpu_rs, rev_q, dir).journeys_,
                iv))
        << what << " (gpu)";
  }
#endif
}

// the reverse page over a fixed window has to yield the same connections as
// the standard pong over that window (the standard pong's dominance already
// takes the connections beyond the window into account)
TEST(routing, pong_reverse_fwd_earlier_equals_fixed_window) {
  auto const tt = pong_reverse_tt();
  auto const day = sys_days{2026_y / June / 10};
  for (auto const& [from, to] :
       {std::pair{"A", "F"}, std::pair{"A", "D"}, std::pair{"B", "F"}}) {
    for (auto h = 7; h <= 20; h += 2) {
      auto const cursor = unixtime_t{day + std::chrono::hours{h}};
      auto const iv = interval{cursor - 90_minutes, cursor};
      expect_same_keys(tt,
                       pong_reverse_query(tt, from, to, iv, false, false, 0U),
                       pong_reverse_query(tt, from, to, iv, true, false, 0U),
                       direction::kForward, iv,
                       fmt::format("{} -> {} cursor {}", from, to, cursor));
    }
  }
}

TEST(routing, pong_reverse_bwd_later_equals_fixed_window) {
  auto const tt = pong_reverse_tt();
  auto const day = sys_days{2026_y / June / 10};
  for (auto const& [from, to] :
       {std::pair{"F", "A"}, std::pair{"D", "A"}, std::pair{"F", "B"}}) {
    for (auto h = 7; h <= 20; h += 2) {
      auto const cursor = unixtime_t{day + std::chrono::hours{h}};
      auto const iv = interval{cursor, cursor + 90_minutes};
      // arriveBy: the query is in search orientation (start = destination)
      expect_same_keys(tt,
                       pong_reverse_query(tt, from, to, iv, false, false, 0U),
                       pong_reverse_query(tt, from, to, iv, false, true, 0U),
                       direction::kBackward, iv,
                       fmt::format("{} <- {} cursor {}", from, to, cursor));
    }
  }
}

// with a minimum connection count the page extends until it holds that many
TEST(routing, pong_reverse_min_connection_count) {
  auto const tt = pong_reverse_tt();
  auto const cursor = unixtime_t{sys_days{2026_y / June / 10} + 12h};
  auto const iv = interval{cursor - 15_minutes, cursor};
  auto search_state = routing::search_state{};
  auto algo_state = routing::raptor_state{};
  auto const r =
      pong_search(tt, nullptr, search_state, algo_state,
                  pong_reverse_query(tt, "A", "F", iv, true, false, 5U),
                  direction::kForward);
  EXPECT_GE(r.journeys_->size(), 5U);
  EXPECT_LE(r.interval_.from_, iv.from_);
  EXPECT_EQ(r.interval_.to_, cursor);
  for (auto const& j : *r.journeys_) {
    EXPECT_TRUE(r.interval_.contains(j.start_time_));
  }
  // the same window searched with the standard pong: nothing missing inside
  auto const ref = search_pong(
      tt, nullptr,
      pong_reverse_query(tt, "A", "F", r.interval_, false, false, 0U),
      direction::kForward);
  EXPECT_EQ(keys_in(ref, r.interval_), keys_in(*r.journeys_, r.interval_));
}
