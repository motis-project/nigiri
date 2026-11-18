#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <span>

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/load.h"
#include "nigiri/loader/station_graph.h"
#include "nigiri/footpath_restriction.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/station_paths.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;
using fr = nigiri::footpath_restriction;

// Station S: platforms P1, P2 (P2 with boarding area BA2), entrance E.
// P1 -> P2:
//   (a) stairs (60 s) + walkway (50 s): 110 s walking -> 2 min,
//       width 2 m + unknown (stairs without min_width)
//   (b) elevator (90 s fixed) + ramp 8 % (100 s): 4 min, fixed part 2 min,
//       width 1 m
// T1: A 10:00 -> P1 10:10
// P2 -> C: 10:12 (T2), 10:14 (T3), 10:16 (T4), 10:18 (T5)
// T6: A 10:00 -> C 10:40 direct
constexpr auto const kPathwaysFiles = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,level_id
A,A,48.1,9.1,0,,
C,C,48.2,9.2,0,,
S,S,48.0,9.0,1,,
P1,P1,48.0,9.0,0,S,L0
P2,P2,48.001,9.0,0,S,L0
BA2,BA2,48.001,9.0,4,P2,L0
N1,N1,,,3,S,L-1
N2,N2,,,3,S,L-1
E,E,48.0,9.001,2,S,L0

# levels.txt
level_id,level_index,level_name
L0,0,Street
L-1,-1,Underground

# pathways.txt
pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,length,traversal_time,stair_count,max_slope,min_width,signposted_as,reversed_signposted_as
PW1,P1,N1,2,1,,60,30,,,,
PW2,N1,BA2,1,1,60,,,,2.0,Gleis 2,
PW3,P1,N2,5,1,,90,,,1.1,,
PW4,N2,BA2,1,1,120,100,,0.08,1.0,,
PW5,P1,E,1,1,20,,,,,Ausgang,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R2,S1,T3,,
R2,S1,T4,,
R2,S1,T5,,
R3,S1,T6,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,P1,1
T2,10:12:00,10:12:00,P2,0
T2,10:20:00,10:20:00,C,1
T3,10:14:00,10:14:00,P2,0
T3,10:22:00,10:22:00,C,1
T4,10:16:00,10:16:00,P2,0
T4,10:24:00,10:24:00,C,1
T5,10:18:00,10:18:00,P2,0
T5,10:26:00,10:26:00,C,1
T6,10:00:00,10:00:00,A,0
T6,10:40:00,10:40:00,C,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const kNoStairs = fr::kStairs;
constexpr auto const kNoStairsElevator =
    static_cast<std::uint8_t>(fr::kStairs | fr::kElevator);

// Earliest arrival at C (local time HH:MM) departing A at 10:00.
std::string pathways_earliest_arrival(timetable const& tt,
                                      routing::transfer_time_settings const& s,
                                      direction const dir,
                                      rt_timetable const* rtt = nullptr) {
  auto const fwd = dir == direction::kForward;
  auto q = routing::query{
      .start_time_ = fwd ? routing::start_time_t{interval{
                               sys_days{2019_y / May / 1} + 8h,
                               sys_days{2019_y / May / 1} + 8h + 1min}}
                         : routing::start_time_t{interval{
                               sys_days{2019_y / May / 1} + 8h + 10min,
                               sys_days{2019_y / May / 1} + 8h + 41min}},
      .transfer_time_settings_ = s};
  auto const results = raptor_search(tt, rtt, std::move(q), fwd ? "A" : "C",
                                     fwd ? "C" : "A", "", dir);
  auto best = std::optional<unixtime_t>{};
  for (auto const& j : results) {
    auto const arr = std::max(j.start_time_, j.dest_time_);
    if (!best.has_value() || arr < *best) {
      best = arr;
    }
  }
  return best.has_value() ? date::format("%H:%M", *best + 2h) : "none";
}

// Restricted tail of l's footpath list (default profile).
std::span<footpath const> pathways_restricted(timetable const& tt,
                                              bool const is_out,
                                              location_idx_t const l) {
  auto const b = (is_out ? tt.locations_.footpaths_out_
                         : tt.locations_.footpaths_in_)[kDefaultProfile][l];
  auto const split = tt.locations_.restricted_begin(kDefaultProfile, is_out, l);
  return std::span{b.data() + split, b.size() - split};
}

routing::transfer_time_settings pathways_forbid(std::uint8_t const flags) {
  return {.restrictions_ = {.forbidden_ = flags}};
}

TEST(loader, pathways_restricted_footpaths) {
  auto const tt = loader::load(loader::mem_dir::read(kPathwaysFiles),
                               {2019_y / May / 1, 2019_y / May / 2});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };

  // P1 -> P2 only via the station graph (restricted part of the list).
  auto const all = tt.locations_.footpaths_out_[kDefaultProfile][l("P1")];
  auto const split =
      tt.locations_.restricted_begin(kDefaultProfile, true, l("P1"));
  for (auto i = 0U; i != split; ++i) {
    EXPECT_NE(l("P2"), all[i].target());
  }
  auto const fps = pathways_restricted(tt, true, l("P1"));
  auto const rs =
      tt.locations_.footpath_restrictions_out_[kDefaultProfile][l("P1")];
  ASSERT_EQ(2U, fps.size());
  ASSERT_EQ(2U, rs.size());

  EXPECT_EQ(l("P2"), fps[0].target());
  EXPECT_EQ(duration_t{2}, fps[0].duration());
  EXPECT_EQ(fr::kStairs, rs[0].flags_);
  EXPECT_EQ(0U, rs[0].slope());
  EXPECT_FALSE(rs[0].has_unknown_slope());
  EXPECT_EQ(fr::to_width(2.0), rs[0].width());
  EXPECT_TRUE(rs[0].has_unknown_width());
  EXPECT_EQ(0U, rs[0].fixed_seconds());

  EXPECT_EQ(l("P2"), fps[1].target());
  EXPECT_EQ(duration_t{4}, fps[1].duration());
  EXPECT_EQ(fr::kElevator | fr::kRamp, rs[1].flags_);
  EXPECT_EQ(fr::to_slope(8.0), rs[1].slope());
  EXPECT_EQ(fr::to_width(1.0), rs[1].width());
  EXPECT_FALSE(rs[1].has_unknown_width());
  EXPECT_EQ(90U, rs[1].fixed_seconds());

  // In-lists mirror the out-lists.
  auto const in = pathways_restricted(tt, false, l("P2"));
  auto const in_rs =
      tt.locations_.footpath_restrictions_in_[kDefaultProfile][l("P2")];
  ASSERT_EQ(2U, in.size());
  EXPECT_EQ(l("P1"), in[0].target());
  EXPECT_EQ(rs[0], in_rs[0]);
  EXPECT_EQ(rs[1], in_rs[1]);

  // Settings-aware shortest transfer.
  auto const shortest =
      routing::shortest_transfer(tt, nullptr, kDefaultProfile,
                                 pathways_forbid(kNoStairs), l("P1"), l("P2"));
  ASSERT_TRUE(shortest.has_value());
  EXPECT_EQ(duration_t{4}, shortest->first);
  EXPECT_FALSE(routing::shortest_transfer(tt, nullptr, kDefaultProfile,
                                          pathways_forbid(kNoStairsElevator),
                                          l("P1"), l("P2"))
                   .has_value());
}

TEST(routing, pathways_restrictions) {
  auto const tt = loader::load(loader::mem_dir::read(kPathwaysFiles),
                               {2019_y / May / 1, 2019_y / May / 2});

  auto const width = fr::to_width(1.5);
  auto const slope = fr::to_slope(6.0);
  for (auto const dir : {direction::kForward, direction::kBackward}) {
    SCOPED_TRACE(dir == direction::kForward ? "fwd" : "bwd");

    // (a), 2 min -> T2
    EXPECT_EQ("10:20", pathways_earliest_arrival(tt, {}, dir));

    // (b), 4 min -> T3
    EXPECT_EQ("10:22",
              pathways_earliest_arrival(tt, pathways_forbid(kNoStairs), dir));

    // (b) with factor 2: only the walking part doubles: 2 * 2 + 2 = 6 -> T4
    EXPECT_EQ("10:24", pathways_earliest_arrival(
                           tt,
                           {.default_ = false,
                            .factor_ = 2.0F,
                            .restrictions_ = {.forbidden_ = kNoStairs}},
                           dir));

    // no route -> direct trip
    EXPECT_EQ("10:40", pathways_earliest_arrival(
                           tt, pathways_forbid(kNoStairsElevator), dir));

    // (b) too steep
    EXPECT_EQ(
        "10:40",
        pathways_earliest_arrival(
            tt,
            {.restrictions_ = {.forbidden_ = kNoStairs, .max_slope_ = slope}},
            dir));

    // (a) passes in tolerant mode (stairs width unknown), (b) too narrow
    EXPECT_EQ("10:20", pathways_earliest_arrival(
                           tt, {.restrictions_ = {.min_width_ = width}}, dir));

    // strict: (a) fails because of the unknown width
    EXPECT_EQ("10:40",
              pathways_earliest_arrival(
                  tt, {.restrictions_ = {.min_width_ = width, .strict_ = true}},
                  dir));

    // stairs penalty 3 min: (a) 5 min > (b) 4 min -> T3
    auto stairs_penalty = routing::transfer_time_settings{};
    stairs_penalty.restrictions_.penalty_[0] = 3U;
    EXPECT_EQ("10:22", pathways_earliest_arrival(tt, stairs_penalty, dir));

    // (b) with elevator penalty 5 min: 9 min, no departure left -> direct
    auto elevator_penalty = pathways_forbid(kNoStairs);
    elevator_penalty.restrictions_.penalty_[2] = 5U;
    EXPECT_EQ("10:40", pathways_earliest_arrival(tt, elevator_penalty, dir));
  }
}

// An out-of-service elevator (rt) blocks route (b).
TEST(routing, pathways_blocked_elevator) {
  auto const tt = loader::load(loader::mem_dir::read(kPathwaysFiles),
                               {2019_y / May / 1, 2019_y / May / 2});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };
  auto rtt = rt::create_rt_timetable(tt, 2019_y / May / 1);
  // blocked bit = record index
  auto const block = [&](auto const& records, bitvec& blocked,
                         location_idx_t const from) {
    blocked.resize(static_cast<bitvec::size_type>(records.data_.size()));
    auto const rs = records[from];
    for (auto i = 0U; i != rs.size(); ++i) {
      if ((rs[i].flags_ & fr::kElevator) != 0U) {
        blocked.set(static_cast<bitvec::size_type>(
                        records.bucket_starts_[to_idx(from)] + i),
                    true);
      }
    }
  };
  auto const& loc = tt.locations_;
  block(loc.footpath_restrictions_out_[kDefaultProfile],
        rtt.blocked_restricted_out_[kDefaultProfile], l("P1"));
  block(loc.footpath_restrictions_in_[kDefaultProfile],
        rtt.blocked_restricted_in_[kDefaultProfile], l("P2"));

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    SCOPED_TRACE(dir == direction::kForward ? "fwd" : "bwd");
    EXPECT_EQ("10:20", pathways_earliest_arrival(tt, {}, dir, &rtt));
    EXPECT_EQ("10:40", pathways_earliest_arrival(tt, pathways_forbid(kNoStairs),
                                                 dir, &rtt));
  }
  EXPECT_FALSE(routing::get_footpath_restriction(tt, kDefaultProfile,
                                                 pathways_forbid(kNoStairs),
                                                 l("P1"), l("P2"), &rtt)
                   .has_value());
}

TEST(loader, footpath_restriction_aggregation) {
  auto a = fr{};
  a.flags_ = fr::kStairs;
  a.slope_ = fr::to_slope(2.0);
  a.width_ = static_cast<std::uint8_t>(fr::to_width(1.2) | fr::kUnknown);

  auto b = fr{};
  b.flags_ = fr::kElevator;
  b.slope_ = static_cast<std::uint8_t>(fr::to_slope(4.0) | fr::kUnknown);
  b.width_ = fr::to_width(0.9);
  b.fixed_ = fr::to_fixed(95U);

  auto const c = a + b;
  EXPECT_EQ(fr::kStairs | fr::kElevator, c.flags_);
  EXPECT_EQ(fr::to_slope(4.0), c.slope());
  EXPECT_TRUE(c.has_unknown_slope());
  EXPECT_EQ(fr::to_width(0.9), c.width());
  EXPECT_TRUE(c.has_unknown_width());
  EXPECT_EQ(100U, c.fixed_seconds());
  EXPECT_EQ(a, a + fr{});

  EXPECT_TRUE(is_less_restricted(a, c));
  EXPECT_FALSE(is_less_restricted(c, a));

  // known violation fails in tolerant mode, too
  auto const tolerant =
      routing::footpath_restriction_settings{.max_slope_ = fr::to_slope(3.0)};
  EXPECT_FALSE(routing::is_usable(c, tolerant));
  EXPECT_TRUE(routing::is_usable(c, routing::footpath_restriction_settings{
                                        .max_slope_ = fr::to_slope(5.0)}));
  EXPECT_FALSE(routing::is_usable(
      c, routing::footpath_restriction_settings{.max_slope_ = fr::to_slope(5.0),
                                                .strict_ = true}));

  // rounding: slope up, width down
  EXPECT_EQ(30U, fr::to_slope(6.0));
  EXPECT_EQ(31U, fr::to_slope(6.05));
  EXPECT_EQ(18U, fr::to_width(0.9));
  EXPECT_EQ(17U, fr::to_width(0.89));
}

// transfers.txt rules are authoritative: every route P1 -> P2 takes 5 min.
TEST(routing, pathways_transfer_rule) {
  auto const files = std::string{kPathwaysFiles} + R"(
# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
P1,P2,2,300
)";
  auto const tt = loader::load(loader::mem_dir::read(files),
                               {2019_y / May / 1, 2019_y / May / 2});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };

  auto const fps = pathways_restricted(tt, true, l("P1"));
  ASSERT_EQ(2U, fps.size());
  EXPECT_EQ(duration_t{5}, fps[0].duration());
  EXPECT_EQ(duration_t{5}, fps[1].duration());

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    SCOPED_TRACE(dir == direction::kForward ? "fwd" : "bwd");
    EXPECT_EQ("10:24", pathways_earliest_arrival(tt, {}, dir));
    EXPECT_EQ("10:24",
              pathways_earliest_arrival(tt, pathways_forbid(kNoStairs), dir));
    EXPECT_EQ("10:40", pathways_earliest_arrival(
                           tt, pathways_forbid(kNoStairsElevator), dir));
  }
}

TEST(loader, pathways_station_paths) {
  auto const dir =
      std::filesystem::temp_directory_path() / "nigiri_pathways_test_paths";
  std::filesystem::remove_all(dir);
  auto paths = station_paths{dir, cista::mmap::protection::WRITE};
  auto const tt = loader::load(loader::mem_dir::read(kPathwaysFiles),
                               {2019_y / May / 1, 2019_y / May / 2},
                               {.station_paths_ = &paths});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };

  EXPECT_EQ(4U, paths.paths_.size());
  ASSERT_EQ(1U, paths.entrances_.size());
  auto const e = station_entrance_idx_t{0U};
  EXPECT_EQ("E", paths.get_string(paths.entrances_[e].name_));

  // Entrance E: P1 by a 20 m walkway, P2 via P1 by stairs or elevator.
  for (auto const is_exit : {false, true}) {
    SCOPED_TRACE(is_exit ? "exit" : "entrance");
    auto const ep = paths.entrance_paths(e, is_exit);
    ASSERT_EQ(3U, ep.size());
    auto const to_p1 = utl::find_if(
        ep, [&](station_entrance_path const& x) { return x.l_ == l("P1"); });
    ASSERT_NE(end(ep), to_p1);
    EXPECT_EQ(17U, to_p1->walk_seconds_);
    EXPECT_EQ(0U, to_p1->fixed_seconds_);
    EXPECT_EQ(0U, to_p1->restriction_.flags_);
    auto const pieces = paths.entrance_path_pieces_[paths.get_idx(*to_p1)];
    ASSERT_EQ(1U, pieces.size());
    EXPECT_EQ(is_exit ? "Ausgang" : "",
              paths.get_string(paths.pieces_[pieces[0]].signpost_));
    EXPECT_EQ(2, utl::count_if(ep, [&](station_entrance_path const& x) {
                return x.l_ == l("P2");
              }));
    EXPECT_TRUE(utl::all_of(ep, [&](station_entrance_path const& x) {
      return x.is_exit_ == is_exit;
    }));
  }
  EXPECT_TRUE(paths.is_entrance_only(l("P1")));
  EXPECT_TRUE(paths.is_entrance_only(l("P2")));
  EXPECT_FALSE(paths.is_entrance_only(l("A")));

  // No stairs: elevator + ramp + boarding area -> platform.
  auto const r = routing::get_footpath_restriction(
      tt, kDefaultProfile, pathways_forbid(kNoStairs), l("P1"), l("P2"));
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(fr::kElevator | fr::kRamp, r->flags_);
  auto const path = paths.find(l("P1"), l("P2"), *r);
  ASSERT_TRUE(path.has_value());
  auto const pieces = paths.path_pieces_[*path];
  ASSERT_EQ(3U, pieces.size());
  EXPECT_EQ(station_piece_mode::kElevator, paths.pieces_[pieces[0]].mode_);
  EXPECT_EQ(station_piece_mode::kWalkway, paths.pieces_[pieces[1]].mode_);
  EXPECT_EQ(fr::kRamp, paths.pieces_[pieces[1]].restriction_.flags_);
  EXPECT_FLOAT_EQ(-1.F, paths.nodes_[paths.pieces_[pieces[0]].to_].level_);
  EXPECT_FALSE(paths.nodes_[paths.pieces_[pieces[0]].to_].has_pos());

  // Default: stairs, signposted walkway to the boarding area.
  auto const fastest = routing::get_footpath_restriction(tt, kDefaultProfile,
                                                         {}, l("P1"), l("P2"));
  ASSERT_TRUE(fastest.has_value());
  auto const stairs = paths.find(l("P1"), l("P2"), *fastest);
  ASSERT_TRUE(stairs.has_value());
  auto const stairs_pieces = paths.path_pieces_[*stairs];
  ASSERT_EQ(3U, stairs_pieces.size());
  EXPECT_EQ(station_piece_mode::kStairs, paths.pieces_[stairs_pieces[0]].mode_);
  EXPECT_EQ("Gleis 2",
            paths.get_string(paths.pieces_[stairs_pieces[1]].signpost_));

  std::filesystem::remove_all(dir);
}

// stop_access=1: P1 stays reachable from the street directly.
TEST(loader, pathways_stop_access) {
  auto files = std::string{kPathwaysFiles};
  auto const from = files.find("# stops.txt");
  auto const to = files.find("# levels.txt");
  ASSERT_NE(std::string::npos, from);
  ASSERT_NE(std::string::npos, to);
  files.replace(from, to - from, R"(# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,level_id,stop_access
A,A,48.1,9.1,0,,,
C,C,48.2,9.2,0,,,
S,S,48.0,9.0,1,,,
P1,P1,48.0,9.0,0,S,L0,1
P2,P2,48.001,9.0,0,S,L0,0
BA2,BA2,48.001,9.0,4,P2,L0,
N1,N1,,,3,S,L-1,
N2,N2,,,3,S,L-1,
E,E,48.0,9.001,2,S,L0,

)");

  auto const dir =
      std::filesystem::temp_directory_path() / "nigiri_pathways_test_access";
  std::filesystem::remove_all(dir);
  auto paths = station_paths{dir, cista::mmap::protection::WRITE};
  auto const tt = loader::load(loader::mem_dir::read(files),
                               {2019_y / May / 1, 2019_y / May / 2},
                               {.station_paths_ = &paths});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };
  EXPECT_FALSE(paths.is_entrance_only(l("P1")));
  EXPECT_TRUE(paths.is_entrance_only(l("P2")));
  std::filesystem::remove_all(dir);
}

// The user script drops the pathways of S: walks as without pathways.txt.
TEST(loader, pathways_lua_drop_station) {
  auto const d = loader::mem_dir::read(kPathwaysFiles);
  auto const tt = loader::load(
      {loader::dir_source{.dir_ = d, .loader_config_ = {.user_script_ = R"(
function process_pathways(station)
  return station:get_id() ~= "S"
end
)"}}},
      {2019_y / May / 1, 2019_y / May / 2});
  auto const l = [&](std::string_view const id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}});
  };

  EXPECT_EQ(
      0U,
      tt.locations_.footpath_restrictions_out_[kDefaultProfile].data_.size());
  auto const base = tt.locations_.footpaths_out_[kDefaultProfile][l("P1")];
  EXPECT_TRUE(utl::any_of(
      base, [&](footpath const fp) { return fp.target() == l("P2"); }));
  EXPECT_EQ("10:20", pathways_earliest_arrival(tt, pathways_forbid(kNoStairs),
                                               direction::kForward));
}

// Oracle: for random station graphs and random settings, the fastest usable
// stored route equals a constrained Dijkstra on the graph itself (the Pareto
// set misses no setting).
std::optional<std::uint32_t> station_graph_oracle(
    loader::station_graph const& g,
    routing::footpath_restriction_settings const& s,
    std::uint32_t const factor,
    loader::station_graph::node_idx_t const from,
    loader::station_graph::node_idx_t const to) {
  constexpr auto const kInf = std::numeric_limits<std::uint32_t>::max();
  auto dist = std::vector<std::uint32_t>(g.nodes_.size(), kInf);
  auto pq =
      std::priority_queue<std::pair<std::uint32_t, std::uint32_t>,
                          std::vector<std::pair<std::uint32_t, std::uint32_t>>,
                          std::greater<>>{};
  dist[from] = 0U;
  pq.emplace(0U, from);
  while (!pq.empty()) {
    auto const [d, n] = pq.top();
    pq.pop();
    if (d != dist[n]) {
      continue;
    }
    for (auto const& e : g.edges_) {
      if (e.from_ != n || e.to_ == from ||
          !routing::is_usable(e.restriction_, s)) {
        continue;
      }
      auto const next = d + e.walk_seconds_ * factor + e.fixed_seconds_;
      if (next < dist[e.to_]) {
        dist[e.to_] = next;
        pq.emplace(next, e.to_);
      }
    }
  }
  return dist[to] == kInf ? std::nullopt : std::optional{dist[to]};
}

TEST(loader, station_routes_oracle) {
  auto rng = std::mt19937{42U};
  auto const rand = [&](std::uint32_t const n) {
    return std::uniform_int_distribution<std::uint32_t>{0U, n - 1U}(rng);
  };
  auto const random_restriction = [&]() {
    auto r = fr{};
    for (auto const f : {fr::kStairs, fr::kEscalator, fr::kElevator, fr::kRamp,
                         fr::kStep, fr::kBanWheelchair}) {
      if (rand(4U) == 0U) {
        r.flags_ |= f;
      }
    }
    r.slope_ =
        rand(3U) == 0U ? fr::kUnknown : static_cast<std::uint8_t>(rand(60U));
    r.width_ = rand(3U) == 0U
                   ? static_cast<std::uint8_t>(fr::kNoWidth | fr::kUnknown)
                   : static_cast<std::uint8_t>(10U + rand(50U));
    return r;
  };

  auto n_checked = 0U;
  for (auto i = 0U; i != 200U; ++i) {
    auto g = loader::station_graph{};
    auto const n_nodes = 3U + rand(8U);
    for (auto n = 0U; n != n_nodes; ++n) {
      // every third node: inner node (no stop)
      auto node = loader::station_graph::node{};
      node.l_ = n % 3U == 2U ? location_idx_t::invalid() : location_idx_t{n};
      g.add_node(std::move(node));
    }
    for (auto e = 0U; e != 3U * n_nodes; ++e) {
      auto const from = rand(n_nodes);
      auto const to = rand(n_nodes);
      if (from == to) {
        continue;
      }
      g.add_edge({.from_ = from,
                  .to_ = to,
                  .walk_seconds_ = rand(120U),
                  .fixed_seconds_ = rand(2U) == 0U ? 0U : 30U + rand(60U),
                  .restriction_ = random_restriction()});
    }
    auto const routes = loader::get_station_routes(g);

    for (auto k = 0U; k != 20U; ++k) {
      auto s = routing::footpath_restriction_settings{};
      for (auto const f : {fr::kStairs, fr::kEscalator, fr::kElevator,
                           fr::kRamp, fr::kStep, fr::kBanWheelchair}) {
        if (rand(3U) == 0U) {
          s.forbidden_ |= f;
        }
      }
      if (rand(2U) == 0U) {
        s.max_slope_ = static_cast<std::uint8_t>(rand(60U));
      }
      if (rand(2U) == 0U) {
        s.min_width_ = static_cast<std::uint8_t>(10U + rand(50U));
      }
      s.strict_ = rand(2U) == 0U;
      auto const factor = 1U + rand(3U);

      for (auto a = 0U; a != n_nodes; ++a) {
        for (auto b = 0U; b != n_nodes; ++b) {
          if (a == b || a % 3U == 2U || b % 3U == 2U) {
            continue;
          }
          auto actual = std::optional<std::uint32_t>{};
          for (auto const& r : routes) {
            if (r.from_ == location_idx_t{a} && r.to_ == location_idx_t{b} &&
                routing::is_usable(r.restriction_, s)) {
              auto const c = r.walk_seconds_ * factor + r.fixed_seconds_;
              actual = std::min(actual.value_or(c), c);
            }
          }
          EXPECT_EQ(station_graph_oracle(g, s, factor, a, b), actual)
              << "graph " << i << " settings " << k << ": " << a << " -> " << b;
          ++n_checked;
        }
      }
    }
  }
  EXPECT_GT(n_checked, 10000U);
}
