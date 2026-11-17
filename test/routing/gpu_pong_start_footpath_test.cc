#include "gtest/gtest.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#if defined(NIGIRI_CUDA)
#include "nigiri/routing/gpu/raptor.h"
#endif

using namespace date;
using namespace nigiri;
using namespace std::chrono_literals;

namespace {

// X <-> P: 20 minute walks (transfers.txt). T: P 10:30 -> Q 10:40,
// T2: Q 11:00 -> P 11:10.
constexpr auto kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MTA,https://motis-project.de/,Etc/UTC

# calendar_dates.txt
service_id,date,exception_type
D,20190503,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,MTA,R,,,2
R2,MTA,R2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,D,T,,
R2,D,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T,10:30:00,10:30:00,P,1,0,0
T,10:40:00,10:40:00,Q,2,0,0
T2,11:00:00,11:00:00,Q,1,0,0
T2,11:10:00,11:10:00,P,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
X,P,2,1200
P,X,2,1200

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
X,X,,,,,,
P,P,,,,,,
Q,Q,,,,,,
)";

std::string print(timetable const& tt,
                  rt_timetable const* rtt,
                  pareto_set<routing::journey> const& js) {
  auto ss = std::stringstream{};
  for (auto const& j : js) {
    j.print(ss, tt, rtt);
  }
  return ss.str();
}

enum class expect { kNone, kWalk, kOffsetOnly };

struct side {
  char const* stop_;
  bool intermodal_;  // a 5 minute offset to the stop, else the stop itself
};

struct test_case {
  side from_, to_;  // in travel order
  bool start_fps_;  // use_start_footpaths_ at the search start
  direction dir_;  // kBackward: arriveBy, the search starts at to_
  expect expect_;
};

}  // namespace

// pong's reverse search starts at the other end of the query, where it walks
// unless intermodal (get_starts), whatever use_start_footpaths_ says:
// flip_dir() leaves the flag as it was. The GPU reconstruct has to find those
// walks too - for arriveBy queries the first one, for departure queries the
// last one, and with an intermodal start (which forces the flag off) also for
// departure queries that asked for start footpaths.
TEST(gpu_pong, walks_at_search_ends) {
  auto const tt = loader::load(loader::mem_dir::read(kFeed),
                               {2019_y / May / 2, 2019_y / May / 4});
  auto rtt = rt::create_rt_timetable(tt, sys_days{2019_y / May / 3});
  auto const loc = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
  };
  auto const day = unixtime_t{sys_days{2019_y / May / 3}};

  using enum expect;
  constexpr auto kFwd = direction::kForward;
  constexpr auto kBwd = direction::kBackward;
  auto const X = side{"X", false}, Q = side{"Q", false};
  auto const X_ = side{"X", true}, P_ = side{"P", true}, Q_ = side{"Q", true};
  auto const cases = std::vector<test_case>{
      // walk X -> P before T
      {X, Q, false, kFwd, kNone},  // no walk at the search start
      {X, Q, true, kFwd, kWalk},
      {X, Q, false, kBwd, kWalk},  // the walk is at the search end
      {X, Q, true, kBwd, kWalk},
      {X, Q_, false, kFwd, kNone},
      {X, Q_, true, kFwd, kWalk},
      {X, Q_, false, kBwd, kWalk},  // intermodal search start: flag off
      {X_, Q, false, kFwd, kNone},  // the offset is the whole first mile
      {X_, Q, false, kBwd, kNone},
      {P_, Q, false, kFwd, kOffsetOnly},
      {P_, Q, false, kBwd, kOffsetOnly},
      // walk P -> X after T2
      {Q, X, false, kFwd, kWalk},  // the walk is at the search end
      {Q, X, true, kFwd, kWalk},
      {Q, X, false, kBwd, kNone},  // no walk at the search start
      {Q, X, true, kBwd, kWalk},
      {Q_, X, false, kFwd, kWalk},  // intermodal search start: flag off
      {Q_, X, false, kBwd, kNone},
      {Q_, X, true, kBwd, kWalk},
      {Q, X_, false, kFwd, kNone},  // the offset is the whole last mile
      {Q, X_, false, kBwd, kNone},
      {Q, P_, false, kFwd, kOffsetOnly},
      {Q, P_, false, kBwd, kOffsetOnly}};

  for (auto const* const r : {static_cast<rt_timetable const*>(nullptr),
                              static_cast<rt_timetable const*>(&rtt)}) {
#if defined(NIGIRI_CUDA)
    auto const gtt = routing::gpu::gpu_timetable{tt};
    if (r != nullptr) {
      rtt.gpu_rtt_.ptr_ = routing::gpu::make_gpu_rtt(tt, rtt);
    }
#endif
    for (auto const& c : cases) {
      auto const offset = [&](side const& s) {
        return routing::offset{loc(s.stop_), s.intermodal_ ? 5min : 0min, 0U};
      };
      auto const mode = [](side const& s) {
        return s.intermodal_ ? routing::location_match_mode::kIntermodal
                             : routing::location_match_mode::kEquivalent;
      };
      auto const fwd = c.dir_ == kFwd;
      auto const& start = fwd ? c.from_ : c.to_;
      auto const& dest = fwd ? c.to_ : c.from_;
      auto q = routing::query{};
      q.start_time_ = fwd ? interval<unixtime_t>{day + 9h, day + 11h + 30min}
                          : interval<unixtime_t>{day + 10h, day + 13h};
      q.start_match_mode_ = mode(start);
      q.dest_match_mode_ = mode(dest);
      q.use_start_footpaths_ = c.start_fps_;
      q.start_ = {offset(start)};
      q.destination_ = {offset(dest)};

      auto const tag =
          fmt::format("{}{} -> {}{}, start_fps={}, {}, {}", c.from_.stop_,
                      c.from_.intermodal_ ? "*" : "", c.to_.stop_,
                      c.to_.intermodal_ ? "*" : "", c.start_fps_,
                      fwd ? "fwd" : "bwd", r == nullptr ? "static" : "rt");

      auto ss = routing::search_state{};
      auto rs = routing::raptor_state{};
      auto const pong = print(
          tt, r, *routing::pong_search(tt, r, ss, rs, q, c.dir_).journeys_);
      auto const raptor = print(
          tt, r, *routing::raptor_search(tt, r, ss, rs, q, c.dir_).journeys_);
      auto const has_walk = pong.find("FOOTPATH (duration=20)");
      switch (c.expect_) {
        case kNone: EXPECT_EQ("", pong) << tag; break;
        case kWalk: EXPECT_NE(std::string::npos, has_walk) << tag; break;
        case kOffsetOnly:
          EXPECT_NE("", pong) << tag;
          EXPECT_EQ(std::string::npos, has_walk) << tag;
          break;
      }

#if defined(NIGIRI_CUDA)
      auto gss = routing::search_state{};
      auto gs = routing::gpu::gpu_raptor_state{gtt};
      EXPECT_EQ(
          pong,
          print(tt, r,
                *routing::pong_search(tt, r, gss, gs, q, c.dir_).journeys_))
          << "gpu pong: " << tag;
      EXPECT_EQ(
          raptor,
          print(tt, r,
                *routing::raptor_search(tt, r, gss, gs, q, c.dir_).journeys_))
          << "gpu raptor: " << tag;
#endif
    }
  }
}
