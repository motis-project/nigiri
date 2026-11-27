#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/special_stations.h"

#include "../loader/hrd/hrd_timetable.h"

#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using nigiri::test::raptor_search;
using namespace std::chrono_literals;

constexpr auto const fwd_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]

[2020-03-30 05:30, 2020-03-30 07:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:45]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]

)";

TEST(routing, raptor_forward) {

  auto const tt = loader::load(files_abc(), full_period());

  auto const results = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});

  EXPECT_EQ(std::string_view{fwd_journeys}, to_string(tt, results));
}

constexpr auto const bwd_journeys = R"(
[2020-03-30 03:00, 2020-03-30 05:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 03:00]
       TO: (C, 0000003) [2020-03-30 05:15]
leg 0: (A, 0000001) [2020-03-30 03:00] -> (B, 0000002) [2020-03-30 04:00]
   0: 0000001 A...............................................                               d: 30.03 03:00 [30.03 05:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/180/0000002/240/, src=0}]
   1: 0000002 B............................................... a: 30.03 04:00 [30.03 06:00]
leg 1: (B, 0000002) [2020-03-30 04:00] -> (B, 0000002) [2020-03-30 04:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 04:15] -> (C, 0000003) [2020-03-30 05:15]
   0: 0000002 B...............................................                               d: 30.03 04:15 [30.03 06:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/255/0000003/315/, src=0}]
   1: 0000003 C............................................... a: 30.03 05:15 [30.03 07:15]

[2020-03-30 03:30, 2020-03-30 05:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 03:30]
       TO: (C, 0000003) [2020-03-30 05:45]
leg 0: (A, 0000001) [2020-03-30 03:30] -> (B, 0000002) [2020-03-30 04:30]
   0: 0000001 A...............................................                               d: 30.03 03:30 [30.03 05:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/210/0000002/270/, src=0}]
   1: 0000002 B............................................... a: 30.03 04:30 [30.03 06:30]
leg 1: (B, 0000002) [2020-03-30 04:30] -> (B, 0000002) [2020-03-30 04:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 04:45] -> (C, 0000003) [2020-03-30 05:45]
   0: 0000002 B...............................................                               d: 30.03 04:45 [30.03 06:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/285/0000003/345/, src=0}]
   1: 0000003 C............................................... a: 30.03 05:45 [30.03 07:45]

)";

TEST(routing, raptor_backward) {
  using namespace date;

  auto const tt = loader::load(files_abc(), full_period());

  auto const results = raptor_search(
      tt, nullptr, "0000003", "0000001",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
      direction::kBackward);

  EXPECT_EQ(2U, results.size());

  EXPECT_EQ(std::string_view{bwd_journeys}, to_string(tt, results));
}

namespace {

// A -> B -> C -> D -> E -> F: four transfers
// X -> F and A -> Y: direct
mem_dir state_reuse_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,1.0,2.0,,
C,C,,2.0,3.0,,
D,D,,3.0,4.0,,
E,E,,4.0,5.0,,
F,F,,5.0,6.0,,
X,X,,6.0,7.0,,
Y,Y,,7.0,8.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,2
R2,DB,2,,,2
R3,DB,3,,,2
R4,DB,4,,,2
R5,DB,5,,,2
RXF,DB,XF,,,2
RAY,DB,AY,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,,
R2,S,T2,,
R3,S,T3,,
R4,S,T4,,
R5,S,T5,,
RXF,S,XF,,
RAY,S,AY,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,08:00:00,08:00:00,A,1,0,0
T1,08:30:00,08:30:00,B,2,0,0
T2,08:40:00,08:40:00,B,1,0,0
T2,09:10:00,09:10:00,C,2,0,0
T3,09:20:00,09:20:00,C,1,0,0
T3,09:50:00,09:50:00,D,2,0,0
T4,10:00:00,10:00:00,D,1,0,0
T4,10:30:00,10:30:00,E,2,0,0
T5,10:40:00,10:40:00,E,1,0,0
T5,11:10:00,11:10:00,F,2,0,0
XF,13:00:00,13:00:00,X,1,0,0
XF,14:00:00,14:00:00,F,2,0,0
AY,05:00:00,05:00:00,A,1,0,0
AY,06:00:00,06:00:00,Y,2,0,0
)");
}

}  // namespace

// Pong initializes the rounds of a raptor_state on first use (lazy reset). The
// rounds that only the first (deep) search reached must not leak into the
// second (shallow) search on the same state: both reach the same destination,
// so its stale arrival looks like a journey with more transfers.
TEST(routing, raptor_state_reuse) {
  auto const tt = loader::load(state_reuse_files(),
                               {2024_y / June / 18, 2024_y / June / 20});

  auto const day = sys_days{2024_y / June / 19};
  auto const make_query = [&](std::string_view from, std::string_view to,
                              unixtime_t const from_time) {
    return routing::query{
        .start_time_ =
            interval<unixtime_t>{from_time, from_time + std::chrono::hours{1}},
        .start_match_mode_ = routing::location_match_mode::kEquivalent,
        .dest_match_mode_ = routing::location_match_mode::kEquivalent,
        .use_start_footpaths_ = false,
        .start_ = {{tt.find(location_id{from, source_idx_t{0}}).value(),
                    0_minutes, 0U}},
        .destination_ = {{tt.find(location_id{to, source_idx_t{0}}).value(),
                          0_minutes, 0U}}};
  };

  using result_t = std::vector<std::tuple<unixtime_t, unixtime_t, unsigned>>;
  auto const run = [&](bool const pong, routing::query q, direction const dir,
                       routing::search_state& ss, routing::raptor_state& rs) {
    auto const r =
        pong ? routing::pong_search(tt, nullptr, ss, rs, std::move(q), dir)
             : routing::raptor_search(tt, nullptr, ss, rs, std::move(q), dir);
    auto journeys = result_t{};
    for (auto const& j : *r.journeys_) {
      journeys.emplace_back(j.departure_time(), j.arrival_time(), j.transfers_);
    }
    return journeys;
  };

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    auto const is_fwd = dir == direction::kForward;
    auto const deep = [&]() {
      return is_fwd ? make_query("A", "F", day + 7h + 30min)
                    : make_query("F", "A", day + 10h + 30min);
    };
    auto const shallow = [&]() {
      return is_fwd ? make_query("X", "F", day + 12h + 30min)
                    : make_query("Y", "A", day + 5h + 30min);
    };

    for (auto const first_pong : {false, true}) {
      for (auto const second_pong : {false, true}) {
        auto ss = routing::search_state{};
        auto rs = routing::raptor_state{};
        auto const deep_result = run(first_pong, deep(), dir, ss, rs);
        ASSERT_EQ(1U, deep_result.size());
        EXPECT_EQ(4U, std::get<2>(deep_result.front()));

        auto const reused = run(second_pong, shallow(), dir, ss, rs);

        auto fresh_ss = routing::search_state{};
        auto fresh_rs = routing::raptor_state{};
        auto const fresh = run(second_pong, shallow(), dir, fresh_ss, fresh_rs);

        ASSERT_EQ(1U, fresh.size());
        EXPECT_EQ(0U, std::get<2>(fresh.front()));
        EXPECT_EQ(fresh, reused)
            << "dir=" << (is_fwd ? "fwd" : "bwd")
            << ", first_pong=" << first_pong << ", second_pong=" << second_pong;
      }
    }
  }
}
