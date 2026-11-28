#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/raptor/bmraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace std::chrono_literals;

namespace {

// Europe/Berlin, UTC+2 on this day: night is local time (night_offset).
//   N0 -> N9  23:00 -> 03:00 (FAST: alights in the night)
//             23:00 -> 06:30 (SLOW: sleeps through it, a route of its own)
//   N0 -> M   22:00 -> 22:30, M -> N9 06:00 -> 06:20 (EVE, MORN: neither
//             rides in the night, but waits at M through it)
// Night-free on different days, at B: 21:30 (DAY) and 06:40 (SLEEPB):
//   A -> B    21:00 -> 21:30 (DAY), 21:00 -> 06:40 (SLEEPB, sleeps)
//   B -> C    07:00 -> 07:30 (MORNC: waiting for it from 21:30 is awake)
//   B -> D    21:45 -> 00:30 (LATE: alights in the night), 07:00 -> 08:00
//             (MORND, LATE's route: the earlier trip, but boarded a day
//             before)
// The first trip's wait counts too, so a later start can be better:
//   S1 -> T1  03:00 -> 03:30 (NIGHTBUS), 07:00 -> 07:30 (MORNBUS)
//   S2 -> T2  22:00 -> 22:30 (EVEBUS), 23:30 -> 00:30 (LATEBUS)
constexpr auto const kNightGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240619,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
N0,N0,,50.0,8.0,,,
N9,N9,,51.0,8.0,,,
M,M,,50.5,8.0,,,
A,A,,52.0,8.0,,,
B,B,,52.5,8.0,,,
C,C,,53.0,8.0,,,
D,D,,53.0,9.0,,,
S1,S1,,54.0,8.0,,,
T1,T1,,54.5,8.0,,,
S2,S2,,55.0,8.0,,,
T2,T2,,55.5,8.0,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,MTA,R,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,D,FAST,,
R,D,SLOW,,
R,D,EVE,,
R,D,MORN,,
R,D,DAY,,
R,D,SLEEPB,,
R,D,MORNC,,
R,D,LATE,,
R,D,MORND,,
R,D,NIGHTBUS,,
R,D,MORNBUS,,
R,D,EVEBUS,,
R,D,LATEBUS,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
FAST,23:00:00,23:00:00,N0,0,0,0
FAST,27:00:00,27:00:00,N9,1,0,0
SLOW,23:00:00,23:00:00,N0,0,0,0
SLOW,30:30:00,30:30:00,N9,1,0,0
EVE,22:00:00,22:00:00,N0,0,0,0
EVE,22:30:00,22:30:00,M,1,0,0
MORN,30:00:00,30:00:00,M,0,0,0
MORN,30:20:00,30:20:00,N9,1,0,0
DAY,21:00:00,21:00:00,A,0,0,0
DAY,21:30:00,21:30:00,B,1,0,0
SLEEPB,21:00:00,21:00:00,A,0,0,0
SLEEPB,30:40:00,30:40:00,B,1,0,0
MORNC,31:00:00,31:00:00,B,0,0,0
MORNC,31:30:00,31:30:00,C,1,0,0
LATE,21:45:00,21:45:00,B,0,0,0
LATE,24:30:00,24:30:00,D,1,0,0
MORND,31:00:00,31:00:00,B,0,0,0
MORND,32:00:00,32:00:00,D,1,0,0
NIGHTBUS,27:00:00,27:00:00,S1,0,0,0
NIGHTBUS,27:30:00,27:30:00,T1,1,0,0
MORNBUS,31:00:00,31:00:00,S1,0,0,0
MORNBUS,31:30:00,31:30:00,T1,1,0,0
EVEBUS,22:00:00,22:00:00,S2,0,0,0
EVEBUS,22:30:00,22:30:00,T2,1,0,0
LATEBUS,23:30:00,23:30:00,S2,0,0,0
LATEBUS,24:30:00,24:30:00,T2,1,0,0
)";

timetable const& night_tt() {
  static auto const tt = loader::load(loader::mem_dir::read(kNightGTFS),
                                      {2024_y / June / 18, 2024_y / June / 21});
  return tt;
}

constexpr auto const kUtcOffset = 2h;

// local time on 2024-06-19
unixtime_t at(int const hour) {
  return unixtime_t{sys_days{2024_y / June / 19}} + std::chrono::hours{hour} -
         kUtcOffset;
}

// GTFS-style, hours past 24 for the next day
std::string hhmm(unixtime_t const t) {
  auto const m = (t - at(0)).count();
  return fmt::format("{:02}:{:02}", m / 60, m % 60);
}

std::vector<std::string> rows(pareto_set<journey> const& js) {
  auto v = std::vector<std::string>{};
  for (auto const& j : js) {
    v.push_back(fmt::format("{} {} n{}", hhmm(j.departure_time()),
                            hhmm(j.arrival_time()),
                            j.criteria_night_transfer_ ? 1 : 0));
  }
  std::sort(begin(v), end(v));
  return v;
}

location_idx_t stop_of(char const* id) {
  return night_tt().find({id, source_idx_t{0U}}).value();
}

query make(direction const dir) {
  auto q = query{};
  q.start_ = {{stop_of("N0"), 0min, 0U}};
  q.destination_ = {{stop_of("N9"), 0min, 0U}};
  if (dir == direction::kBackward) {
    q.flip_dir();
  }
  q.start_time_ = dir == direction::kForward ? interval{at(22), at(24)}
                                             : interval{at(24 + 2), at(24 + 7)};
  // paging on (BMRAPP's anchors come from its scan, as in motis)
  q.extend_interval_later_ = dir == direction::kForward;
  return q;
}

}  // namespace

TEST(night_transfer, keeps_the_fastest_and_the_best_without_night) {
  auto const both =
      std::vector<std::string>{"23:00 27:00 n1", "23:00 30:30 n0"};
  for (auto const dir : {direction::kForward, direction::kBackward}) {
    auto const tag = dir == direction::kForward ? "forward" : "backward";
    auto ss = search_state{};
    auto ms = mcraptor_night_transfer_state{};
    EXPECT_EQ(both,
              rows(*raptor_search(night_tt(), nullptr, ss, ms, make(dir), dir)
                        .journeys_))
        << tag;

    // BMRAPP, restricted: forward, 06:30 is past the anchor's deadline
    // (03:00 + 2 h); backward, it anchors itself
    auto const bmrapp = [&]<typename Criteria>() {
      auto as = raptor_state{};
      auto bms = basic_mcraptor_state<Criteria>{};
      return rows(*bmrap_profile_search<Criteria>(night_tt(), nullptr, ss, as,
                                                  bms, make(dir), dir)
                       .journeys_);
    };
    auto const restricted = dir == direction::kForward
                                ? std::vector<std::string>{"23:00 27:00 n1"}
                                : both;
    EXPECT_EQ(restricted,
              bmrapp.template operator()<arr_night_transfer_criteria>())
        << tag;
    // composed with the criteria the UI sets by default
    EXPECT_EQ(
        restricted,
        bmrapp.template
        operator()<arr_non_transit_mode_switches_night_transfer_criteria>())
        << tag << " composed";
  }
}

TEST(night_transfer, sleepers_ride_alone) {
  // FAST and SLOW share their stops, as DAY and SLEEPB, LATE and MORND,
  // NIGHTBUS and MORNBUS, EVEBUS and LATEBUS do
  EXPECT_EQ(10U, night_tt().n_routes());
}

TEST(night_transfer, night_free_on_another_day) {
  auto const search = [](char const* to) {
    auto q = query{};
    q.start_ = {{stop_of("A"), 0min, 0U}};
    q.destination_ = {{stop_of(to), 0min, 0U}};
    q.start_time_ = interval{at(20), at(22)};
    auto ss = search_state{};
    auto ms = mcraptor_night_transfer_state{};
    return rows(*raptor_search(night_tt(), nullptr, ss, ms, std::move(q),
                               direction::kForward)
                     .journeys_);
  };
  // on B: 21:30 does not beat 06:40, nor (on its route) LATE MORND
  EXPECT_EQ(std::vector<std::string>{"21:00 31:30 n0"}, search("C"));
  EXPECT_EQ((std::vector<std::string>{"21:00 24:30 n1", "21:00 32:00 n0"}),
            search("D"));
}

// The wait for the first trip counts like any other: from a fixed 23:00 the
// 07:00 bus waits through the night. A profile search starts at every
// departure, BMRAPP also after every night: both find it without.
TEST(night_transfer, first_trip_after_the_night) {
  // the rides only (a start time's journeys all start then)
  auto const rides = [](pareto_set<journey> const& js) {
    auto v = std::vector<std::string>{};
    for (auto const& j : js) {
      v.push_back(fmt::format("{} {} n{}", hhmm(j.legs_.front().dep_time_),
                              hhmm(j.legs_.back().arr_time_),
                              j.criteria_night_transfer_ ? 1 : 0));
    }
    std::sort(begin(v), end(v));
    return v;
  };
  auto const make_query = [](char const* from, char const* to,
                             direction const dir, start_time_t const t) {
    auto q = query{};
    q.start_ = {{stop_of(from), 0min, 0U}};
    q.destination_ = {{stop_of(to), 0min, 0U}};
    if (dir == direction::kBackward) {
      q.flip_dir();
    }
    q.start_time_ = t;
    return q;
  };
  auto const search = [&](char const* from, char const* to, direction const dir,
                          start_time_t const t) {
    auto ss = search_state{};
    auto ms = mcraptor_night_transfer_state{};
    return rides(*raptor_search(night_tt(), nullptr, ss, ms,
                                make_query(from, to, dir, t), dir)
                      .journeys_);
  };
  // one minute, extended towards the search direction until both are found
  auto const bmrapp = [&](char const* from, char const* to, direction const dir,
                          unixtime_t const t) {
    auto q = make_query(from, to, dir,
                        dir == direction::kForward ? interval{t, t + 1min}
                                                   : interval{t - 1min, t});
    q.extend_interval_later_ = dir == direction::kForward;
    q.extend_interval_earlier_ = dir == direction::kBackward;
    q.min_connection_count_ = 2U;
    auto ss = search_state{};
    auto as = raptor_state{};
    auto ms = basic_mcraptor_state<arr_night_transfer_criteria>{};
    return rides(*bmrap_profile_search<arr_night_transfer_criteria>(
                      night_tt(), nullptr, ss, as, ms, std::move(q), dir)
                      .journeys_);
  };

  auto const fwd = std::vector<std::string>{"27:00 27:30 n1", "31:00 31:30 n0"};
  EXPECT_EQ(std::vector<std::string>{"27:00 27:30 n1"},
            search("S1", "T1", direction::kForward, at(23)));
  EXPECT_EQ(fwd, search("S1", "T1", direction::kForward,
                        interval{at(23), at(24 + 7) + 1min}));
  EXPECT_EQ(fwd, bmrapp("S1", "T1", direction::kForward, at(23)));

  // arriving by 10:00, the 22:30 arrival waits through the night
  auto const bwd = std::vector<std::string>{"22:00 22:30 n0", "23:30 24:30 n1"};
  EXPECT_EQ(std::vector<std::string>{"23:30 24:30 n1"},
            search("S2", "T2", direction::kBackward, at(24 + 10)));
  EXPECT_EQ(bwd, search("S2", "T2", direction::kBackward,
                        interval{at(22) + 30min, at(24 + 10)}));
  EXPECT_EQ(bwd, bmrapp("S2", "T2", direction::kBackward, at(24 + 10)));
}

TEST(night_transfer, night_day_by_direction) {
  // the night belongs to the day before forward, to the day after backward
  EXPECT_EQ(night_day(1380, 0, true), night_day(1440 + 330, 0, true));
  EXPECT_NE(night_day(1380, 0, false), night_day(1440 + 330, 0, false));
  EXPECT_EQ(night_day(1440 + 330, 0, false), night_day(1440 + 400, 0, false));
  EXPECT_NE(night_day(1440 + 330, 0, true), night_day(1440 + 400, 0, true));
}

TEST(night_transfer, awake_in_the_night) {
  auto const night = [](auto&& update) {
    auto f = mc_fields{};
    update(f, mc_context{});  // UTC
    return f.night_ == 1U;
  };
  auto const ride = [&](clasz const c, delta_t const board,
                        delta_t const alight) {
    return night([&](mc_fields& f, mc_context const& ctx) {
      night_transfer_dim::after_ride_times(f, c, board, alight, ctx);
    });
  };
  auto const wait = [&](delta_t const from, delta_t const to) {
    return night([&](mc_fields& f, mc_context const& ctx) {
      night_transfer_dim::after_wait(f, from, to, ctx);
    });
  };
  // 23:00 -> 06:30: a sleeper sleeps, a bus keeps one awake
  EXPECT_FALSE(ride(clasz::kRegional, 1380, 1830));
  EXPECT_TRUE(ride(clasz::kBus, 1380, 1830));
  // into the night: either
  EXPECT_TRUE(ride(clasz::kRegional, 1380, 1500));
  EXPECT_TRUE(ride(clasz::kBus, 1380, 1440));
  EXPECT_FALSE(ride(clasz::kBus, 1380, 1439));
  // waits, either direction
  EXPECT_TRUE(wait(1410, 1830));
  EXPECT_TRUE(wait(1830, 1410));
  EXPECT_FALSE(wait(400, 1380));
}

TEST(night_transfer, meets_night) {
  EXPECT_FALSE(meets_night(1380, 1439, 0));
  EXPECT_TRUE(meets_night(1380, 1440, 0));
  EXPECT_TRUE(meets_night(100, 120, 0));
  EXPECT_FALSE(meets_night(360, 1439, 0));
  EXPECT_TRUE(meets_night(-60, 0, 0));
  EXPECT_TRUE(meets_night(1300, 1330, 120));  // 23:40 -> 00:10 at UTC+2
  EXPECT_FALSE(meets_night(1300, 1310, 120));
}

TEST(night_transfer, is_night) {
  EXPECT_TRUE(is_night(0, 0));
  EXPECT_TRUE(is_night(359, 0));
  EXPECT_FALSE(is_night(360, 0));
  EXPECT_FALSE(is_night(-1, 0));  // 23:59 the day before
  EXPECT_FALSE(is_night(240, 120));  // 04:00 UTC is 06:00 at UTC+2
  EXPECT_TRUE(is_night(-60, 120));  // 23:00 UTC is 01:00
}

TEST(night_transfer, night_offset_averages_start_and_destination) {
  // both in Europe/Berlin, summer time
  EXPECT_EQ(120, night_offset(night_tt(), make(direction::kForward)));
}

namespace {

// Only a step at the end of the night finds J without the bit. From a step in
// the night J waits through it at O, and Z, later and faster but in the night,
// beats it there; nor is J an anchor (Z beats it on departure, arrival and
// transfers), so no anchor departure brings the scan back to 07:00.
//   J  O -> X 07:00 -> 07:30 (JBUS), X -> Z 22:00 -> 06:50 (SLEEPER, sleeps)
//   Z  O -> Y 08:00 -> 08:30 (ZBUS), Y -> Z 03:00 -> 04:00 (ZNIGHT)
constexpr auto const kNightEndGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240619,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
O,O,,50.0,8.0,,,
X,X,,50.5,8.0,,,
Y,Y,,50.5,9.0,,,
Z,Z,,51.0,8.0,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
JBUS,MTA,JBUS,,,3
SLEEPER,MTA,SLEEPER,,,2
ZBUS,MTA,ZBUS,,,3
ZNIGHT,MTA,ZNIGHT,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
JBUS,D,JBUS,,
SLEEPER,D,SLEEPER,,
ZBUS,D,ZBUS,,
ZNIGHT,D,ZNIGHT,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
JBUS,07:00:00,07:00:00,O,0,0,0
JBUS,07:30:00,07:30:00,X,1,0,0
SLEEPER,22:00:00,22:00:00,X,0,0,0
SLEEPER,30:50:00,30:50:00,Z,1,0,0
ZBUS,08:00:00,08:00:00,O,0,0,0
ZBUS,08:30:00,08:30:00,Y,1,0,0
ZNIGHT,27:00:00,27:00:00,Y,0,0,0
ZNIGHT,28:00:00,28:00:00,Z,1,0,0
)";

}  // namespace

TEST(night_transfer, bmrapp_steps_at_the_end_of_the_night) {
  auto const tt = loader::load(loader::mem_dir::read(kNightEndGTFS),
                               {2024_y / June / 18, 2024_y / June / 21});
  auto const stop = [&](char const* id) {
    return tt.find({id, source_idx_t{0U}}).value();
  };
  auto q = query{};
  q.start_ = {{stop("O"), 0min, 0U}};
  q.destination_ = {{stop("Z"), 0min, 0U}};
  q.start_time_ = interval{at(3), at(3) + 1min};
  q.extend_interval_later_ = true;
  q.min_connection_count_ = 2U;
  auto const both =
      std::vector<std::string>{"07:00 30:50 n0", "08:00 28:00 n1"};

  auto ss = search_state{};
  auto ms = mcraptor_night_transfer_state{};
  auto range = q;
  range.start_time_ = interval{at(3), at(8) + 1min};
  EXPECT_EQ(both, rows(*raptor_search(tt, nullptr, ss, ms, std::move(range),
                                      direction::kForward)
                            .journeys_));

  auto as = raptor_state{};
  auto bms = basic_mcraptor_state<arr_night_transfer_criteria>{};
  EXPECT_EQ(both, rows(*bmrap_profile_search<arr_night_transfer_criteria>(
                            tt, nullptr, ss, as, bms, q, direction::kForward)
                            .journeys_));
}
