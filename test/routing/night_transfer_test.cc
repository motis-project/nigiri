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

// Europe/Berlin, UTC+2 on this day: night is local time.
//   N0 -> N9  23:00 -> 03:00 (alights in the night)
//             23:00 -> 06:30 (04:30 UTC: neither boards nor alights in the
//             local night)
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

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,MTA,R,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,D,FAST,,
R,D,SLOW,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
FAST,23:00:00,23:00:00,N0,0,0,0
FAST,27:00:00,27:00:00,N9,1,0,0
SLOW,23:00:00,23:00:00,N0,0,0,0
SLOW,30:30:00,30:30:00,N9,1,0,0
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

query make(direction const dir) {
  auto const l = [](char const* id) {
    return night_tt().find({id, source_idx_t{0U}}).value();
  };
  auto q = query{};
  q.start_ = {{l("N0"), 0min, 0U}};
  q.destination_ = {{l("N9"), 0min, 0U}};
  if (dir == direction::kBackward) {
    q.flip_dir();
  }
  q.start_time_ = dir == direction::kForward ? interval{at(22), at(24)}
                                             : interval{at(24 + 2), at(24 + 7)};
  return q;
}

}  // namespace

TEST(night_transfer, keeps_the_fastest_and_the_best_without_night) {
  auto const expected =
      std::vector<std::string>{"23:00 27:00 n1", "23:00 30:30 n0"};
  for (auto const dir : {direction::kForward, direction::kBackward}) {
    auto ss = search_state{};
    auto ms = mcraptor_night_transfer_state{};
    EXPECT_EQ(expected,
              rows(*raptor_search(night_tt(), nullptr, ss, ms, make(dir), dir)
                        .journeys_))
        << (dir == direction::kForward ? "forward" : "backward");

    auto as = raptor_state{};
    EXPECT_EQ(expected,
              rows(*bmrap_profile_search<arr_night_transfer_criteria>(
                        night_tt(), nullptr, ss, as, ms, make(dir), dir)
                        .journeys_))
        << "bmrapp " << (dir == direction::kForward ? "forward" : "backward");

    // composed with the criteria the UI sets by default
    using all_t = arr_non_transit_mode_switches_night_transfer_criteria;
    auto all_ms = basic_mcraptor_state<all_t>{};
    EXPECT_EQ(expected,
              rows(*bmrap_profile_search<all_t>(night_tt(), nullptr, ss, as,
                                                all_ms, make(dir), dir)
                        .journeys_))
        << "composed " << (dir == direction::kForward ? "forward" : "backward");
  }
}

TEST(night_transfer, is_night) {
  EXPECT_TRUE(is_night(0));
  EXPECT_TRUE(is_night(359));
  EXPECT_FALSE(is_night(360));
  EXPECT_FALSE(is_night(1439));
}
