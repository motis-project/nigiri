#include <vector>

#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// Source 0: A -> B, 10:00 - 10:30
constexpr auto const kSlowFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,0,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,T0,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T0,10:00:00,10:00:00,A,0
T0,10:30:00,10:30:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

// Source 1: A -> B, 10:00 - 10:10
constexpr auto const kFastFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
SH,Shuttle,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,SH,1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

}  // namespace

TEST(routing, blocked_routes) {
  auto const slow = loader::mem_dir::read(kSlowFeed);
  auto const fast = loader::mem_dir::read(kFastFeed);
  auto const tt =
      loader::load({{slow}, {fast}}, {2019_y / May / 1, 2019_y / May / 2});

  auto const loc = [&](std::string_view id, source_idx_t const src) {
    return tt.locations_.location_id_to_idx_.at({id, src});
  };

  auto const search = [&](std::vector<source_idx_t> const& blocked) {
    auto srcs = bitvec_map<source_idx_t>{};
    srcs.resize(tt.n_sources());
    for (auto const src : blocked) {
      srcs.set(src);
    }
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ =
                interval{unixtime_t{sys_days{2019_y / May / 1} + 8h},
                         unixtime_t{sys_days{2019_y / May / 1} + 9h}},
            .start_ = {{loc("A", source_idx_t{0}), 0min, 0U},
                       {loc("A", source_idx_t{1}), 0min, 0U}},
            .destination_ = {{loc("B", source_idx_t{0}), 0min, 0U},
                             {loc("B", source_idx_t{1}), 0min, 0U}},
            .blocked_ = routing::make_blocked_routes(tt, nullptr, srcs)});
  };

  auto const arrival = [](pareto_set<routing::journey> const& results) {
    EXPECT_EQ(1U, results.size());
    return results.size() == 1U ? results.begin()->arrival_time()
                                : unixtime_t{};
  };

  auto const at = [](std::chrono::minutes const m) {
    return unixtime_t{sys_days{2019_y / May / 1} + 8h + m};
  };

  EXPECT_EQ(at(10min), arrival(search({})));
  EXPECT_EQ(at(30min), arrival(search({source_idx_t{1}})));
  EXPECT_EQ(at(10min), arrival(search({source_idx_t{0}})));
  EXPECT_TRUE(search({source_idx_t{0}, source_idx_t{1}}).empty());
}
