#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

// 10:00 Europe/Berlin
constexpr auto const kTen = unixtime_t{sys_days{2019_y / May / 1}} + 8h;

// S walks to P (3 min) and to D (5 min). The walk to D is a start at the
// destination for the start time 10:02 (OUT departs D at 10:07). Walk-only
// starts are no journeys and must not hide BUS 10:01 -> 10:07.
constexpr auto const kWalkToDestination = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,50.0,6.0,,,
P,P,,50.1,6.0,,,
D,D,,50.2,6.0,,,
Z,Z,,51.0,7.0,,,

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RB,DTA,RB,,,3
RO,DTA,RO,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RB,X,BUS,,
RO,X,OUT,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
BUS,10:04:00,10:04:00,P,0,0,0
BUS,10:07:00,10:07:00,D,1,0,0
OUT,10:07:00,10:07:00,D,0,0,0
OUT,10:30:00,10:30:00,Z,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
S,P,2,180
P,S,2,180
S,D,2,300
D,S,2,300
)"sv;

TEST(routing, walk_to_destination_does_not_hide_journey) {
  auto const tt = loader::load(loader::mem_dir::read(kWalkToDestination),
                               {2019_y / May / 1, 2019_y / May / 2},
                               {.adjust_footpaths_ = false});
  auto const loc = [&](std::string_view const id) {
    return tt.find(location_id{id, source_idx_t{0}}).value();
  };
  auto const results = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = interval{kTen, kTen + 5min},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .use_start_footpaths_ = true,
          .start_ = {{loc("S"), 0min, 0U}},
          .destination_ = {{loc("D"), 0min, 0U}},
          .fastest_direct_factor_ = 10.0},
      direction::kForward);
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(kTen + 1min, results.begin()->start_time_);
  EXPECT_EQ(kTen + 7min, results.begin()->dest_time_);
}

// Origin = destination: walking to P and taking BUS back to S is no journey.
// PONG must not throw ("no pong found") for the round trip.
constexpr auto const kOriginIsDestination = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,50.0,6.0,,,
P,P,,50.1,6.0,,,
Z,Z,,51.0,7.0,,,

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RB,DTA,RB,,,3
RO,DTA,RO,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RB,X,BUS,,
RO,X,OUT,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
BUS,10:04:00,10:04:00,P,0,0,0
BUS,10:10:00,10:10:00,S,1,0,0
OUT,10:03:00,10:03:00,S,0,0,0
OUT,10:30:00,10:30:00,Z,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
S,P,2,180
P,S,2,180
)"sv;

TEST(routing, origin_is_destination_has_no_round_trip) {
  auto const tt = loader::load(loader::mem_dir::read(kOriginIsDestination),
                               {2019_y / May / 1, 2019_y / May / 2},
                               {.adjust_footpaths_ = false});
  auto const loc = [&](std::string_view const id) {
    return tt.find(location_id{id, source_idx_t{0}}).value();
  };
  auto const results = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = interval{kTen, kTen + 5min},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .use_start_footpaths_ = true,
          .start_ = {{loc("S"), 0min, 0U}},
          .destination_ = {{loc("S"), 0min, 0U}}},
      direction::kForward);
  EXPECT_EQ(0U, results.size());
}

// X is an origin (5 min offset) and the destination. BUS S 10:01 -> X 10:03
// beats the 5 min offset and is a journey (RAPTOR and PONG, no "no pong
// found").
constexpr auto const kOriginWithOffsetIsDestination = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,50.0,6.0,,,
PX,PX,,50.1,6.0,,1,
X,X,,50.1,6.0,,,PX
Z,Z,,51.0,7.0,,,

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RB,DTA,RB,,,3
RO,DTA,RO,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RB,X,BUS,,
RO,X,OUT,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
BUS,10:01:00,10:01:00,S,0,0,0
BUS,10:03:00,10:03:00,X,1,0,0
OUT,10:10:00,10:10:00,X,0,0,0
OUT,10:30:00,10:30:00,Z,1,0,0
)"sv;

void expect_bus_to_x(std::string_view const destination,
                     routing::location_match_mode const mode) {
  auto const tt = loader::load(
      loader::mem_dir::read(kOriginWithOffsetIsDestination),
      {2019_y / May / 1, 2019_y / May / 2}, {.adjust_footpaths_ = false});
  auto const loc = [&](std::string_view const id) {
    return tt.find(location_id{id, source_idx_t{0}}).value();
  };
  auto const results = raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = interval{kTen, kTen + 5min},
                     .start_match_mode_ = mode,
                     .dest_match_mode_ = mode,
                     .start_ = {{loc("S"), 0min, 0U}, {loc("X"), 5min, 0U}},
                     .destination_ = {{loc(destination), 0min, 0U}}},
      direction::kForward);
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(kTen + 1min, results.begin()->start_time_);
  EXPECT_EQ(kTen + 3min, results.begin()->dest_time_);
}

TEST(routing, origin_with_offset_is_destination_keeps_faster_trip) {
  expect_bus_to_x("X", routing::location_match_mode::kExact);
}

// Same, but X is a destination only through its parent PX.
TEST(routing, origin_with_offset_is_destination_child_keeps_faster_trip) {
  expect_bus_to_x("PX", routing::location_match_mode::kEquivalent);
}
