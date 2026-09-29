#include "gtest/gtest.h"

#include "fmt/core.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hotel_trains.h"
#include "nigiri/loader/import_context.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::chrono_literals;

namespace {

mem_dir test_files(std::string_view const date) {
  return mem_dir::read(fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# calendar_dates.txt
service_id,date,exception_type
S,{},1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,B 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,RE 1,
R2,S,T2,B 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
T2,10:00:00,10:00:00,B,1,0,0
T2,11:00:00,11:00:00,C,2,0,0
)",
                                   date));
}

timetable load(sys_days const from,
               sys_days const to,
               std::string_view const service_day) {
  auto tt = timetable{};
  tt.date_range_ = {from, to};
  tt.n_sources_ = 1U;
  auto ctx = import_context{tt};
  register_special_stations(tt, ctx);
  gtfs::load_timetable({}, source_idx_t{0}, test_files(service_day), tt, ctx);
  // hotel next to A and C; C is only served by a bus
  add_hotel_night_trains(
      tt, ctx, std::array{geo::latlng{0.001, 1.0}, geo::latlng{4.0, 5.001}});
  finalize(tt, ctx);
  return tt;
}

std::vector<route_idx_t> night_routes(timetable const& tt,
                                      std::string_view const stop) {
  auto const l = tt.find(location_id{stop, source_idx_t{0}}).value();
  auto routes = std::vector<route_idx_t>{};
  for (auto const r : tt.location_routes_[l]) {
    if (tt.route_clasz_[r] == clasz::kNight) {
      routes.push_back(r);
    }
  }
  return routes;
}

}  // namespace

TEST(hotel_trains, only_rail_stations_with_hotel) {
  auto const tt = load(sys_days{2024_y / June / 18},
                       sys_days{2024_y / June / 21}, "20240619");

  EXPECT_EQ(2U, tt.n_sources());
  EXPECT_EQ(1U, night_routes(tt, "A").size());
  EXPECT_TRUE(night_routes(tt, "B").empty());  // no hotel nearby
  EXPECT_TRUE(night_routes(tt, "C").empty());  // no rail or coach service
}

TEST(hotel_trains, departs_22_arrives_08_local_time) {
  auto const tt = load(sys_days{2024_y / June / 18},
                       sys_days{2024_y / June / 21}, "20240619");

  auto const r = night_routes(tt, "A").front();
  auto const a = tt.find(location_id{"A", source_idx_t{0}}).value();
  ASSERT_EQ(2U, tt.route_location_seq_[r].size());
  EXPECT_EQ(a, stop{tt.route_location_seq_[r][0]}.location_idx());
  EXPECT_EQ(a, stop{tt.route_location_seq_[r][1]}.location_idx());

  // summer time (UTC+2): 22:00 local = 20:00 UTC, 08:00 local = 06:00 UTC
  ASSERT_EQ(1U, tt.route_transport_ranges_[r].size());
  auto const t = tt.route_transport_ranges_[r][0];
  EXPECT_EQ(20 * 60, tt.event_mam(r, t, 0, event_type::kDep).count());
  EXPECT_EQ(24 * 60 + 6 * 60, tt.event_mam(r, t, 1, event_type::kArr).count());
  EXPECT_EQ(3U, tt.bitfields_[tt.transport_traffic_days_[t]].count());
}

TEST(hotel_trains, one_transport_per_utc_offset) {
  // daylight saving time starts on 2024-03-31
  auto const tt = load(sys_days{2024_y / March / 25},
                       sys_days{2024_y / April / 5}, "20240327");

  auto const r = night_routes(tt, "A").front();
  ASSERT_EQ(2U, tt.route_transport_ranges_[r].size());
  auto const winter = tt.route_transport_ranges_[r][1];  // sorted by departure
  auto const summer = tt.route_transport_ranges_[r][0];
  EXPECT_EQ(20 * 60, tt.event_mam(r, summer, 0, event_type::kDep).count());
  EXPECT_EQ(21 * 60, tt.event_mam(r, winter, 0, event_type::kDep).count());
}

TEST(hotel_trains, routing_still_uses_train) {
  auto const tt = load(sys_days{2024_y / June / 18},
                       sys_days{2024_y / June / 21}, "20240619");

  // the hotel route never shortens a journey, the train must stay the result
  auto const results =
      test::raptor_search(tt, nullptr, "A", "B", "2024-06-19 05:00 UTC");
  ASSERT_EQ(1U, results.size());
  EXPECT_EQ(0U, results.begin()->transfers_);
  EXPECT_EQ(sys_days{2024_y / June / 19} + 9h, results.begin()->arrival_time());
}
