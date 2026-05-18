#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"

#include "nigiri/routing/raptor/pong.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

void load(timetable& tt, std::string_view const gtfs) {
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, mem_dir::read(std::string{gtfs}), tt);
  finalize(tt);
}

location_idx_t find_loc(timetable const& tt, std::string_view const id) {
  auto const x = tt.find(location_id{id, source_idx_t{0}});
  EXPECT_TRUE(x.has_value()) << id;
  return x.value_or(location_idx_t::invalid());
}

}  // namespace

// Per-section bikes_allowed enforcement on block-concatenated trips. The
// block trip A→B→C has bikes_allowed_seq={false, true} (A→B without bikes,
// B→C with bikes). `tt.has_bike_transport(r)` returns true at the route
// level, so the route is a candidate — but the journey A→C requires bikes
// through both sections and must be rejected because A→B is bikeless.
TEST(pong, get_earliest_alternative_per_transport_bikes_allowed) {
  constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,0.0,1.0,,
B,B,2.0,3.0,,
C,C,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3
R2,DB,R2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,bikes_allowed,cars_allowed
R1,S,T_NO_BIKE,,BLK,2,2
R2,S,T_BIKE,,BLK,1,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_NO_BIKE,10:00:00,10:00:00,A,0
T_NO_BIKE,10:15:00,10:15:00,B,1
T_BIKE,10:15:00,10:15:00,B,0
T_BIKE,10:30:00,10:30:00,C,1

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

  auto tt = timetable{};
  load(tt, kGTFS);
  auto const A = find_loc(tt, "A");
  auto const C = find_loc(tt, "C");

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  // Trip A 10:00 → B 10:15 → C 10:30 in Europe/Berlin (= 08:00–08:30 UTC).
  auto const from_arr =
      unixtime_t{sys_days{2024_y / June / 19}} + 6h;  // 06:00 UTC
  auto const to_dep =
      unixtime_t{sys_days{2024_y / June / 19}} + 12h;  // 12:00 UTC

  // Sanity: without require_bike, the block trip is reachable (proves the
  // block A→B→C route was constructed).
  {
    auto q_any = routing::query{};
    auto const r_any = routing::get_earliest_alternative(tt, &rtt, q_any, A, C,
                                                         from_arr, to_dep);
    ASSERT_TRUE(r_any.has_value()) << "block trip A→B→C must be reachable";
  }

  auto q = routing::query{};
  q.require_bike_transport_ = true;

  auto const result =
      routing::get_earliest_alternative(tt, &rtt, q, A, C, from_arr, to_dep);

  EXPECT_FALSE(result.has_value())
      << "block trip A→B (no bikes) → B→C must be rejected for require_bike";
}

// Per-section cars_allowed on a block trip. Symmetric to the bike test:
// block trip A→B (cars) → B→C (no cars); journey A→C must be rejected for
// require_car_transport_=true.
TEST(pong, get_earliest_alternative_per_transport_cars_allowed) {
  constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,0.0,1.0,,
B,B,2.0,3.0,,
C,C,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3
R2,DB,R2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,bikes_allowed,cars_allowed
R1,S,T_CAR,,BLK,2,1
R2,S,T_NO_CAR,,BLK,2,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_CAR,10:00:00,10:00:00,A,0
T_CAR,10:15:00,10:15:00,B,1
T_NO_CAR,10:15:00,10:15:00,B,0
T_NO_CAR,10:30:00,10:30:00,C,1

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

  auto tt = timetable{};
  load(tt, kGTFS);
  auto const A = find_loc(tt, "A");
  auto const C = find_loc(tt, "C");

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  auto const from_arr =
      unixtime_t{sys_days{2024_y / June / 19}} + 6h;  // 06:00 UTC
  auto const to_dep = unixtime_t{sys_days{2024_y / June / 19}} + 12h;

  // Sanity: without require_car, the block trip is reachable.
  {
    auto q_any = routing::query{};
    auto const r_any = routing::get_earliest_alternative(tt, &rtt, q_any, A, C,
                                                         from_arr, to_dep);
    ASSERT_TRUE(r_any.has_value()) << "block trip A→B→C must be reachable";
  }

  auto q = routing::query{};
  q.require_car_transport_ = true;

  auto const result =
      routing::get_earliest_alternative(tt, &rtt, q, A, C, from_arr, to_dep);

  EXPECT_FALSE(result.has_value())
      << "block trip A→B (cars) → B→C (no cars) must be rejected";
}

// TD footpath at the ingress side (q.prf_idx_ != 0).
// Ingress X→A only via a TD footpath (5min from 10:55). Static is blocked
// (kMaxDuration). With from_arr=10:50, the TD footpath enables boarding at
// A at 11:00.
TEST(pong, get_earliest_alternative_td_footpath_ingress) {
  // One transport A→B at 11:00; reaching A on time requires the TD footpath
  // from X→A active at 10:55 (5min duration). The static footpath X→A is
  // set to kMaxDuration to exclude it.
  constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
X,X,0.0,1.0,,
A,A,0.0,1.0,,
B,B,2.0,3.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,DB,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
R,S,T,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T,11:00:00,11:00:00,A,0
T,11:30:00,11:30:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

  constexpr auto const kProfile = profile_idx_t{2U};
  auto tt = timetable{};
  load(tt, kGTFS);

  auto const X = find_loc(tt, "X");
  auto const A = find_loc(tt, "A");
  auto const B = find_loc(tt, "B");

  // Static footpath X→A is "blocked" (kMaxDuration).
  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][X].push_back(
      footpath(A, footpath::kMaxDuration));
  tt.locations_.footpaths_in_[kProfile][A].push_back(
      footpath(X, footpath::kMaxDuration));

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  // Trip T departs A at 11:00 Berlin local = 09:00 UTC (CEST).
  // TD footpath X→A active from 10:55 Berlin (= 08:55 UTC) with 5min
  // duration — i.e. starting walk at 08:55 UTC arrives at A at 09:00 UTC,
  // just in time to board T.
  rtt.has_td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.has_td_footpaths_in_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());
  rtt.has_td_footpaths_out_[kProfile].set(X, true);
  rtt.has_td_footpaths_in_[kProfile].set(A, true);
  rtt.td_footpaths_out_[kProfile][X].push_back(td_footpath{
      A, unixtime_t{sys_days{2024_y / June / 19}} + 8h + 55min, 5min});
  rtt.td_footpaths_in_[kProfile][A].push_back(td_footpath{
      X, unixtime_t{sys_days{2024_y / June / 19}} + 8h + 55min, 5min});

  auto q = routing::query{};
  q.prf_idx_ = kProfile;

  auto const result = routing::get_earliest_alternative(
      tt, &rtt, q, X, B, unixtime_t{sys_days{2024_y / June / 19}} + 8h + 50min,
      unixtime_t{sys_days{2024_y / June / 19}} + 12h);

  EXPECT_TRUE(result.has_value())
      << "TD footpath X→A should enable boarding T at A 11:00";
}

// TD footpath at the egress side (q.prf_idx_ != 0).
// Egress B→Y only via a TD footpath active at 11:30 (5min). Static is
// kMaxDuration.
TEST(pong, get_earliest_alternative_td_footpath_egress) {
  // One transport A→B at 11:00; the egress B→Y requires a TD footpath
  // active at 11:30 (5min duration). The static footpath B→Y is set to
  // kMaxDuration to exclude it.
  constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,DB,https://db.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,0.0,1.0,,
B,B,2.0,3.0,,
Y,Y,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,DB,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
R,S,T,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T,11:00:00,11:00:00,A,0
T,11:30:00,11:30:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1
)"sv;

  constexpr auto const kProfile = profile_idx_t{2U};
  auto tt = timetable{};
  load(tt, kGTFS);

  auto const A = find_loc(tt, "A");
  auto const B = find_loc(tt, "B");
  auto const Y = find_loc(tt, "Y");

  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B].push_back(
      footpath(Y, footpath::kMaxDuration));
  tt.locations_.footpaths_in_[kProfile][Y].push_back(
      footpath(B, footpath::kMaxDuration));

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  rtt.has_td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.has_td_footpaths_in_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());
  // Trip T arrives B at 11:30 Berlin local = 09:30 UTC (CEST).
  // TD footpath B→Y active from 11:30 Berlin (= 09:30 UTC) with 5min duration.
  rtt.has_td_footpaths_out_[kProfile].set(B, true);
  rtt.has_td_footpaths_in_[kProfile].set(Y, true);
  rtt.td_footpaths_out_[kProfile][B].push_back(td_footpath{
      Y, unixtime_t{sys_days{2024_y / June / 19}} + 9h + 30min, 5min});
  rtt.td_footpaths_in_[kProfile][Y].push_back(td_footpath{
      B, unixtime_t{sys_days{2024_y / June / 19}} + 9h + 30min, 5min});

  auto q = routing::query{};
  q.prf_idx_ = kProfile;

  auto const result = routing::get_earliest_alternative(
      tt, &rtt, q, A, Y, unixtime_t{sys_days{2024_y / June / 19}} + 8h + 50min,
      unixtime_t{sys_days{2024_y / June / 19}} + 13h);

  EXPECT_TRUE(result.has_value())
      << "TD footpath B→Y should enable alighting T at B 11:30";
}
