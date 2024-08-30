#include "gtest/gtest.h"

#include "geo/polyline.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/shape.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "./shape_test.h"

using namespace nigiri;
using namespace date;
using namespace std::string_view_literals;

constexpr auto const test_files_without_shapes = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
LH,Lufthansa,https://lufthansa.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AIR,LH,X,,,1100
R1,DB,1,,,3
R2,DB,2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AIR,S1,AIR,,
R1,S1,T1,,
R2,S1,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AIR,10:00:00,10:00:00,A,0,0,0
AIR,11:00:00,11:00:00,C,1,0,0
T1,10:00:00,10:00:00,A,0,0,0
T1,10:55:00,10:55:00,B,1,0,0
T2,11:05:00,11:05:00,B,0,0,0
T2,12:00:00,12:00:00,C,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
)"sv;

TEST(gtfs, shapeRequest_noShape_getEmptyVector) {
  auto mmap = loader::gtfs::shape_test_mmap{"shape-route-trip-with-shape"};
  auto& shape_data = mmap.get_shape_data();

  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files_without_shapes),
                               tt);
  loader::finalize(tt);

  auto const shape_by_trip_index = get_shape(trip_idx_t{1}, tt, shape_data);
  auto const shape_by_shape_index = get_shape(shape_idx_t{1},   shape_data);

  EXPECT_EQ(geo::polyline{}, shape_by_trip_index);
  EXPECT_EQ(geo::polyline{}, shape_by_shape_index);
}

constexpr auto const test_files_with_shapes = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,0.0,,,
B,B,,2.0,0.0,,,
C,C,,2.504,1.999,,,
D,D,,0.0,2.0,,,
E,E,,0.0,3.0,,,
F,F,,1.0,4.0,,,
G,G,,3.0,4.0,,,
H,H,,1.0,6.0,,,
I,I,,2.0,6.0,,,
J,J,,4.0,5.0,,,
K,K,,6.0,4.0,,,
L,L,,4.0,2.0,,,
M,M,,5.0,2.0,,,
N,N,,6.0,3.0,,,
O,O,,5.0,4.0,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,2
R3,DB,3,,,2
R4,DB,4,,,1
R5,DB,5,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,shape_id
R1,S1,Trip 1,,,not used
R2,S1,Trip 2,,,Interior
R3,S1,Trip 3,,,Last
R4,S1,Trip 4,,,
R5,S1,Trip 5,,,Sec1

# shapes.txt
"shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
Sec1,0.0,0.0,0
Sec1,2.0,0.0,1
Sec1,3.0,4.0,2
Sec1,1.0,4.0,4
not used,1.0,4.0,0
not used,0.0,3.0,1
not used,0.0,2.0,2
Interior,1.0,4.0,10001
Interior,3.0,4.0,20002
Interior,6.0,4.0,20003
Last,4.0,5.0,1
Last,5.5,2.5,2
Last,5.5,3.0,3
Last,6.0,3.0,5
Last,5.0,2.0,8
Last,4.0,2.0,13

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
Trip 1,11:15:00,11:15:00,F,0,0,0
Trip 1,11:30:00,11:30:00,E,1,0,0
Trip 1,11:45:00,11:45:00,D,2,0,0
Trip 2,11:10:00,11:10:00,F,2000,0,0
Trip 2,11:20:00,11:20:00,G,2001,0,0
Trip 2,11:30:00,11:30:00,O,2002,0,0
Trip 2,11:40:00,11:40:00,K,2003,0,0
Trip 3,11:30:00,11:30:00,J,3,0,0
Trip 3,11:45:00,11:45:00,O,6,0,0
Trip 3,12:00:00,12:00:00,N,9,0,0
Trip 3,12:15:00,12:15:00,M,12,0,0
Trip 3,12:30:00,12:30:00,L,15,0,0
Trip 4,13:00:00,13:00:00,H,0,0,0
Trip 4,14:00:00,14:00:00,I,1,0,0
Trip 5,10:00:00,10:00:00,A,2,0,0
Trip 5,10:15:00,10:15:00,B,3,0,0
Trip 5,10:30:00,10:30:00,C,5,0,0
Trip 5,10:45:00,10:45:00,G,8,0,0
Trip 5,11:00:00,11:00:00,F,13,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
)"sv;

TEST(gtfs, shapeRequest_singleTripWithShape_getFullShape) {
  auto mmap = loader::gtfs::shape_test_mmap{"shape-route-trip-with-shape"};
  auto& shape_data = mmap.get_shape_data();

  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  auto local_bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  loader::gtfs::load_timetable(
      {}, source_idx_t{1}, loader::mem_dir::read(test_files_with_shapes), tt,
      local_bitfield_indices, nullptr, &shape_data);
  loader::finalize(tt);

  // Testing shape 'Last', used by 'Trip 3' (index == 2)
  auto const shape_by_trip_index =
      get_shape(trip_idx_t{2}, tt, shape_data);
  auto const shape_by_shape_index =
      get_shape(shape_idx_t{3}, shape_data);

  auto const expected_shape = geo::polyline{
      {4.0f, 5.0f}, {5.5f, 2.5f}, {5.5f, 3.0f},
      {6.0f, 3.0f}, {5.0f, 2.0f}, {4.0f, 2.0f},
  };
  EXPECT_EQ(expected_shape, shape_by_trip_index);
  EXPECT_EQ(expected_shape, shape_by_shape_index);
}

TEST(gtfs, shapeRequest_singleTripWithoutShape_getEmptyShape) {
  auto mmap = loader::gtfs::shape_test_mmap{"shape-route-trip-without-shape"};
  auto& shape_data = mmap.get_shape_data();

  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  auto local_bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  loader::gtfs::load_timetable(
      {}, source_idx_t{1}, loader::mem_dir::read(test_files_with_shapes), tt,
      local_bitfield_indices, nullptr, &shape_data);
  loader::finalize(tt);

  // Testing trip without shape, i.e. 'Trip 4' (index == 3)
  auto const shape_by_trip_index =
      get_shape(trip_idx_t{3}, tt, shape_data);
  auto const shape_by_shape_index =
      get_shape(shape_idx_t::invalid(), shape_data);

  auto const expected_shape = geo::polyline{};
  EXPECT_EQ(expected_shape, shape_by_trip_index);
  EXPECT_EQ(expected_shape, shape_by_shape_index);
}
