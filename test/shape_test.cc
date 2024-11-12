#include "gtest/gtest.h"

#include "geo/box.h"
#include "geo/polyline.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/span_cmp.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace nigiri;
using namespace date;
using namespace std::string_view_literals;

namespace {

constexpr auto const kWithShapes = R"(
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
Last,4.0,1.9,11
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

}  // namespace

TEST(shape, single_trip_with_shape) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  auto local_bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  auto shapes_data = shapes_storage{"shape-route-trip-with-shape",
                                    cista::mmap::protection::WRITE};
  loader::gtfs::load_timetable({}, source_idx_t{1},
                               loader::mem_dir::read(kWithShapes), tt,
                               local_bitfield_indices, nullptr, &shapes_data);
  loader::finalize(tt);

  // Testing shape 'Last', used by 'Trip 3' (index == 2)
  {
    auto const shape_by_trip_idx = shapes_data.get_shape(trip_idx_t{2});
    auto const shape_by_shape_idx = shapes_data.get_shape(shape_idx_t{3});

    auto const expected_shape = geo::polyline{
        {4.0, 5.0}, {5.5, 2.5}, {5.5, 3.0}, {6.0, 3.0},
        {5.0, 2.0}, {4.0, 1.9}, {4.0, 2.0},
    };
    EXPECT_EQ(expected_shape, shape_by_trip_idx);
    EXPECT_EQ(expected_shape, shape_by_shape_idx);
  }

  // Testing trip without shape, i.e. 'Trip 4' (index == 3)
  {
    auto const shape_by_trip_idx = shapes_data.get_shape(trip_idx_t{3});
    auto const shape_by_shape_idx =
        shapes_data.get_shape(shape_idx_t::invalid());

    EXPECT_TRUE(shape_by_trip_idx.empty());
    EXPECT_TRUE(shape_by_shape_idx.empty());
  }

  // Testing out of bounds
  {
    auto const shape_by_huge_trip_idx = shapes_data.get_shape(trip_idx_t{999});
    auto const shape_by_huge_shape_idx =
        shapes_data.get_shape(shape_idx_t{999});
    auto const shape_by_invalid_trip_idx =
        shapes_data.get_shape(trip_idx_t::invalid());
    auto const shape_by_invalid_shape_idx =
        shapes_data.get_shape(shape_idx_t::invalid());

    EXPECT_TRUE(shape_by_huge_trip_idx.empty());
    EXPECT_TRUE(shape_by_huge_shape_idx.empty());
    EXPECT_TRUE(shape_by_invalid_trip_idx.empty());
    EXPECT_TRUE(shape_by_invalid_shape_idx.empty());
  }

  // Testing bounding boxes
  {
    // Full shape in bounding box included
    EXPECT_EQ((geo::make_box({{0.0, 2.0}, {1.0, 4.0}})),
              shapes_data.get_bounding_box(route_idx_t{0U}));
    // Shape in bounding box included
    EXPECT_EQ((geo::make_box({{4.0, 1.9}, {6.0, 5.0}})),
              shapes_data.get_bounding_box(route_idx_t{2U}));
    // Bounding boxes for segments
    // Bounding box extended by shape
    {
      auto const extended_by_shape =
          shapes_data.get_bounding_box(route_idx_t{2}, 3);
      ASSERT_TRUE(extended_by_shape.has_value());
      EXPECT_EQ((geo::make_box({{4.0, 1.9}, {5.0, 2.0}})), *extended_by_shape);

      auto const before_last_extend =
          shapes_data.get_bounding_box(route_idx_t{2}, 2);
      ASSERT_TRUE(before_last_extend.has_value());
      EXPECT_EQ((geo::make_box({{5.0, 2.0}, {6.0, 3.0}})), *before_last_extend);
    }
    // Shape contained in bounding box
    {
      EXPECT_FALSE(shapes_data.get_bounding_box(route_idx_t{4}, 0).has_value());
    }
  }
}
