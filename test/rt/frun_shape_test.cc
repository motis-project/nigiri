#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <string_view>
#include <variant>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"
#include "nigiri/shape.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using std::operator""sv;

// linked from gtfs/shape_test.cc
shapes_storage create_tmp_shapes_storage(char const*);

namespace std {
bool operator==(std::span<geo::latlng const> const& lhs,
                std::span<geo::latlng const> const& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  auto const zip = std::views::zip(lhs, rhs);
  return std::all_of(begin(zip), end(zip), [](auto const ab) {
    auto const& [a, b] = ab;
    return a == b;
  });
}
}  // namespace std

namespace {

constexpr auto kSchedule = R"(
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_id
test,https://test.com,Europe/Berlin,DE,0800123456,AGENCY_1

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,1.0,1.0
B,B,1.0,2.0
C,C,1.0,3.0
D,D,1.0,4.0
F,F,2.0,1.0
G,G,3.0,1.0
H,H,3.0,2.0
I,I,3.0,3.0
J,J,4.0,3.0
K,K,5.0,3.0
M,M,2.0,2.0
N,N,3.0,3.0
O,O,4.0,4.0
Q,O,0.0,0.0

# calendar_dates.txt
service_id,date,exception_type
SERVICE_1,20240101,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
ROUTE_1,AGENCY_1,Route 1,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,shape_id,
ROUTE_1,SERVICE_1,TRIP_1,E,BLOCK_1,SHAPE_1,
ROUTE_1,SERVICE_1,TRIP_2,E,BLOCK_2,SHAPE_2,
ROUTE_1,SERVICE_1,TRIP_3,E,BLOCK_2,SHAPE_3,
ROUTE_1,SERVICE_1,TRIP_4,E,BLOCK_2,SHAPE_4,
ROUTE_1,SERVICE_1,TRIP_5,E,BLOCK_3,SHAPE_5,
ROUTE_1,SERVICE_1,TRIP_6,E,BLOCK_4,

# shapes.txt
"shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
SHAPE_1,1.0,1.0,0
SHAPE_1,0.5,1.5,1
SHAPE_1,1.0,2.0,2
SHAPE_1,0.5,2.5,3
SHAPE_1,1.0,3.0,4
SHAPE_1,0.5,3.5,5
SHAPE_1,1.0,4.0,6
SHAPE_2,1.0,1.0,0
SHAPE_2,1.5,0.5,1
SHAPE_2,2.0,1.0,2
SHAPE_2,2.5,0.5,3
SHAPE_2,3.0,1.0,4
SHAPE_3,3.0,1.0,0
SHAPE_3,3.5,1.5,1
SHAPE_3,3.0,2.0,2
SHAPE_3,3.5,2.5,3
SHAPE_3,3.0,3.0,4
SHAPE_4,3.0,3.0,0
SHAPE_4,3.5,2.5,1
SHAPE_4,4.0,3.0,2
SHAPE_4,4.5,2.5,3
SHAPE_4,5.0,3.0,4

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TRIP_1,10:00:00,10:00:00,A,1,0,0
TRIP_1,11:00:00,11:00:00,B,2,0,0
TRIP_1,12:00:00,12:00:00,C,3,0,0
TRIP_1,13:00:00,13:00:00,D,4,0,0
TRIP_2,10:00:00,10:00:00,A,1,0,0
TRIP_2,11:00:00,11:00:00,F,2,0,0
TRIP_2,12:00:00,12:00:00,G,3,0,0
TRIP_3,12:00:00,12:00:00,G,3,0,0
TRIP_3,13:00:00,13:00:00,H,4,0,0
TRIP_3,14:00:00,14:00:00,I,5,0,0
TRIP_4,14:00:00,14:00:00,I,5,0,0
TRIP_4,15:00:00,15:00:00,J,6,0,0
TRIP_4,16:00:00,16:00:00,K,6,0,0
TRIP_5,10:00:00,10:00:00,A,1,0,0
TRIP_5,11:00:00,11:00:00,M,2,0,0
TRIP_5,12:00:00,12:00:00,N,3,0,0
TRIP_5,13:00:00,13:00:00,O,4,0,0
TRIP_6,10:00:00,10:00:00,A,1,0,0
TRIP_6,11:00:00,11:00:00,Q,2,0,0

)"sv;

TEST(
    rt,
    frun_for_each_shape_point_when_shapes_are_provided_then_process_all_subshapes) {
  auto const schedule = mem_dir::read(kSchedule);
  auto shapes_data = create_tmp_shapes_storage("rfun-for-each-shape-point");

  // Load static timetable.
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / January / 1},
                    date::sys_days{2024_y / January / 2}};
  load_timetable({}, source_idx_t{0}, schedule, tt, nullptr, &shapes_data);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / January / 1});

  auto expected_shape = std::vector<geo::latlng>{};
  auto leg_shape = std::vector<geo::latlng>{};
  auto plot_point = [&leg_shape](geo::latlng const& point) {
    leg_shape.push_back(point);
  };

  // TRIP_1
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_1");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // Full trip
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{0}, stop_idx_t{3 + 1}}, plot_point);

      expected_shape = {
          geo::latlng{1.0F, 1.0F}, geo::latlng{0.5F, 1.5F},
          geo::latlng{1.0F, 2.0F}, geo::latlng{0.5F, 2.5F},
          geo::latlng{1.0F, 3.0F}, geo::latlng{0.5F, 3.5F},
          geo::latlng{1.0F, 4.0F},
      };
      EXPECT_EQ(expected_shape, leg_shape);
    }
    // Single leg
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1}, stop_idx_t{2 + 1}}, plot_point);

      expected_shape = {
          geo::latlng{1.0F, 2.0F},
          geo::latlng{0.5F, 2.5F},
          geo::latlng{1.0F, 3.0F},
      };
      EXPECT_EQ(expected_shape, leg_shape);
    }
    // Single stop
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{0}, stop_idx_t{0 + 1}}, plot_point);

      expected_shape = {
          geo::latlng{1.0F, 1.0F},
      };
      EXPECT_EQ(expected_shape, leg_shape);
    }
  }
  // TRIP_6 (trip without shape)
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_6");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    leg_shape.clear();
    full_run.for_each_shape_point(
        &shapes_data, interval{stop_idx_t{0}, stop_idx_t{1 + 1}}, plot_point);

    expected_shape = {
        geo::latlng{1.0F, 1.0F},
        geo::latlng{0.0F, 0.0F},
    };
    EXPECT_EQ(expected_shape, leg_shape);
  }
  // frun containing a sub trip
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_1");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create sub run containing single trip leg
    auto const r_modified =
        rt::run{r.t_, interval{stop_idx_t{2}, stop_idx_t{4}}, r.rt_};
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r_modified};

    leg_shape.clear();
    full_run.for_each_shape_point(
        &shapes_data, interval{stop_idx_t{0}, stop_idx_t{1 + 1}}, plot_point);

    expected_shape = {
        geo::latlng{1.0F, 3.0F},
        geo::latlng{0.5F, 3.5F},
        geo::latlng{1.0F, 4.0F},
    };
    EXPECT_EQ(expected_shape, leg_shape);
  }
}

}  // namespace