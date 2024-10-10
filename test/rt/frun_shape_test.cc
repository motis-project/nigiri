#include "gtest/gtest.h"

#include <ranges>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "geo/latlng.h"
#include "geo/polyline.h"

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
N1,N1,3.5,3.0
O,O,4.0,4.0
Q,Q,0.0,0.0
S,S,4.0,1.0
T,T,5.0,1.0
U,U,6.0,2.0
V,V,7.0,3.0
W,W,7.0,2.0
X,X,7.0,1.0

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
ROUTE_1,SERVICE_1,TRIP_5+,E,BLOCK_5+,SHAPE_5,
ROUTE_1,SERVICE_1,TRIP_6,E,BLOCK_4,,
ROUTE_1,SERVICE_1,TRIP_7,E,BLOCK_5,SHAPE_2,
ROUTE_1,SERVICE_1,TRIP_8,E,BLOCK_5,,
ROUTE_1,SERVICE_1,TRIP_9,E,BLOCK_5,,
ROUTE_1,SERVICE_1,TRIP_10,E,BLOCK_5,SHAPE_6,
ROUTE_1,SERVICE_1,TRIP_11,E,BLOCK_6,SHAPE_7,
ROUTE_1,SERVICE_1,TRIP_12,E,BLOCK_7,SHAPE_8,
ROUTE_1,SERVICE_1,TRIP_13,E,BLOCK_8,SHAPE_9,

# shapes.txt
"shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence","shape_dist_traveled"
SHAPE_1,1.0,1.0,0,
SHAPE_1,0.5,1.5,1,
SHAPE_1,1.0,2.0,2,
SHAPE_1,0.5,2.5,3,
SHAPE_1,1.0,3.0,4,
SHAPE_1,0.5,3.5,5,
SHAPE_1,1.0,4.0,6,
SHAPE_2,1.0,1.0,0,
SHAPE_2,1.5,0.5,1,
SHAPE_2,2.0,1.0,2,
SHAPE_2,2.5,0.5,3,
SHAPE_2,3.0,1.0,4,
SHAPE_3,3.0,1.0,0,
SHAPE_3,3.5,1.5,1,
SHAPE_3,3.0,2.0,2,
SHAPE_3,3.5,2.5,3,
SHAPE_3,3.0,3.0,4,
SHAPE_4,3.0,3.0,0,
SHAPE_4,3.5,2.5,1,
SHAPE_4,4.0,3.0,2,
SHAPE_4,4.5,2.5,3,
SHAPE_4,5.0,3.0,4,
SHAPE_5,1.0,1.0,0,0.00
SHAPE_5,2.0,1.5,1,1.12
SHAPE_5,2.0,2.0,2,1.62
SHAPE_5,2.0,2.5,3,2.12
SHAPE_5,1.5,2.0,4,2.83
SHAPE_5,2.0,2.0,5,3.33
SHAPE_5,3.0,2.0,6,4.33
SHAPE_5,3.0,3.0,7,5.33
SHAPE_5,3.0,3.5,8,5.83
SHAPE_5,2.5,3.0,9,6.53
SHAPE_5,3.0,3.0,10,7.03
SHAPE_5,3.5,3.0,11,7.53
SHAPE_5,4.0,4.0,11,8.71
SHAPE_6,7.0,3.0,1,
SHAPE_6,6.5,2.5,2,
SHAPE_6,7.0,2.0,3,
SHAPE_6,6.5,1.5,4,
SHAPE_6,7.0,1.0,5,
SHAPE_7,1.0,1.0,0,0.0
SHAPE_7,1.5,1.5,1,0.7
SHAPE_7,2.0,2.0,2,1.4
SHAPE_7,2.5,2.5,3,2.1
SHAPE_7,3.0,3.0,4,2.9
SHAPE_7,3.5,3.5,5,3.5
SHAPE_7,4.0,4.0,6,4.2
SHAPE_8,1.0,1.0,0,0.0
SHAPE_8,1.5,1.5,1,0.0
SHAPE_8,2.0,2.0,2,2.0
SHAPE_8,3.0,3.0,3,4.0
SHAPE_9,1.0,1.0,0,
SHAPE_9,2.5,2.5,1,
SHAPE_9,3.625,3.625,2,
SHAPE_9,4.0,4.0,3,
SHAPE_9,4.0,4.0,4,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,shape_dist_traveled
TRIP_1,10:00:00,10:00:00,A,1,0,0,
TRIP_1,11:00:00,11:00:00,B,2,0,0,
TRIP_1,12:00:00,12:00:00,C,3,0,0,
TRIP_1,13:00:00,13:00:00,D,4,0,0,
TRIP_2,10:00:00,10:00:00,A,1,0,0,
TRIP_2,11:00:00,11:00:00,F,2,0,0,
TRIP_2,12:00:00,12:00:00,G,3,0,0,
TRIP_3,12:00:00,12:00:00,G,3,0,0,
TRIP_3,13:00:00,13:00:00,H,4,0,0,
TRIP_3,14:00:00,14:00:00,I,5,0,0,
TRIP_4,14:00:00,14:00:00,I,5,0,0,
TRIP_4,15:00:00,15:00:00,J,6,0,0,
TRIP_4,16:00:00,16:00:00,K,7,0,0,
TRIP_5,10:00:00,10:00:00,A,1,0,0,0.00
TRIP_5,11:00:00,11:00:00,M,2,0,0,1.62
TRIP_5,12:00:00,12:00:00,N,3,0,0,7.03
TRIP_5,13:00:00,13:00:00,O,4,0,0,8.71
TRIP_5+,10:00:00,10:00:00,A,1,0,0,
TRIP_5+,11:00:00,11:00:00,M,2,0,0,
TRIP_5+,12:00:00,12:00:00,N,3,0,0,
TRIP_5+,12:30:00,12:30:00,N1,4,0,0,
TRIP_5+,13:00:00,13:00:00,O,5,0,0,
TRIP_6,10:00:00,10:00:00,A,1,0,0,
TRIP_6,11:00:00,11:00:00,Q,2,0,0,
TRIP_7,10:00:00,10:00:00,A,1,0,0,
TRIP_7,11:00:00,11:00:00,F,2,0,0,
TRIP_7,12:00:00,12:00:00,G,3,0,0,
TRIP_8,12:00:00,12:00:00,G,3,0,0,
TRIP_8,13:00:00,13:00:00,S,4,0,0,
TRIP_8,14:00:00,14:00:00,T,5,0,0,
TRIP_9,14:00:00,14:00:00,T,0,0,0,
TRIP_9,15:00:00,15:00:00,U,1,0,0,
TRIP_9,16:00:00,16:00:00,V,2,0,0,
TRIP_10,17:00:00,17:00:00,V,1,0,0,
TRIP_10,18:00:00,18:00:00,W,2,0,0,
TRIP_10,19:00:00,19:00:00,X,3,0,0,
TRIP_11,10:00:00,10:00:00,A,1,0,0,0.00
TRIP_11,11:00:00,11:00:00,M,2,0,0,1.41
TRIP_11,12:00:00,12:00:00,N,3,0,0,2.83
TRIP_11,13:00:00,13:00:00,O,4,0,0,4.24
TRIP_12,10:00:00,10:00:00,A,1,0,0,0.0
TRIP_12,11:00:00,11:00:00,M,2,0,0,2.0
TRIP_12,12:00:00,12:00:00,N,3,0,0,4.0
TRIP_13,10:00:00,10:00:00,A,1,0,0,
TRIP_13,11:00:00,11:00:00,M,2,0,0,
TRIP_13,12:00:00,12:00:00,N,3,0,0,
TRIP_13,13:00:00,13:00:00,O,4,0,0,
TRIP_13,13:05:00,13:05:00,O,5,0,0,

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
  load_timetable({}, source_idx_t{0U}, schedule, tt, nullptr, &shapes_data);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / January / 1});

  auto leg_shape = std::vector<geo::latlng>{};
  auto const plot_point = [&leg_shape](geo::latlng const& point) {
    leg_shape.push_back(point);
  };

  // TRIP_1
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_1");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0U}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // Full trip
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{0U}, stop_idx_t{3U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {1.0F, 1.0F},
                    {0.5F, 1.5F},
                    {1.0F, 2.0F},
                    {0.5F, 2.5F},
                    {1.0F, 3.0F},
                    {0.5F, 3.5F},
                    {1.0F, 4.0F},
                }),
                leg_shape);
    }
    // Single leg
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{2U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {1.0F, 2.0F},
                    {0.5F, 2.5F},
                    {1.0F, 3.0F},
                }),
                leg_shape);
    }
    // Single stop
    {
      EXPECT_THROW(
          {
            try {
              full_run.for_each_shape_point(
                  &shapes_data, interval{stop_idx_t{0U}, stop_idx_t{0U + 1U}},
                  plot_point);
            } catch (std::runtime_error& e) {
              EXPECT_STREQ("Range must contain at least 2 stops. Is 1",
                           e.what());
              throw e;
            }
          },
          std::runtime_error);
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
    full_run.for_each_shape_point(&shapes_data,
                                  interval{stop_idx_t{0U}, stop_idx_t{1U + 1U}},
                                  plot_point);

    EXPECT_EQ((geo::polyline{
                  {1.0F, 1.0F},
                  {0.0F, 0.0F},
              }),
              leg_shape);
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
        rt::run{r.t_, interval{stop_idx_t{2U}, stop_idx_t{4U}}, r.rt_};
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r_modified};

    leg_shape.clear();
    full_run.for_each_shape_point(&shapes_data,
                                  interval{stop_idx_t{0U}, stop_idx_t{1U + 1U}},
                                  plot_point);

    EXPECT_EQ((geo::polyline{
                  {1.0F, 3.0F},
                  {0.5F, 3.5F},
                  {1.0F, 4.0F},
              }),
              leg_shape);
  }
  // sub trip of a merged trip
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_3");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create sub run containing single trip leg
    auto const r_modified =
        rt::run{r.t_, interval{stop_idx_t{3U}, stop_idx_t{5U}}, r.rt_};
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r_modified};

    // H -> I
    leg_shape.clear();
    full_run.for_each_shape_point(&shapes_data,
                                  interval{stop_idx_t{0U}, stop_idx_t{1U + 1U}},
                                  plot_point);

    EXPECT_EQ((geo::polyline{
                  {3.0F, 2.0F},
                  {3.5F, 2.5F},
                  {3.0F, 3.0F},
              }),
              leg_shape);
  }
  // Full run covering multiple trips
  {
    auto const results = nigiri::test::raptor_search(
        tt, &rtt, "F", "J",
        unixtime_t{sys_days{2024_y / January / 1}} + 10_hours);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(1, results.begin()->legs_.size());
    auto const& leg = results.begin()->legs_[0];
    ASSERT_TRUE(
        std::holds_alternative<nigiri::routing::journey::run_enter_exit>(
            leg.uses_));
    auto const& run_ee =
        std::get<nigiri::routing::journey::run_enter_exit>(leg.uses_);
    auto const full_run = rt::frun(tt, &rtt, run_ee.r_);

    // Shape for a single trip
    {
      // A -> F -> G
      {
        leg_shape.clear();
        full_run.for_each_shape_point(
            &shapes_data, interval{stop_idx_t{0U}, stop_idx_t{2U + 1U}},
            plot_point);

        EXPECT_EQ((geo::polyline{
                      {1.0F, 1.0F},
                      {1.5F, 0.5F},
                      {2.0F, 1.0F},
                      {2.5F, 0.5F},
                      {3.0F, 1.0F},
                  }),
                  leg_shape);
      }
      // G -> H -> I
      {
        leg_shape.clear();
        full_run.for_each_shape_point(
            &shapes_data, interval{stop_idx_t{2U}, stop_idx_t{4U + 1U}},
            plot_point);

        EXPECT_EQ((geo::polyline{
                      {3.0F, 1.0F},
                      {3.5F, 1.5F},
                      {3.0F, 2.0F},
                      {3.5F, 2.5F},
                      {3.0F, 3.0F},
                  }),
                  leg_shape);
      }
    }
    // Joined shape for continuous trips
    {
      // H -> I -> J
      {
        leg_shape.clear();
        full_run.for_each_shape_point(
            &shapes_data, interval{stop_idx_t{3U}, stop_idx_t{5U + 1U}},
            plot_point);

        EXPECT_EQ((geo::polyline{
                      {3.0F, 2.0F},
                      {3.5F, 2.5F},
                      {3.0F, 3.0F},
                      {3.5F, 2.5F},
                      {4.0F, 3.0F},
                  }),
                  leg_shape);
      }
      // F -> G -> H -> I -> J
      {
        leg_shape.clear();
        full_run.for_each_shape_point(
            &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{5U + 1U}},
            plot_point);

        EXPECT_EQ((geo::polyline{
                      {2.0F, 1.0F},
                      {2.5F, 0.5F},
                      {3.0F, 1.0F},
                      {3.5F, 1.5F},
                      {3.0F, 2.0F},
                      {3.5F, 2.5F},
                      {3.0F, 3.0F},
                      {3.5F, 2.5F},
                      {4.0F, 3.0F},
                  }),
                  leg_shape);
      }
    }
  }
  // Multiple trips, some with and some without shape
  {
    auto const results = nigiri::test::raptor_search(
        tt, &rtt, "F", "X",
        unixtime_t{sys_days{2024_y / January / 1}} + 10_hours);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(1, results.begin()->legs_.size());
    auto const& leg = results.begin()->legs_[0];
    ASSERT_TRUE(
        std::holds_alternative<nigiri::routing::journey::run_enter_exit>(
            leg.uses_));
    auto const& run_ee =
        std::get<nigiri::routing::journey::run_enter_exit>(leg.uses_);
    auto const full_run = rt::frun(tt, &rtt, run_ee.r_);

    // F -> G -> S -> T -> U -> V -> W -> X
    // Shape -> No shape -> No shape -> Shape
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{8U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {2.0F, 1.0F},
                    {2.5F, 0.5F},
                    {3.0F, 1.0F},
                    {4.0F, 1.0F},
                    {5.0F, 1.0F},
                    {6.0F, 2.0F},
                    {7.0F, 3.0F},
                    {6.5F, 2.5F},
                    {7.0F, 2.0F},
                    {6.5F, 1.5F},
                    {7.0F, 1.0F},
                }),
                leg_shape);
    }
    // F -> G -> S
    // Shape -> No shape
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{3U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {2.0F, 1.0F},
                    {2.5F, 0.5F},
                    {3.0F, 1.0F},
                    {4.0F, 1.0F},
                }),
                leg_shape);
    }
    // U -> V -> W
    // No shape -> Shape
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{5U}, stop_idx_t{7U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {6.0F, 2.0F},
                    {7.0F, 3.0F},
                    {6.5F, 2.5F},
                    {7.0F, 2.0F},
                }),
                leg_shape);
    }
  }
  // Trip with distance traveled available
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_5");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // Full trip
    {
      leg_shape.clear();

      full_run.for_each_shape_point(&shapes_data, full_run.stop_range_,
                                    plot_point);

      EXPECT_EQ((geo::polyline{
                    {1.0F, 1.0F},
                    {2.0F, 1.5F},
                    {2.0F, 2.0F},
                    {2.0F, 2.5F},
                    {1.5F, 2.0F},
                    {2.0F, 2.0F},
                    {3.0F, 2.0F},
                    {3.0F, 3.0F},
                    {3.0F, 3.5F},
                    {2.5F, 3.0F},
                    {3.0F, 3.0F},
                    {3.5F, 3.0F},
                    {4.0F, 4.0F},
                }),
                leg_shape);
    }
    // First leg, no loop
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{0U}, stop_idx_t{1U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {1.0F, 1.0F},
                    {2.0F, 1.5F},
                    {2.0F, 2.0F},
                }),
                leg_shape);
    }
    // Last leg, no loop
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{2U}, stop_idx_t{3U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {3.0F, 3.0F},
                    {3.5F, 3.0F},
                    {4.0F, 4.0F},
                }),
                leg_shape);
    }
    // Loop on start and end
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{2U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {2.0F, 2.0F},
                    {2.0F, 2.5F},
                    {1.5F, 2.0F},
                    {2.0F, 2.0F},
                    {3.0F, 2.0F},
                    {3.0F, 3.0F},
                    {3.0F, 3.5F},
                    {2.5F, 3.0F},
                    {3.0F, 3.0F},
                }),
                leg_shape);
    }
  }
  // Distance traveled available for shape but not on trip
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_5+");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // Loop on start and end
    // Match first stop each loop
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{2U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {2.0F, 2.0F},
                    {2.0F, 2.5F},
                    {1.5F, 2.0F},
                    {2.0F, 2.0F},
                    {3.0F, 2.0F},
                    {3.0F, 3.0F},
                }),
                leg_shape);
    }
  }
  // Trip with not exactly matching distances traveled
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_11");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // M -> N
    // For M: shape < stop_times => shape
    // For N: stop_times < shape => shape
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{2U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {2.0F, 2.0F},
                    {2.5F, 2.5F},
                    {3.0F, 3.0F},
                }),
                leg_shape);
    }
  }
  // Trip with multiple leading 0.0 distances
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_12");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // A -> M
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{0U}, stop_idx_t{1U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {1.0F, 1.0F},
                    {1.5F, 1.5F},
                    {2.0F, 2.0F},
                }),
                leg_shape);
    }
  }
  // Segments must always contain at least 2 points
  {
    // Create run
    transit_realtime::TripDescriptor td;
    td.set_trip_id("TRIP_13");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    // Create full run
    auto const full_run = rt::frun{tt, &rtt, r};

    // Two stops close to same shape point (1.5, 1.5)
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{1U}, stop_idx_t{2U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {2.0F, 2.0F},
                    {2.5F, 2.5F},
                    {3.625F, 3.625F},
                    {3.0F, 3.0F},
                }),
                leg_shape);
    }
    // Duplicated end stop
    {
      leg_shape.clear();

      full_run.for_each_shape_point(
          &shapes_data, interval{stop_idx_t{3U}, stop_idx_t{4U + 1U}},
          plot_point);

      EXPECT_EQ((geo::polyline{
                    {4.0F, 4.0F},
                    {4.0F, 4.0F},
                }),
                leg_shape);
    }
  }
}

}  // namespace