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
#include "nigiri/shape.h"
#include "nigiri/timetable.h"

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

constexpr auto kScheduleWithoutShape = R"(
     "(
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_id
test,https://test.com,Europe/Berlin,DE,0800123456,AGENCY_1

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,1.0,1.0
B,B,2.0,2.0
C,C,3.0,3.0
D,D,4.0,4.0
E,E,5.0,5.0
F,F,6.0,6.0

# calendar_dates.txt
service_id,date,exception_type
SERVICE_1,20240101,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
ROUTE_1,AGENCY_1,Route 1,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,shape_id,
ROUTE_1,SERVICE_1,TRIP_1,E,,SHAPE_1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TRIP_1,10:00:00,10:00:00,A,1,0,0
TRIP_1,11:00:00,11:00:00,B,2,0,0
TRIP_1,12:00:00,12:00:00,C,3,0,0
TRIP_1,13:00:00,13:00:00,D,4,0,0
TRIP_1,14:00:00,14:00:00,E,5,0,0
TRIP_1,15:00:00,15:00:00,F,6,0,0

)"sv;

constexpr auto kShapeWithoutDistances = R"(
# shapes.txt
"shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
SHAPE_1,1.0,1.0,0
SHAPE_1,1.0,1.5,1
SHAPE_1,2.0,2.0,2
SHAPE_1,2.0,2.5,3
SHAPE_1,3.0,3.0,4
SHAPE_1,3.0,3.5,5
SHAPE_1,4.0,4.0,6
SHAPE_1,4.0,4.5,7
SHAPE_1,5.0,5.0,8
SHAPE_1,5.0,5.5,9
SHAPE_1,6.0,6.0,10
)"sv;

// Shapes tested for: B -> C, A -> B, E -> F, B -> E
template <typename Variant>
constexpr void parametrized_test(std::string const& schedule_data,
                                 shapes_storage* shapes_data,
                                 std::array<Variant, 4> test_cases) {
  auto const schedule = mem_dir::read(schedule_data);
  // Load static timetable.
  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / January / 1},
                    date::sys_days{2024_y / January / 2}};
  load_timetable({}, source_idx_t{0}, schedule, tt, nullptr, shapes_data);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / January / 1});

  // Create run
  transit_realtime::TripDescriptor td;
  td.set_start_date("20240101");
  td.set_trip_id("TRIP_1");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  // Create full run
  auto const full_run = rt::frun{tt, &rtt, r};

  // B -> C
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{1}, stop_idx_t{2}});

    ASSERT_TRUE(std::holds_alternative<Variant>(shape));
    EXPECT_EQ(test_cases[0], std::get<Variant>(shape));
  }
  // A -> B
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{0}, stop_idx_t{1}});

    ASSERT_TRUE(std::holds_alternative<Variant>(shape));
    EXPECT_EQ(test_cases[1], std::get<Variant>(shape));
  }
  // E -> F
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{4}, stop_idx_t{5}});

    ASSERT_TRUE(std::holds_alternative<Variant>(shape));
    EXPECT_EQ(test_cases[2], std::get<Variant>(shape));
  }
  // B -> E
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{1}, stop_idx_t{4}});

    ASSERT_TRUE(std::holds_alternative<Variant>(shape));
    EXPECT_EQ(test_cases[3], std::get<Variant>(shape));
  }
}

TEST(
    rt,
    frun_get_shape_when_no_shapes_and_storage_is_used_then_get_owning_array_variant) {
  auto const schedule_data = std::string{kScheduleWithoutShape};
  parametrized_test<std::array<geo::latlng const, 2>>(
      schedule_data, nullptr,
      {{
          {geo::latlng{2.0F, 2.0F}, geo::latlng{3.0F, 3.0F}},
          {geo::latlng{1.0F, 1.0F}, geo::latlng{2.0F, 2.0F}},
          {geo::latlng{5.0F, 5.0F}, geo::latlng{6.0F, 6.0F}},
          {geo::latlng{2.0F, 2.0F}, geo::latlng{5.0F, 5.0F}},
      }});
}

TEST(rt, frun_get_shape_when_no_storage_is_used_then_get_owning_array_variant) {
  auto const schedule_data =
      std::string{kScheduleWithoutShape} + std::string{kShapeWithoutDistances};
  parametrized_test<std::array<geo::latlng const, 2>>(
      schedule_data, nullptr,
      {{
          {geo::latlng{2.0F, 2.0F}, geo::latlng{3.0F, 3.0F}},
          {geo::latlng{1.0F, 1.0F}, geo::latlng{2.0F, 2.0F}},
          {geo::latlng{5.0F, 5.0F}, geo::latlng{6.0F, 6.0F}},
          {geo::latlng{2.0F, 2.0F}, geo::latlng{5.0F, 5.0F}},
      }});
}

TEST(
    rt,
    frun_get_shape_when_shapes_without_distances_are_used_then_get_correct_span) {
  auto shapes_data =
      create_tmp_shapes_storage("rfun-get-shape-without-distances");
  auto const schedule_data =
      std::string{kScheduleWithoutShape} + std::string{kShapeWithoutDistances};
  auto const expected_shapes = std::vector<std::vector<geo::latlng>>{
      {{geo::latlng{2.0F, 2.0F}, geo::latlng{2.0F, 2.5F},
        geo::latlng{3.0F, 3.0F}}},
      {{geo::latlng{1.0F, 1.0F}, geo::latlng{1.0F, 1.5F},
        geo::latlng{2.0F, 2.0F}}},
      {{geo::latlng{5.0F, 5.0F}, geo::latlng{5.0F, 5.5F},
        geo::latlng{6.0F, 6.0F}}},
      {{geo::latlng{2.0F, 2.0F}, geo::latlng{2.0F, 2.5F},
        geo::latlng{3.0F, 3.0F}, geo::latlng{3.0F, 3.5F},
        geo::latlng{4.0F, 4.0F}, geo::latlng{4.0F, 4.5F},
        geo::latlng{5.0F, 5.0F}}},
  };
  parametrized_test<std::span<geo::latlng const>>(
      schedule_data, &shapes_data,
      {{
          {begin(expected_shapes[0]), end(expected_shapes[0])},
          {begin(expected_shapes[1]), end(expected_shapes[1])},
          {begin(expected_shapes[2]), end(expected_shapes[2])},
          {begin(expected_shapes[3]), end(expected_shapes[3])},
      }});
}

TEST(
    rt,
    frun_get_shape_when_shapes_with_shared_stops_are_used_then_get_correct_span_even_when_last_two_stops_fall_together) {
  constexpr auto kShapeWithSharedStops = R"(
# shapes.txt
"shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
SHAPE_1,1.0,1.0,0
SHAPE_1,2.5,2.5,3
SHAPE_1,6.0,6.0,10
)"sv;
  auto shapes_data =
      create_tmp_shapes_storage("rfun-get-shape-with-shared-stops");
  auto const schedule_data =
      std::string{kScheduleWithoutShape} + std::string{kShapeWithSharedStops};
  auto const expected_shapes = std::vector<std::vector<geo::latlng>>{
      {{geo::latlng{2.5F, 2.5F}}},
      {{geo::latlng{1.0F, 1.0F}, geo::latlng{2.5F, 2.5F}}},
      {{geo::latlng{6.0F, 6.0F}}},
      {{geo::latlng{2.5F, 2.5F}, geo::latlng{6.0F, 6.0F}}},
  };
  parametrized_test<std::span<geo::latlng const>>(
      schedule_data, &shapes_data,
      {{
          {begin(expected_shapes[0]), end(expected_shapes[0])},
          {begin(expected_shapes[1]), end(expected_shapes[1])},
          {begin(expected_shapes[2]), end(expected_shapes[2])},
          {begin(expected_shapes[3]), end(expected_shapes[3])},
      }});
}

}  // namespace