#include "gtest/gtest.h"

#include <array>
#include <variant>

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
route_id,service_id,trip_id,trip_headsign,block_id,
ROUTE_1,SERVICE_1,TRIP_1,E,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TRIP_1,10:00:00,10:00:00,A,1,0,0
TRIP_1,11:00:00,11:00:00,B,2,0,0
TRIP_1,12:00:00,12:00:00,C,3,0,0
TRIP_1,13:00:00,13:00:00,D,4,0,0
TRIP_1,14:00:00,14:00:00,E,5,0,0
TRIP_1,15:00:00,15:00:00,F,6,0,0
)";

TEST(rt, frun_get_shape_when_no_storage_is_used_then_get_owning_array_variant) {
  auto const data = mem_dir::read(kScheduleWithoutShape);
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / January / 1},
                    date::sys_days{2024_y / January / 2}};
  auto shapes_data = shapes_storage{};
  load_timetable({}, source_idx_t{0}, data, tt, shapes_data);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / January / 1});

  transit_realtime::TripDescriptor td;
  td.set_start_date("20240101");
  td.set_trip_id("TRIP_1");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / January / 1}, tt, rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const full_run = rt::frun{tt, &rtt, r};

  // B -> C
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{1}, stop_idx_t{2}});

    ASSERT_TRUE(
        (std::holds_alternative<std::array<geo::latlng const, 2>>(shape)));
    EXPECT_EQ((std::array<geo::latlng const, 2>{geo::latlng{2.0f, 2.0f},
                                                geo::latlng{3.0f, 3.0f}}),
              (std::get<std::array<geo::latlng const, 2>>(shape)));
  }
  // A -> B
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{0}, stop_idx_t{1}});

    ASSERT_TRUE(
        (std::holds_alternative<std::array<geo::latlng const, 2>>(shape)));
    EXPECT_EQ((std::array<geo::latlng const, 2>{geo::latlng{1.0f, 1.0f},
                                                geo::latlng{2.0f, 2.0f}}),
              (std::get<std::array<geo::latlng const, 2>>(shape)));
  }
  // E -> F
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{4}, stop_idx_t{5}});

    ASSERT_TRUE(
        (std::holds_alternative<std::array<geo::latlng const, 2>>(shape)));
    EXPECT_EQ((std::array<geo::latlng const, 2>{geo::latlng{5.0f, 5.0f},
                                                geo::latlng{6.0f, 6.0f}}),
              (std::get<std::array<geo::latlng const, 2>>(shape)));
  }
  // B -> E
  {
    auto const shape =
        full_run.get_shape(shapes_data, interval{stop_idx_t{1}, stop_idx_t{4}});

    ASSERT_TRUE(
        (std::holds_alternative<std::array<geo::latlng const, 2>>(shape)));
    EXPECT_EQ((std::array<geo::latlng const, 2>{geo::latlng{2.0f, 2.0f},
                                                geo::latlng{5.0f, 5.0f}}),
              (std::get<std::array<geo::latlng const, 2>>(shape)));
  }
}

}  // namespace