#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

#include "nigiri/rt/vdv/vdv_update.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;

namespace {

mem_dir vdv_test_files() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240710,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,,,,,
B,B,,,,,,
C,C,,,,,,
D,D,,,,,,
E,E,,,,,,


# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AE,MTA,AE,AE,A -> E,0
BC,MTA,BC,BC,B -> C,0
BD,MTA,BD,BD,B -> D,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AE,D,AE_TRIP,AE_TRIP,1
BC,D,BC_TRIP,BC_TRIP,2
BD,D,BD_TRIP,BD_TRIP,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AE_TRIP,00:00,00:00,A,0,0,0
AE_TRIP,01:00,01:00,B,1,0,0
AE_TRIP,02:00,02:00,C,2,0,0
AE_TRIP,03:00,03:00,D,3,0,0
AE_TRIP,04:00,04:00,E,4,0,0
BC_TRIP,01:00,01:00,B,0,0,0
BC_TRIP,02:00,02:00,C,1,0,0
BD_TRIP,01:00,01:00,B,0,0,0
BD_TRIP,02:00,02:00,C,1,0,0
BD_TRIP,03:00,03:00,D,2,0,0

)__");
}

auto const abcde_run =
    vdv_run{.t_ = date::sys_days{2024_y / July / 10} + 2_minutes,
            .route_id_ = "AE",
            .route_text_ = "AE",
            .direction_id_ = "1",
            .direction_text_ = "1",
            .vehicle_ = "vehicle_str",
            .trip_ref_ = "tripref_str",
            .operator_ = "operator_str",
            .date_ = date::sys_days{2024_y / July / 10},
            .complete_ = true,
            .canceled_ = false,
            .additional_run_ = false,
            .stops_ = std::vector<vdv_stop>{
                vdv_stop{.stop_id_ = "A",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = std::nullopt,
                         .t_dep_ = date::sys_days{2024_y / July / 9} + 22_hours,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "B",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 9} + 23_hours,
                         .t_dep_ = date::sys_days{2024_y / July / 9} + 23_hours,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "C",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 10},
                         .t_dep_ = date::sys_days{2024_y / July / 10},
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "D",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 10} + 1_hours,
                         .t_dep_ = date::sys_days{2024_y / July / 10} + 1_hours,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "E",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 10} + 2_hours,
                         .t_dep_ = std::nullopt,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt}}};

auto const bcd_run =
    vdv_run{.t_ = date::sys_days{2024_y / July / 10} + 2_minutes,
            .route_id_ = "AE",
            .route_text_ = "AE",
            .direction_id_ = "1",
            .direction_text_ = "1",
            .vehicle_ = "Train",
            .trip_ref_ = "tripref_string",
            .operator_ = "operator_string",
            .date_ = date::sys_days{2024_y / July / 10},
            .complete_ = false,
            .canceled_ = false,
            .additional_run_ = false,
            .stops_ = std::vector<vdv_stop>{
                vdv_stop{.stop_id_ = "B",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 9} + 23_hours,
                         .t_dep_ = date::sys_days{2024_y / July / 9} + 23_hours,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "C",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 10},
                         .t_dep_ = date::sys_days{2024_y / July / 10},
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "D",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 10} + 1_hours,
                         .t_dep_ = date::sys_days{2024_y / July / 10} + 1_hours,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt}}};

auto const bc_run =
    vdv_run{.t_ = date::sys_days{2024_y / July / 10} + 2_minutes,
            .route_id_ = "AE",
            .route_text_ = "AE",
            .direction_id_ = "1",
            .direction_text_ = "1",
            .vehicle_ = "Train",
            .trip_ref_ = "tripref_string",
            .operator_ = "operator_string",
            .date_ = date::sys_days{2024_y / July / 10},
            .complete_ = false,
            .canceled_ = false,
            .additional_run_ = false,
            .stops_ = std::vector<vdv_stop>{
                vdv_stop{.stop_id_ = "B",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 9} + 23_hours,
                         .t_dep_ = date::sys_days{2024_y / July / 9} + 23_hours,
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt},
                vdv_stop{.stop_id_ = "C",
                         .platform_arr_ = std::nullopt,
                         .platform_dep_ = std::nullopt,
                         .t_arr_ = date::sys_days{2024_y / July / 10},
                         .t_dep_ = date::sys_days{2024_y / July / 10},
                         .t_arr_rt_ = std::nullopt,
                         .t_dep_rt_ = std::nullopt,
                         .in_allowed_ = std::nullopt,
                         .out_allowed_ = std::nullopt,
                         .additional_stop_ = std::nullopt}}};
}  // namespace

TEST(vdv, match_location) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, vdv_test_files(), tt);
  finalize(tt);

  auto const match_a = match_location(tt, "A");
  ASSERT_TRUE(match_a.has_value());
  EXPECT_EQ(location_idx_t{special_stations_names.size()}, match_a.value());

  auto const match_e = match_location(tt, "E");
  ASSERT_TRUE(match_e.has_value());
  EXPECT_EQ(location_idx_t{special_stations_names.size() + 4}, match_e.value());
}

TEST(vdv, match_time) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / July / 1},
                    date::sys_days{2024_y / July / 31}};
  load_timetable({}, source_idx_t{0}, vdv_test_files(), tt);
  finalize(tt);

  auto matches = hash_set<transport>{};
  match_time<event_type::kArr>(
      tt, location_idx_t{special_stations_names.size()},
      unixtime_t{date::sys_days{2024_y / July / 10} - 2_hours}, matches);
  EXPECT_TRUE(matches.empty());

  matches.clear();
  match_time<event_type::kDep>(
      tt, location_idx_t{special_stations_names.size()},
      unixtime_t{date::sys_days{2024_y / July / 10} - 2_hours - 3_minutes},
      matches);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));

  matches.clear();
  match_time<event_type::kArr>(
      tt, location_idx_t{special_stations_names.size() + 1},
      unixtime_t{date::sys_days{2024_y / July / 10} - 1_hours + 2_minutes},
      matches);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));

  matches.clear();
  match_time<event_type::kDep>(
      tt, location_idx_t{special_stations_names.size() + 1},
      unixtime_t{date::sys_days{2024_y / July / 10} - 1_hours - 1_minutes},
      matches);
  EXPECT_EQ(matches.size(), 3);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));
  EXPECT_TRUE(matches.contains({transport_idx_t{1}, day_idx_t{13}}));
  EXPECT_TRUE(matches.contains({transport_idx_t{2}, day_idx_t{13}}));

  matches.clear();
  match_time<event_type::kArr>(
      tt, location_idx_t{special_stations_names.size() + 2},
      unixtime_t{date::sys_days{2024_y / July / 10} - 3_minutes}, matches);
  EXPECT_EQ(matches.size(), 3);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));
  EXPECT_TRUE(matches.contains({transport_idx_t{1}, day_idx_t{13}}));
  EXPECT_TRUE(matches.contains({transport_idx_t{2}, day_idx_t{13}}));

  matches.clear();
  match_time<event_type::kDep>(
      tt, location_idx_t{special_stations_names.size() + 2},
      unixtime_t{date::sys_days{2024_y / July / 10} + 2_minutes}, matches);
  EXPECT_EQ(matches.size(), 2);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));
  EXPECT_TRUE(matches.contains({transport_idx_t{2}, day_idx_t{13}}));

  matches.clear();
  match_time<event_type::kArr>(
      tt, location_idx_t{special_stations_names.size() + 3},
      unixtime_t{date::sys_days{2024_y / July / 10} + 1_hours + 3_minutes},
      matches);
  EXPECT_EQ(matches.size(), 2);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));
  EXPECT_TRUE(matches.contains({transport_idx_t{2}, day_idx_t{13}}));

  matches.clear();
  match_time<event_type::kArr>(
      tt, location_idx_t{special_stations_names.size() + 4},
      unixtime_t{date::sys_days{2024_y / July / 10} + 2_hours}, matches);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_TRUE(matches.contains({transport_idx_t{0}, day_idx_t{13}}));
}

TEST(vdv, vdv_resolve_run) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / July / 1},
                    date::sys_days{2024_y / July / 31}};
  load_timetable({}, source_idx_t{0}, vdv_test_files(), tt);
  finalize(tt);

  auto transport_matches = match_transport(tt, abcde_run);
  EXPECT_EQ(transport_matches.size(), 1);
  EXPECT_TRUE(transport_matches.contains({transport_idx_t{0}, day_idx_t{13}}));

  transport_matches.clear();
  transport_matches = match_transport(tt, bcd_run);
  EXPECT_EQ(transport_matches.size(), 1);
  EXPECT_TRUE(transport_matches.contains({transport_idx_t{0}, day_idx_t{13}}));

  transport_matches.clear();
  transport_matches = match_transport(tt, bc_run);
  EXPECT_EQ(transport_matches.size(), 2);
  EXPECT_TRUE(transport_matches.contains({transport_idx_t{0}, day_idx_t{13}}));
  EXPECT_TRUE(transport_matches.contains({transport_idx_t{2}, day_idx_t{13}}));
}