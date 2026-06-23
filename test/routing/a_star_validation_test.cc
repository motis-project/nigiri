#include <chrono>
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/a_star/a_star.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/tb_data.h"

#include "../loader/hrd/hrd_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::routing::tb;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;

timetable load_gtfs(auto const& files) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2021_y / March / 1},
                    date::sys_days{2021_y / March / 8}};
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0}, files(), tt);
  finalize(tt);
  return tt;
}

pareto_set<routing::journey> algo_search(timetable const& tt,
                                         tb_data const& tbd,
                                         routing::query q,
                                         bool is_a_star = true) {
  static auto search_state = routing::search_state{};

  if (is_a_star) {
    auto algo_state = a_star_state{tbd};
    return *(routing::search<direction::kForward, a_star<false>>{
        tt, nullptr, search_state, algo_state, std::move(q)}
                 .execute()
                 .journeys_);
  } else {
    auto trip_based_state = tb::query_state{tt, tbd};
    return *(routing::search<direction::kForward, tb::query_engine<false>>{
        tt, nullptr, search_state, trip_based_state, std::move(q)}
                 .execute()
                 .journeys_);
  }
}

pareto_set<routing::journey> algo_search(timetable const& tt,
                                         tb_data const& tbd,
                                         std::string_view from,
                                         std::string_view to,
                                         routing::start_time_t const time,
                                         float const transfer_factor = 1.0,
                                         bool const is_a_star = true) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .use_start_footpaths_ = true,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                        0_minutes, 0U}},
      .max_transfers_ = 8,
      .transfer_time_settings_ = {.factor_ = transfer_factor}};
  return algo_search(tt, tbd, std::move(q), is_a_star);
}

mem_dir simple_pareto_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2
R2,DTA,R2,R2,"S0 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2
R2,MON,R2_MON,R2_MON,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,01:00:00,01:00:00,S0,0,0,0
R0_MON,01:30:00,01:30:00,S1,1,0,0
R1_MON,02:00:00,02:00:00,S1,0,0,0
R1_MON,02:30:00,02:30:00,S2,1,0,0
R2_MON,01:00:00,01:00:00,S0,0,0,0
R2_MON,03:00:00,03:00:00,S2,1,0,0
)");
}

TEST(a_star_validation, simple_pareto_files) {
  auto const tt = load_gtfs(simple_pareto_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const start = "S0";
  auto const end = "S2";
  auto const transfer_factor = 31.0F;
  auto const time = unixtime_t{sys_days{2021_y / March / 1}};
  auto results_a_star = algo_search(tt, tbd, start, end, time);
  EXPECT_EQ(results_a_star.size(), 1U);
  auto result_without_transfer =
      algo_search(tt, tbd, start, end, time, transfer_factor);
  EXPECT_EQ(result_without_transfer.size(), 1U);
  result_without_transfer.add_not_optimal(results_a_star.els_.at(0));
  auto const tb_results = algo_search(tt, tbd, start, end, time, 1.0F, false);
  EXPECT_EQ(tb_results.size(), 2U);
  for (auto i = 0U; i < result_without_transfer.size(); ++i) {
    EXPECT_TRUE(result_without_transfer.els_.at(i) == tb_results.els_.at(i));
  }
}

mem_dir pareto_files() {
  return mem_dir::read(R"(

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,
S5,S5,,,,,,
S6,S6,,,,,,
S7,S7,,,,,,
S8,S8,,,,,,
S9,S9,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S9",2
R2,DTA,R2,R2,"S0 -> S3",2
R3,DTA,R3,R3,"S4 -> S5",2
R4,DTA,R4,R4,"S5 -> S9",2
R5,DTA,R5,R5,"S2 -> S7",2
R6,DTA,R6,R6,"S4 -> S8",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2
R2,MON,R2_MON,R2_MON,3
R3,MON,R3_MON,R3_MON,4
R4,MON,R4_MON,R4_MON,5
R5,MON,R5_MON,R5_MON,6
R6,MON,R6_MON,R6_MON,7

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,01:00:00,01:00:00,S0,0,0,0
R0_MON,01:30:00,01:30:00,S1,1,0,0
R1_MON,01:35:00,01:35:00,S1,0,0,0
R1_MON,02:30:00,02:30:00,S9,1,0,0
R2_MON,01:00:00,01:00:00,S0,0,0,0
R2_MON,01:15:00,01:15:00,S3,1,0,0
R3_MON,01:30:00,01:30:00,S4,0,0,0
R3_MON,01:40:00,01:40:00,S5,1,0,0
R4_MON,01:45:00,01:45:00,S5,0,0,0
R4_MON,02:01:00,02:01:00,S9,1,0,0
R5_MON,01:10:00,01:10:00,S2,0,0,0
R5_MON,02:00:00,02:00:00,S7,1,0,0
R6_MON,01:20:00,01:20:00,S4,0,0,0
R6_MON,02:10:00,02:10:00,S8,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
S3,S4,2,5
)");
}

TEST(a_star_validation, pareto_files) {
  auto const tt = load_gtfs(pareto_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const start = "S0";
  auto const end = "S9";
  auto const transfer_factor = 30.0F;
  auto const time = unixtime_t{sys_days{2021_y / March / 1}};
  auto results_a_star = algo_search(tt, tbd, start, end, time);
  EXPECT_EQ(results_a_star.size(), 1U);
  auto result_without_transfer =
      algo_search(tt, tbd, start, end, time, transfer_factor);
  EXPECT_EQ(result_without_transfer.size(), 1U);
  result_without_transfer.add_not_optimal(results_a_star.els_.at(0));
  auto const tb_results = algo_search(tt, tbd, start, end, time, 1.0F, false);
  EXPECT_EQ(tb_results.size(), 2U);
  for (auto i = 0U; i < result_without_transfer.size(); ++i) {
    EXPECT_TRUE(result_without_transfer.els_.at(i) == tb_results.els_.at(i));
  }
}