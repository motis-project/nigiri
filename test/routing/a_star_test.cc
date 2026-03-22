#include <chrono>
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/a_star/a_star.h"
#include "nigiri/routing/a_star/a_star_search.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/tb_data.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::routing::tb;
using namespace nigiri::loader;

timetable load_gtfs(auto const& files) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2021_y / March / 1},
                    date::sys_days{2021_y / March / 8}};
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0}, files(), tt);
  finalize(tt);
  return tt;
}

timetable load_hrd(auto const& files) {
  timetable tt;
  tt.date_range_ = nigiri::test_data::hrd_timetable::full_period();
  register_special_stations(tt);
  hrd::load_timetable(source_idx_t{0U}, loader::hrd::hrd_5_20_26, files(), tt);
  finalize(tt);
  return tt;
}

std::string results_str_as(auto const& results, timetable const& tt) {
  std::stringstream ss;
  for (auto const& r : results) {
    ss << "\n";
    r.print(ss, tt);
    ss << "\n";
  }
  return ss.str();
}

pareto_set<routing::journey> a_star_search(timetable const& tt,
                                           tb_data const& tbd,
                                           routing::query q) {
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};

  return *(routing::a_star_search(tt, search_state, algo_state, std::move(q))
               .journeys_);
}

pareto_set<routing::journey> a_star_search(timetable const& tt,
                                           tb_data const& tbd,
                                           std::string_view from,
                                           std::string_view to,
                                           routing::start_time_t const time) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .use_start_footpaths_ = true,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                        0_minutes, 0U}},
      .max_transfers_ = 8};
  return a_star_search(tt, tbd, std::move(q));
}

pareto_set<routing::journey> a_star_intermodal_search(
    timetable const& tt,
    tb::tb_data const& tbd,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    routing::start_time_t const time,
    std::uint8_t const min_connection_count = 0U,
    bool const extend_interval_earlier = false,
    bool const extend_interval_later = false) {
  auto q = routing::query{
      .start_time_ = time,
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .start_ = std::move(start),
      .destination_ = std::move(destination),
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later};
  return a_star_search(tt, tbd, std::move(q));
}

mem_dir one_run_journey_files() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2
R2,DATA,R2,R2,"S0 -> S2",2
R3,DTA,R3,R3,"S3 -> S0 -> S2 -> S1",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1
R1,TUE,R1_TUE,R1_TUE,2
R2,TUE,R2_TUE,R2_TUE,3
R3,TUE,R3_TUE,R3_TUE,4
R0,TUE,R0B_TUE,R0B_TUE,5

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,05:00:00,05:00:00,S0,0,0,0
R0_TUE,06:00:00,06:00:00,S1,1,0,0
R1_TUE,12:00:00,12:00:00,S1,0,0,0
R1_TUE,13:00:00,13:00:00,S2,1,0,0
R2_TUE,01:00:00,01:00:00,S0,0,0,0
R2_TUE,11:00:00,11:00:00,S2,1,0,0
R3_TUE,04:30:00,04:30:00,S3,0,0,0
R3_TUE,05:31:00,05:31:00,S0,1,0,0
R3_TUE,07:00:00,07:00:00,S2,2,0,0
R3_TUE,13:00:00,13:00:00,S1,3,0,0
R0B_TUE,06:30:00,06:30:00,S0,0,0,0
R0B_TUE,06:00:00,06:00:00,S1,1,0,0
)");
}

constexpr auto const one_run_journey = R"(
[2021-03-02 00:00, 2021-03-02 07:00]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-02 05:31]
       TO: (S2, S2) [2021-03-02 07:00]
leg 0: (S0, S0) [2021-03-02 05:31] -> (S2, S2) [2021-03-02 07:00]
   1: S0      S0..............................................                               d: 02.03 05:31 [02.03 05:31]  [{name=R3, day=2021-03-02, id=R3_TUE, src=0}]
   2: S2      S2.............................................. a: 02.03 07:00 [02.03 07:00]
leg 1: (S2, S2) [2021-03-02 07:00] -> (S2, S2) [2021-03-02 07:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, one_run_journey) {
  auto const tt = load_gtfs(one_run_journey_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S2",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(one_run_journey, results_str_as(results, tt));
}

mem_dir multiple_segment_run_files() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R3,R3,"S0 -> S3 -> S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,04:30:00,04:30:00,S0,0,0,0
R0_MON,05:30:00,05:30:00,S3,1,0,0
R0_MON,13:00:00,13:00:00,S1,2,0,0
R0_MON,14:00:00,14:00:00,S2,3,0,0
)");
}

constexpr auto const multiple_segment_run_journey = R"(
[2021-03-02 00:00, 2021-03-02 14:00]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-02 04:30]
       TO: (S2, S2) [2021-03-02 14:00]
leg 0: (S0, S0) [2021-03-02 04:30] -> (S2, S2) [2021-03-02 14:00]
   0: S0      S0..............................................                               d: 02.03 04:30 [02.03 04:30]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   1: S3      S3.............................................. a: 02.03 05:30 [02.03 05:30]  d: 02.03 05:30 [02.03 05:30]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   2: S1      S1.............................................. a: 02.03 13:00 [02.03 13:00]  d: 02.03 13:00 [02.03 13:00]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   3: S2      S2.............................................. a: 02.03 14:00 [02.03 14:00]
leg 1: (S2, S2) [2021-03-02 14:00] -> (S2, S2) [2021-03-02 14:00]
  FOOTPATH (duration=0)

)";
TEST(a_star, multiple_segment_run) {
  auto const tt = load_gtfs(multiple_segment_run_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S2",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(multiple_segment_run_journey, results_str_as(results, tt));
}

TEST(a_star, too_long_journey) {
  auto const tt = load_gtfs(multiple_segment_run_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 01 / 2021} + 14_hours});
  EXPECT_EQ(results.size(), 0U);
}

TEST(a_star, start_segments_too_late) {
  auto const tt = load_gtfs(multiple_segment_run_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2",
      unixtime_t{sys_days{March / 01 / 2021} + 4_hours + 30_minutes});
  EXPECT_EQ(results.size(), 0U);
}

mem_dir footpaths_before_and_after_files() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R3,R3,"S1 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,06:00:00,06:00:00,S1,0,0,0
R0_MON,07:00:00,07:00:00,S3,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
S0,S1,2,900
S3,S2,2,600
)");
}

constexpr auto const footpaths_before_and_after_journey = R"(
[2021-03-02 00:00, 2021-03-02 07:10]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-02 05:45]
       TO: (S2, S2) [2021-03-02 07:10]
leg 0: (S0, S0) [2021-03-02 05:45] -> (S1, S1) [2021-03-02 06:00]
  FOOTPATH (duration=15)
leg 1: (S1, S1) [2021-03-02 06:00] -> (S3, S3) [2021-03-02 07:00]
   0: S1      S1..............................................                               d: 02.03 06:00 [02.03 06:00]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   1: S3      S3.............................................. a: 02.03 07:00 [02.03 07:00]
leg 2: (S3, S3) [2021-03-02 07:00] -> (S2, S2) [2021-03-02 07:10]
  FOOTPATH (duration=10)

)";

TEST(a_star, footpaths_before_and_after) {
  auto const tt = load_gtfs(footpaths_before_and_after_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S2",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(footpaths_before_and_after_journey, results_str_as(results, tt));
}

mem_dir two_dest_segments_reached_files() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R3,R3,"S0 -> S3 -> S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,04:30:00,04:30:00,S0,0,0,0
R0_MON,04:35:00,04:35:00,S3,1,0,0
R0_MON,04:40:00,04:40:00,S1,2,0,0
R0_MON,04:45:00,04:45:00,S2,3,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
S3,S2,2,900
)");
}

mem_dir midnight_cross_files() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R3,R3,"S0 -> S3 -> S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,23:00:00,23:00:00,S0,0,0,0
R0_TUE,23:30:00,23:30:00,S3,1,0,0
R0_TUE,24:00:00,24:00:00,S1,2,0,0
R0_TUE,25:00:00,25:00:00,S2,3,0,0
)");
}

constexpr auto const midnight_cross_journey = R"(
[2021-03-02 20:00, 2021-03-03 01:00]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-02 23:00]
       TO: (S2, S2) [2021-03-03 01:00]
leg 0: (S0, S0) [2021-03-02 23:00] -> (S2, S2) [2021-03-03 01:00]
   0: S0      S0..............................................                               d: 02.03 23:00 [02.03 23:00]  [{name=R3, day=2021-03-02, id=R0_TUE, src=0}]
   1: S3      S3.............................................. a: 02.03 23:30 [02.03 23:30]  d: 02.03 23:30 [02.03 23:30]  [{name=R3, day=2021-03-02, id=R0_TUE, src=0}]
   2: S1      S1.............................................. a: 03.03 00:00 [03.03 00:00]  d: 03.03 00:00 [03.03 00:00]  [{name=R3, day=2021-03-02, id=R0_TUE, src=0}]
   3: S2      S2.............................................. a: 03.03 01:00 [03.03 01:00]
leg 1: (S2, S2) [2021-03-03 01:00] -> (S2, S2) [2021-03-03 01:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, midnight_cross) {
  auto const tt = load_gtfs(midnight_cross_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 02 / 2021}} + 20_hours);
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(midnight_cross_journey, results_str_as(results, tt));
}

constexpr auto const two_dest_segments_reached_journey = R"(
[2021-03-02 00:00, 2021-03-02 04:45]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-02 04:30]
       TO: (S2, S2) [2021-03-02 04:45]
leg 0: (S0, S0) [2021-03-02 04:30] -> (S2, S2) [2021-03-02 04:45]
   0: S0      S0..............................................                               d: 02.03 04:30 [02.03 04:30]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   1: S3      S3.............................................. a: 02.03 04:35 [02.03 04:35]  d: 02.03 04:35 [02.03 04:35]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   2: S1      S1.............................................. a: 02.03 04:40 [02.03 04:40]  d: 02.03 04:40 [02.03 04:40]  [{name=R3, day=2021-03-02, id=R0_MON, src=0}]
   3: S2      S2.............................................. a: 02.03 04:45 [02.03 04:45]
leg 1: (S2, S2) [2021-03-02 04:45] -> (S2, S2) [2021-03-02 04:45]
  FOOTPATH (duration=0)

)";

TEST(a_star, two_dest_segments_reached) {
  auto const tt = load_gtfs(two_dest_segments_reached_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S2",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(two_dest_segments_reached_journey, results_str_as(results, tt));
}

mem_dir same_day_transfer_files_as() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1
R1,TUE,R1_TUE,R1_TUE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,00:00:00,00:00:00,S0,0,0,0
R0_TUE,06:00:00,06:00:00,S1,1,0,0
R1_TUE,12:00:00,12:00:00,S1,0,0,0
R1_TUE,13:00:00,13:00:00,S2,1,0,0
)");
}

constexpr auto const same_day_transfer_journey = R"(
[2021-03-02 00:00, 2021-03-02 13:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-02 00:00]
       TO: (S2, S2) [2021-03-02 13:00]
leg 0: (S0, S0) [2021-03-02 00:00] -> (S1, S1) [2021-03-02 06:00]
   0: S0      S0..............................................                               d: 02.03 00:00 [02.03 00:00]  [{name=R0, day=2021-03-02, id=R0_TUE, src=0}]
   1: S1      S1.............................................. a: 02.03 06:00 [02.03 06:00]
leg 1: (S1, S1) [2021-03-02 06:00] -> (S1, S1) [2021-03-02 06:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-02 12:00] -> (S2, S2) [2021-03-02 13:00]
   0: S1      S1..............................................                               d: 02.03 12:00 [02.03 12:00]  [{name=R1, day=2021-03-02, id=R1_TUE, src=0}]
   1: S2      S2.............................................. a: 02.03 13:00 [02.03 13:00]
leg 3: (S2, S2) [2021-03-02 13:00] -> (S2, S2) [2021-03-02 13:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, same_day_transfer) {
  auto const tt = load_gtfs(same_day_transfer_files_as);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S2",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(same_day_transfer_journey, results_str_as(results, tt));
}

mem_dir next_day_transfer_files_as() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1
R1,WED,R1_WED,R1_WED,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,12:00:00,12:00:00,S0,0,0,0
R0_TUE,23:00:00,23:00:00,S1,1,0,0
R1_WED,06:00:00,06:00:00,S1,0,0,0
R1_WED,08:00:00,08:00:00,S2,1,0,0
)");
}

constexpr auto const next_day_transfer_journey = R"(
[2021-03-02 11:00, 2021-03-03 08:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-02 12:00]
       TO: (S2, S2) [2021-03-03 08:00]
leg 0: (S0, S0) [2021-03-02 12:00] -> (S1, S1) [2021-03-02 23:00]
   0: S0      S0..............................................                               d: 02.03 12:00 [02.03 12:00]  [{name=R0, day=2021-03-02, id=R0_TUE, src=0}]
   1: S1      S1.............................................. a: 02.03 23:00 [02.03 23:00]
leg 1: (S1, S1) [2021-03-02 23:00] -> (S1, S1) [2021-03-02 23:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-03 06:00] -> (S2, S2) [2021-03-03 08:00]
   0: S1      S1..............................................                               d: 03.03 06:00 [03.03 06:00]  [{name=R1, day=2021-03-03, id=R1_WED, src=0}]
   1: S2      S2.............................................. a: 03.03 08:00 [03.03 08:00]
leg 3: (S2, S2) [2021-03-03 08:00] -> (S2, S2) [2021-03-03 08:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, next_day_transfer) {
  auto const tt = load_gtfs(next_day_transfer_files_as);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 02 / 2021} + 11_hours});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(next_day_transfer_journey, results_str_as(results, tt));
}
mem_dir transfer_to_journey_from_previous_day_files() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S3 -> S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,TUE,R1_TUE,R1_TUE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,22:00:00,22:00:00,S3,0,0,0
R0_MON,30:00:00,30:00:00,S1,1,0,0
R0_MON,31:00:00,31:00:00,S2,2,0,0
R1_TUE,02:00:00,02:00:00,S0,0,0,0
R1_TUE,03:00:00,03:00:00,S1,1,0,0
)");
}

constexpr auto const transfer_to_journey_from_previous_day_journey = R"(
[2021-03-02 01:00, 2021-03-02 07:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-02 02:00]
       TO: (S2, S2) [2021-03-02 07:00]
leg 0: (S0, S0) [2021-03-02 02:00] -> (S1, S1) [2021-03-02 03:00]
   0: S0      S0..............................................                               d: 02.03 02:00 [02.03 02:00]  [{name=R1, day=2021-03-02, id=R1_TUE, src=0}]
   1: S1      S1.............................................. a: 02.03 03:00 [02.03 03:00]
leg 1: (S1, S1) [2021-03-02 03:00] -> (S1, S1) [2021-03-02 03:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-02 06:00] -> (S2, S2) [2021-03-02 07:00]
   1: S1      S1..............................................                               d: 02.03 06:00 [02.03 06:00]  [{name=R0, day=2021-03-01, id=R0_MON, src=0}]
   2: S2      S2.............................................. a: 02.03 07:00 [02.03 07:00]
leg 3: (S2, S2) [2021-03-02 07:00] -> (S2, S2) [2021-03-02 07:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, transfer_to_journey_from_previous_day) {
  auto const tt = load_gtfs(transfer_to_journey_from_previous_day_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 02 / 2021} + 1_hours});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(transfer_to_journey_from_previous_day_journey,
            results_str_as(results, tt));
}

mem_dir transfer_on_next_day_files_as() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1
R1,WED,R1_WED,R1_WED,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,12:00:00,12:00:00,S0,0,0,0
R0_TUE,25:00:00,25:00:00,S1,1,0,0
R1_WED,06:00:00,06:00:00,S1,0,0,0
R1_WED,08:00:00,08:00:00,S2,1,0,0
)");
}

constexpr auto const transfer_on_next_day_journey = R"(
[2021-03-02 11:00, 2021-03-03 08:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-02 12:00]
       TO: (S2, S2) [2021-03-03 08:00]
leg 0: (S0, S0) [2021-03-02 12:00] -> (S1, S1) [2021-03-03 01:00]
   0: S0      S0..............................................                               d: 02.03 12:00 [02.03 12:00]  [{name=R0, day=2021-03-02, id=R0_TUE, src=0}]
   1: S1      S1.............................................. a: 03.03 01:00 [03.03 01:00]
leg 1: (S1, S1) [2021-03-03 01:00] -> (S1, S1) [2021-03-03 01:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-03 06:00] -> (S2, S2) [2021-03-03 08:00]
   0: S1      S1..............................................                               d: 03.03 06:00 [03.03 06:00]  [{name=R1, day=2021-03-03, id=R1_WED, src=0}]
   1: S2      S2.............................................. a: 03.03 08:00 [03.03 08:00]
leg 3: (S2, S2) [2021-03-03 08:00] -> (S2, S2) [2021-03-03 08:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, transfer_on_next_day) {
  auto const tt = load_gtfs(transfer_on_next_day_files_as);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 02 / 2021} + 11_hours});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(transfer_on_next_day_journey, results_str_as(results, tt));
}

mem_dir transfer_on_next_day_follow_up_files_as() {
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
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2
R2,DTA,R2,R2,"S2 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1
R1,WED,R1_WED,R1_WED,2
R2,WED,R2_WED,R2_WED,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,12:00:00,12:00:00,S0,0,0,0
R0_TUE,25:00:00,25:00:00,S1,1,0,0
R1_WED,06:00:00,06:00:00,S1,0,0,0
R1_WED,08:00:00,08:00:00,S2,1,0,0
R2_WED,08:30:00,08:30:00,S2,0,0,0
R2_WED,09:00:00,09:00:00,S3,1,0,0
)");
}

constexpr auto const transfer_on_next_day_follow_up_journey = R"(
[2021-03-02 11:00, 2021-03-03 09:00]
TRANSFERS: 2
     FROM: (S0, S0) [2021-03-02 12:00]
       TO: (S3, S3) [2021-03-03 09:00]
leg 0: (S0, S0) [2021-03-02 12:00] -> (S1, S1) [2021-03-03 01:00]
   0: S0      S0..............................................                               d: 02.03 12:00 [02.03 12:00]  [{name=R0, day=2021-03-02, id=R0_TUE, src=0}]
   1: S1      S1.............................................. a: 03.03 01:00 [03.03 01:00]
leg 1: (S1, S1) [2021-03-03 01:00] -> (S1, S1) [2021-03-03 01:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-03 06:00] -> (S2, S2) [2021-03-03 08:00]
   0: S1      S1..............................................                               d: 03.03 06:00 [03.03 06:00]  [{name=R1, day=2021-03-03, id=R1_WED, src=0}]
   1: S2      S2.............................................. a: 03.03 08:00 [03.03 08:00]
leg 3: (S2, S2) [2021-03-03 08:00] -> (S2, S2) [2021-03-03 08:02]
  FOOTPATH (duration=2)
leg 4: (S2, S2) [2021-03-03 08:30] -> (S3, S3) [2021-03-03 09:00]
   0: S2      S2..............................................                               d: 03.03 08:30 [03.03 08:30]  [{name=R2, day=2021-03-03, id=R2_WED, src=0}]
   1: S3      S3.............................................. a: 03.03 09:00 [03.03 09:00]
leg 5: (S3, S3) [2021-03-03 09:00] -> (S3, S3) [2021-03-03 09:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, transfer_on_next_day_follow_up) {
  auto const tt = load_gtfs(transfer_on_next_day_follow_up_files_as);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S3", unixtime_t{sys_days{March / 02 / 2021} + 11_hours});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(transfer_on_next_day_follow_up_journey,
            results_str_as(results, tt));
}

mem_dir transfer_not_active_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
ALL,1,1,1,1,1,1,1,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S0 -> S2",2
R2,DTA,R2,R2,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,ALL,R0_ALL,R0_ALL,1
R1,ALL,R1_ALL,R1_ALL,2
R2,WED,R2_WED,R2_WED,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_ALL,02:00:00,02:00:00,S0,0,0,0
R0_ALL,03:00:00,03:00:00,S1,1,0,0
R1_ALL,06:00:00,06:00:00,S0,0,0,0
R1_ALL,08:00:00,08:00:00,S2,1,0,0
R2_WED,03:30:00,03:30:00,S1,0,0,0
R2_WED,04:00:00,04:00:00,S2,1,0,0
)");
}

constexpr auto const transfer_not_active_journey = R"(
[2021-03-02 00:00, 2021-03-02 08:00]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-02 06:00]
       TO: (S2, S2) [2021-03-02 08:00]
leg 0: (S0, S0) [2021-03-02 06:00] -> (S2, S2) [2021-03-02 08:00]
   0: S0      S0..............................................                               d: 02.03 06:00 [02.03 06:00]  [{name=R1, day=2021-03-02, id=R1_ALL, src=0}]
   1: S2      S2.............................................. a: 02.03 08:00 [02.03 08:00]
leg 1: (S2, S2) [2021-03-02 08:00] -> (S2, S2) [2021-03-02 08:00]
  FOOTPATH (duration=0)

)";

TEST(a_star, transfer_not_active) {
  auto const tt = load_gtfs(transfer_not_active_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S2",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(transfer_not_active_journey, results_str_as(results, tt));
}

mem_dir too_many_transfers_files() {
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
S10,S10,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2
R2,DTA,R2,R2,"S2 -> S3",2
R3,DTA,R3,R3,"S3 -> S4",2
R4,DTA,R4,R4,"S4 -> S5",2
R5,DTA,R5,R5,"S5 -> S6",2
R6,DTA,R6,R6,"S6 -> S7",2
R7,DTA,R7,R7,"S7 -> S8",2
R8,DTA,R8,R8,"S8 -> S9",2
R9,DTA,R9,R9,"S9 -> S10",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,TUE,R0_TUE,R0_TUE,1
R1,TUE,R1_TUE,R1_TUE,2
R2,TUE,R2_TUE,R2_TUE,3
R3,TUE,R3_TUE,R3_TUE,4
R4,TUE,R4_TUE,R4_TUE,5
R5,TUE,R5_TUE,R5_TUE,6
R6,TUE,R6_TUE,R6_TUE,7
R7,TUE,R7_TUE,R7_TUE,8
R8,TUE,R8_TUE,R8_TUE,9
R9,TUE,R9_TUE,R9_TUE,10

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_TUE,01:00:00,01:00:00,S0,0,0,0
R0_TUE,01:30:00,01:30:00,S1,1,0,0
R1_TUE,02:00:00,02:00:00,S1,0,0,0
R1_TUE,02:30:00,02:30:00,S2,1,0,0
R2_TUE,03:00:00,03:00:00,S2,0,0,0
R2_TUE,03:30:00,03:30:00,S3,1,0,0
R3_TUE,04:00:00,04:00:00,S3,0,0,0
R3_TUE,04:30:00,04:30:00,S4,1,0,0
R4_TUE,05:00:00,05:00:00,S4,0,0,0
R4_TUE,05:30:00,05:30:00,S5,1,0,0
R5_TUE,06:00:00,06:00:00,S5,0,0,0
R5_TUE,06:30:00,06:30:00,S6,1,0,0
R6_TUE,07:00:00,07:00:00,S6,0,0,0
R6_TUE,07:30:00,07:30:00,S7,1,0,0
R7_TUE,08:00:00,08:00:00,S7,0,0,0
R7_TUE,08:30:00,08:30:00,S8,1,0,0
R8_TUE,09:00:00,09:00:00,S8,0,0,0
R8_TUE,09:30:00,09:30:00,S9,1,0,0
R9_TUE,10:00:00,10:00:00,S9,0,0,0
R9_TUE,10:30:00,10:30:00,S10,1,0,0
)");
}

TEST(a_star, too_many_transfers) {
  auto const tt = load_gtfs(too_many_transfers_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(tt, tbd, "S0", "S10",
                                     unixtime_t{sys_days{March / 02 / 2021}});
  EXPECT_EQ(results.size(), 0U);
}

constexpr auto const intermodal_abc_journey = R"(
[2020-03-30 05:00, 2020-03-30 08:00]
TRANSFERS: 1
     FROM: (START, START) [2020-03-30 05:20]
       TO: (END, END) [2020-03-30 08:00]
leg 0: (START, START) [2020-03-30 05:20] -> (A, 0000001) [2020-03-30 05:30]
  MUMO (id=99, duration=10)
leg 1: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 2: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 3: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 4: (C, 0000003) [2020-03-30 07:45] -> (END, END) [2020-03-30 08:00]
  MUMO (id=77, duration=15)

)";

TEST(a_star, intermodal_abc) {
  auto const tt = load_hrd(nigiri::test_data::hrd_timetable::files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_intermodal_search(
      tt, tbd,
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000001", .src_ = source_idx_t{0U}}),
        10_minutes, 99U}},
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000003", .src_ = source_idx_t{0U}}),
        15_minutes, 77U}},
      unixtime_t{sys_days{March / 30 / 2020} + 5_hours});
  EXPECT_EQ(results.size(), 1U);
  EXPECT_EQ(std::string_view{intermodal_abc_journey},
            results_str_as(results, tt));
}

constexpr auto const intermodal_abc_journeys = R"(
[2020-03-30 05:20, 2020-03-30 08:00]
TRANSFERS: 1
     FROM: (START, START) [2020-03-30 05:20]
       TO: (END, END) [2020-03-30 08:00]
leg 0: (START, START) [2020-03-30 05:20] -> (A, 0000001) [2020-03-30 05:30]
  MUMO (id=99, duration=10)
leg 1: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 2: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 3: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 4: (C, 0000003) [2020-03-30 07:45] -> (END, END) [2020-03-30 08:00]
  MUMO (id=77, duration=15)


[2020-03-30 05:50, 2020-03-30 08:30]
TRANSFERS: 1
     FROM: (START, START) [2020-03-30 05:50]
       TO: (END, END) [2020-03-30 08:30]
leg 0: (START, START) [2020-03-30 05:50] -> (A, 0000001) [2020-03-30 06:00]
  MUMO (id=99, duration=10)
leg 1: (A, 0000001) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 07:00]
   0: 0000001 A...............................................                               d: 30.03 06:00 [30.03 08:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/360/0000002/420/, src=0}]
   1: 0000002 B............................................... a: 30.03 07:00 [30.03 09:00]
leg 2: (B, 0000002) [2020-03-30 07:00] -> (B, 0000002) [2020-03-30 07:02]
  FOOTPATH (duration=2)
leg 3: (B, 0000002) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 08:15]
   0: 0000002 B...............................................                               d: 30.03 07:15 [30.03 09:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/435/0000003/495/, src=0}]
   1: 0000003 C............................................... a: 30.03 08:15 [30.03 10:15]
leg 4: (C, 0000003) [2020-03-30 08:15] -> (END, END) [2020-03-30 08:30]
  MUMO (id=77, duration=15)

)";

TEST(a_star, intermodal_abc_interval_start_time) {
  auto const tt = load_hrd(nigiri::test_data::hrd_timetable::files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_intermodal_search(
      tt, tbd,
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000001", .src_ = source_idx_t{0U}}),
        10_minutes, 99U}},
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000003", .src_ = source_idx_t{0U}}),
        15_minutes, 77U}},
      interval{unixtime_t{sys_days{March / 30 / 2020}} + 5_hours,
               unixtime_t{sys_days{March / 30 / 2020}} + 6_hours});
  EXPECT_EQ(results.size(), 2U);
  EXPECT_EQ(std::string_view{intermodal_abc_journeys},
            results_str_as(results, tt));
}
