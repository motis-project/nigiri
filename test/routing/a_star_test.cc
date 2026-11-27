#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/a_star/a_star_search.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/query_engine.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::loader;
using namespace std::chrono_literals;
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

timetable load_hrd(auto const& files) {
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  hrd::load_timetable(source_idx_t{0U}, loader::hrd::hrd_5_20_26, files(), tt);
  finalize(tt);
  return tt;
}

std::vector<routing::journey> a_star_search(timetable const& tt,
                                            tb::tb_data const& tbd,
                                            routing::query q) {
  static auto search_state = routing::search_state{};
  auto algo_state = tb::query_state{tt, tbd};
  auto journey_pareto_set =
      (routing::search<direction::kForward, a_star::a_star_search>{
          tt, nullptr, search_state, algo_state, std::move(q)}
           .execute()
           .journeys_);
  return journey_pareto_set->els_;
}

std::vector<routing::journey> a_star_search(timetable const& tt,
                                            tb::tb_data const& tbd,
                                            std::string_view from,
                                            std::string_view to,
                                            routing::start_time_t const time) {
  auto const src = source_idx_t{0};
  return a_star_search(
      tt, tbd,
      routing::query{
          .start_time_ = time,
          .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                            0_minutes, 0U}}});
}

std::vector<routing::journey> a_star_intermodal_search(
    timetable const& tt,
    tb::tb_data const& tbd,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    interval<unixtime_t> interval,
    std::uint8_t const min_connection_count = 0U,
    bool const extend_interval_earlier = false,
    bool const extend_interval_later = false) {
  auto q = routing::query{
      .start_time_ = interval,
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .start_ = std::move(start),
      .destination_ = std::move(destination),
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later};
  return a_star_search(tt, tbd, std::move(q));
}

std::string result_str(std::vector<journey> const& results,
                       timetable const& tt) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n";
  }
  return ss.str();
}

// NOTE: The preprocessing tests from trip-based routing (-> tb_test.cc) are not
// needed, because the same preprocessing data is used here.

// --- single trip ---

mem_dir single_trip_files() {
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
DAILY,1,1,1,1,1,1,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,DAILY,R0_DAILY,R0_DAILY,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_DAILY,07:00:00,07:00:00,S0,0,0,0
R0_DAILY,08:00:00,08:00:00,S1,1,0,0
R0_DAILY,09:00:00,09:00:00,S2,2,0,0
)");
}

constexpr auto const single_trip_journey = R"(
[2021-03-01 06:00, 2021-03-01 09:00]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-01 07:00]
       TO: (S2, S2) [2021-03-01 09:00]
leg 0: (S0, S0) [2021-03-01 07:00] -> (S2, S2) [2021-03-01 09:00]
   0: S0      S0..............................................                               d: 01.03 07:00 [01.03 07:00]  [{name=R0, day=2021-03-01, id=R0_DAILY, src=0}]
   1: S1      S1.............................................. a: 01.03 08:00 [01.03 08:00]  d: 01.03 08:00 [01.03 08:00]  [{name=R0, day=2021-03-01, id=R0_DAILY, src=0}]
   2: S2      S2.............................................. a: 01.03 09:00 [01.03 09:00]
leg 1: (S2, S2) [2021-03-01 09:00] -> (S2, S2) [2021-03-01 09:00]
  FOOTPATH (duration=0)

)";

TEST(a_star_query, single_trip) {
  auto const tt = load_gtfs(single_trip_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  // tbd.print(std::cout, tt);
  // auto [a, b] = tt.day_idx_mam(unixtime_t{i32_minutes{26908200}});
  // std::cout << "DAY: " << a <<    " MAM: " << b << "\n \n";
  auto const result = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 1 / 2021} + 6h});
  EXPECT_EQ(std::string_view{single_trip_journey}, result_str(result, tt));
}

// --- same_day_transfer ---

mem_dir same_day_transfer_files_2() {
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
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,06:00:00,06:00:00,S1,1,0,0
R1_MON,12:00:00,12:00:00,S1,0,0,0
R1_MON,13:00:00,13:00:00,S2,1,0,0
)");
}

constexpr auto const same_day_transfer_journeys = R"(
[2021-02-28 23:00, 2021-03-01 13:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-01 00:00]
       TO: (S2, S2) [2021-03-01 13:00]
leg 0: (S0, S0) [2021-03-01 00:00] -> (S1, S1) [2021-03-01 06:00]
   0: S0      S0..............................................                               d: 01.03 00:00 [01.03 00:00]  [{name=R0, day=2021-03-01, id=R0_MON, src=0}]
   1: S1      S1.............................................. a: 01.03 06:00 [01.03 06:00]
leg 1: (S1, S1) [2021-03-01 06:00] -> (S1, S1) [2021-03-01 06:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-01 12:00] -> (S2, S2) [2021-03-01 13:00]
   0: S1      S1..............................................                               d: 01.03 12:00 [01.03 12:00]  [{name=R1, day=2021-03-01, id=R1_MON, src=0}]
   1: S2      S2.............................................. a: 01.03 13:00 [01.03 13:00]
leg 3: (S2, S2) [2021-03-01 13:00] -> (S2, S2) [2021-03-01 13:00]
  FOOTPATH (duration=0)

)";

TEST(a_star_query, same_day_transfer) {
  auto const tt = load_gtfs(same_day_transfer_files_2);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  // tbd.print(std::cout, tt);
  // auto [a, b] = tt.day_idx_mam(unixtime_t{i32_minutes{26908200}});
  // std::cout << "DAY: " << a <<    " MAM: " << b << "\n \n";
  auto const result = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(std::string_view{same_day_transfer_journeys},
            result_str(result, tt));
}

// --- earlier_stop_transfer ---

mem_dir earlier_stop_transfer_files_2() {
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

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON0,R0_MON0,0
R0,MON,R0_MON1,R0_MON1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON0,00:00:00,00:00:00,S0,0,0,0
R0_MON0,01:00:00,01:00:00,S1,1,0,0
R0_MON0,02:00:00,02:00:00,S2,2,0,0
R0_MON0,03:00:00,03:00:00,S3,3,0,0
R0_MON0,04:00:00,04:00:00,S1,4,0,0
R0_MON0,05:00:00,05:00:00,S4,5,0,0
R0_MON1,04:00:00,04:00:00,S0,0,0,0
R0_MON1,05:00:00,05:00:00,S1,1,0,0
R0_MON1,06:00:00,06:00:00,S2,2,0,0
R0_MON1,07:00:00,07:00:00,S3,3,0,0
R0_MON1,08:00:00,08:00:00,S1,4,0,0
R0_MON1,09:00:00,09:00:00,S4,5,0,0
)");
}

constexpr auto const earlier_stop_transfer_journeys = R"(
[2021-02-28 23:00, 2021-03-01 06:00]
TRANSFERS: 1
     FROM: (S3, S3) [2021-03-01 03:00]
       TO: (S2, S2) [2021-03-01 06:00]
leg 0: (S3, S3) [2021-03-01 03:00] -> (S1, S1) [2021-03-01 04:00]
   3: S3      S3..............................................                               d: 01.03 03:00 [01.03 03:00]  [{name=R0, day=2021-03-01, id=R0_MON0, src=0}]
   4: S1      S1.............................................. a: 01.03 04:00 [01.03 04:00]
leg 1: (S1, S1) [2021-03-01 04:00] -> (S1, S1) [2021-03-01 04:02]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-01 05:00] -> (S2, S2) [2021-03-01 06:00]
   1: S1      S1..............................................                               d: 01.03 05:00 [01.03 05:00]  [{name=R0, day=2021-03-01, id=R0_MON1, src=0}]
   2: S2      S2.............................................. a: 01.03 06:00 [01.03 06:00]
leg 3: (S2, S2) [2021-03-01 06:00] -> (S2, S2) [2021-03-01 06:00]
  FOOTPATH (duration=0)

)";

TEST(a_star_query, early_stop_transfer) {
  auto const tt = load_gtfs(earlier_stop_transfer_files_2);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S3", "S2", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(std::string_view{earlier_stop_transfer_journeys},
            result_str(results, tt));
}

// --- no_journey_possible ---  (same data as for early_stop_transfer!)

TEST(a_star_query, no_journey_possible) {
  auto const tt = load_gtfs(earlier_stop_transfer_files_2);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S4", "S0", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(0, results.size());
}

// --- early_train ---

mem_dir early_train_files_2() {
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
THU,0,0,0,1,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2 -> S3",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,76:00:00,80:00:00,S2,1,0,0
R0_MON,81:00:00,81:00:00,S3,2,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

constexpr auto const early_train_journeys = R"(
[2021-03-04 05:00, 2021-03-04 09:00]
TRANSFERS: 1
     FROM: (S1, S1) [2021-03-04 06:00]
       TO: (S3, S3) [2021-03-04 09:00]
leg 0: (S1, S1) [2021-03-04 06:00] -> (S2, S2) [2021-03-04 07:00]
   0: S1      S1..............................................                               d: 04.03 06:00 [04.03 06:00]  [{name=R1, day=2021-03-04, id=R1_THU, src=0}]
   1: S2      S2.............................................. a: 04.03 07:00 [04.03 07:00]
leg 1: (S2, S2) [2021-03-04 07:00] -> (S2, S2) [2021-03-04 07:02]
  FOOTPATH (duration=2)
leg 2: (S2, S2) [2021-03-04 08:00] -> (S3, S3) [2021-03-04 09:00]
   1: S2      S2..............................................                               d: 04.03 08:00 [04.03 08:00]  [{name=R0, day=2021-03-01, id=R0_MON, src=0}]
   2: S3      S3.............................................. a: 04.03 09:00 [04.03 09:00]
leg 3: (S3, S3) [2021-03-04 09:00] -> (S3, S3) [2021-03-04 09:00]
  FOOTPATH (duration=0)

)";

TEST(a_star_query, early_train) {
  auto const tt = load_gtfs(early_train_files_2);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S1", "S3", unixtime_t{sys_days{March / 04 / 2021} + 5h});
  EXPECT_EQ(std::string_view{early_train_journeys}, result_str(results, tt));
}

// --- multiple_paths ---

mem_dir multiple_paths_files() {
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
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
T0,DTA,T0,T0,"S0 -> S2",2
T1,DTA,T1,T1,"S0 -> S2",2
T2,DTA,T2,T2,"S0 -> S1",2
T3,DTA,T3,T3,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
T0,TUE,T0_TUE,T0_TUE,0
T1,MON,T1_MON,T1_MON,1
T2,MON,T2_MON,T2_MON,2
T3,MON,T3_MON,T3_MON,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T0_TUE,04:00:00,04:00:00,S0,0,0,0
T0_TUE,04:30:00,04:30:00,S2,1,0,0
T1_MON,05:00:00,05:00:00,S0,0,0,0
T1_MON,08:00:00,08:00:00,S2,1,0,0
T2_MON,06:00:00,06:00:00,S0,0,0,0
T2_MON,06:20:00,06:20:00,S1,1,0,0
T3_MON,06:40:00,06:40:00,S1,0,0,0
T3_MON,07:00:00,07:00:00,S2,1,0,0
)");
}

constexpr auto const multiple_paths_journey_lowTime = R"(
[2021-03-01 01:00, 2021-03-01 07:00]
TRANSFERS: 1
     FROM: (S0, S0) [2021-03-01 06:00]
       TO: (S2, S2) [2021-03-01 07:00]
leg 0: (S0, S0) [2021-03-01 06:00] -> (S1, S1) [2021-03-01 06:20]
   0: S0      S0..............................................                               d: 01.03 06:00 [01.03 06:00]  [{name=T2, day=2021-03-01, id=T2_MON, src=0}]
   1: S1      S1.............................................. a: 01.03 06:20 [01.03 06:20]
leg 1: (S1, S1) [2021-03-01 06:20] -> (S1, S1) [2021-03-01 06:22]
  FOOTPATH (duration=2)
leg 2: (S1, S1) [2021-03-01 06:40] -> (S2, S2) [2021-03-01 07:00]
   0: S1      S1..............................................                               d: 01.03 06:40 [01.03 06:40]  [{name=T3, day=2021-03-01, id=T3_MON, src=0}]
   1: S2      S2.............................................. a: 01.03 07:00 [01.03 07:00]
leg 3: (S2, S2) [2021-03-01 07:00] -> (S2, S2) [2021-03-01 07:00]
  FOOTPATH (duration=0)

)";

/*constexpr auto const multiple_paths_journey_lowTransfers = R"(
[2021-03-01 01:00, 2021-03-01 08:00]
TRANSFERS: 0
     FROM: (S0, S0) [2021-03-01 05:00]
       TO: (S2, S2) [2021-03-01 08:00]
leg 0: (S0, S0) [2021-03-01 05:00] -> (S2, S2) [2021-03-01 08:00]
   0: S0      S0.............................................. d: 01.03 05:00
[01.03 05:00]  [{name=T1, day=2021-03-01, id=T1_MON, src=0}] 1: S2
S2.............................................. a: 01.03 08:00 [01.03 08:00]
leg 1: (S2, S2) [2021-03-01 08:00] -> (S2, S2) [2021-03-01 08:00]
  FOOTPATH (duration=0)

)";*/

TEST(a_star_query, multiple_paths) {
  auto const tt = load_gtfs(multiple_paths_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = a_star_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{March / 1 / 2021} + 1h});
  // It depends on the parameter ALPHA in the cost function if
  // 'multiple_paths_journey_lowTransfers' or 'multiple_paths_journey_lowTime'
  // is the desired result.
  EXPECT_EQ(std::string_view{multiple_paths_journey_lowTime},
            result_str(results, tt));
}

// --- abc ---  (Note: The data 'files_abc' isn't here, but in
// 'test\loader\hrd\hrd_timetable.h' instead.)

constexpr auto const abc_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]
leg 3: (C, 0000003) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 07:15]
  FOOTPATH (duration=0)

)";

TEST(a_star_query, abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results =
      a_star_search(tt, tbd, "0000001", "0000003",
                    unixtime_t{sys_days{March / 30 / 2020} + 5h});
  EXPECT_EQ(std::string_view{abc_journeys}, result_str(results, tt));
}

// --- profile_abc ---

constexpr auto const profile_abc_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]
leg 3: (C, 0000003) [2020-03-30 07:15] -> (C, 0000003) [2020-03-30 07:15]
  FOOTPATH (duration=0)

[2020-03-30 05:30, 2020-03-30 07:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:45]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]
leg 3: (C, 0000003) [2020-03-30 07:45] -> (C, 0000003) [2020-03-30 07:45]
  FOOTPATH (duration=0)

)";

TEST(a_star_query, profile_abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results =
      a_star_search(tt, tbd, "0000001", "0000003",
                    interval{unixtime_t{sys_days{March / 30 / 2020}} + 5h,
                             unixtime_t{sys_days{March / 30 / 2020}} + 6h});
  EXPECT_EQ(std::string_view{profile_abc_journeys}, result_str(results, tt));
}

// --- intermodal_abc ---

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

TEST(a_star_query, intermodal_abc) {
  auto const tt = load_hrd(files_abc);
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
  EXPECT_EQ(std::string_view{intermodal_abc_journeys}, result_str(results, tt));
}
