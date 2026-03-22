#include <nigiri/routing/raptor/raptor.h>
#include <nigiri/routing/search.h>
#include <nigiri/routing/tb/query_engine.h>
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/tb_a_star/a_star.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "../loader/hrd/hrd_timetable.h"
#include "gtest/gtest.h"
namespace nigiri::routing::tb {

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::loader;
using namespace std::chrono_literals;
using namespace nigiri::test_data::hrd_timetable;

timetable load_gtfs(auto const& files) {
  timetable tt;
  tt.date_range_ = interval<date::sys_days>{sys_days{2021_y / March / 1},
                                            sys_days{2021_y / March / 8}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, files(), tt);
  loader::finalize(tt);
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

unsigned int calculate_transfer_factor(std::vector<journey> const& journeys,
                                       int index) {
  double higher_bound = std::numeric_limits<double>::max();
  double lower_bound = std::numeric_limits<double>::min();
  for (std::size_t j = 0; j < journeys.size(); ++j) {
    if (j == index) continue;
    utl::verify(journeys[index].transfers_ != journeys[j].transfers_,
                "Two travels with same amount of transfers");
    auto factor = static_cast<double>((journeys[j].arrival_time() -
                                       journeys[index].arrival_time())
                                          .count()) /
                  (journeys[index].transfers_ - journeys[j].transfers_);
    if (journeys[index].transfers_ < journeys[j].transfers_ &&
        factor > lower_bound)
      lower_bound = factor;
    if (journeys[index].transfers_ > journeys[j].transfers_ &&
        factor < higher_bound)
      higher_bound = factor;
  }
  if (higher_bound == std::numeric_limits<double>::max())
    return static_cast<int>(lower_bound) == lower_bound
               ? lower_bound + 1
               : std::ceil(lower_bound);
  int higher_bound_floor = static_cast<int>(higher_bound);
  utl::verify(
      (higher_bound_floor == higher_bound ? higher_bound - 1
                                          : higher_bound_floor) > lower_bound,
      "No Integer exists between bounds");
  auto result = higher_bound_floor == higher_bound ? higher_bound - 1
                                                   : higher_bound_floor;
  if (result < 0) throw std::runtime_error("Negative factor");
  return static_cast<unsigned int>(result);
}

void verify_pareto_set(auto const& files,
                       std::string_view const& start,
                       std::string_view const& stop,
                       unixtime_t const& start_time,
                       int expectedNumberOfJourneys) {
  auto const tt = load_gtfs(files);
  auto const tbd = preprocess(tt, profile_idx_t{0});

  query q{.start_time_ = start_time,
          .start_match_mode_ = location_match_mode::kEquivalent,
          .dest_match_mode_ = location_match_mode::kEquivalent,
          .start_ = {{tt.locations_.location_id_to_idx_.at(
                          {start, source_idx_t{0}}),
                      0_minutes, 0U}},
          .destination_ = {
              {tt.locations_.location_id_to_idx_.at({stop, source_idx_t{0}}),
               0_minutes, 0U}}};

  static auto search_state = routing::search_state{};
  auto algo_state = query_state{tt, tbd};

  search<direction::kForward, tb::query_engine<false>> tb_searcher{
      tt, nullptr, search_state, algo_state, q};

  auto results = tb_searcher.execute().journeys_->els_;

  EXPECT_EQ(results.size(), expectedNumberOfJourneys);
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results.size() > 1)
      a_star::transfer_factor = calculate_transfer_factor(results, i);
    static auto a_star_search_state = routing::search_state{};
    auto a_star_algo_state = a_star::a_star_state{tbd};
    search<direction::kForward, a_star::tb_a_star> a_star_searcher{
        tt, nullptr, a_star_search_state, a_star_algo_state, q};

    auto a_star = a_star_searcher.execute().journeys_->els_.front();
    EXPECT_EQ(results[i], a_star_searcher.execute().journeys_->els_.front());
  }

  search_state = routing::search_state{};
  auto raptor_algo_state = raptor_state{};

  search<direction::kForward,
         raptor<direction::kForward, false, 0, search_mode::kOneToOne>>
      raptor_searcher{tt, nullptr,     search_state, raptor_algo_state,
                      q,  std::nullopt};

  results = raptor_searcher.execute().journeys_->els_;

  EXPECT_EQ(results.size(), expectedNumberOfJourneys);
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results.size() > 1)
      a_star::transfer_factor = calculate_transfer_factor(results, i);
    static auto a_star_search_state = routing::search_state{};
    auto a_star_algo_state = a_star::a_star_state{tbd};

    search<direction::kForward, a_star::tb_a_star> a_star_searcher{
        tt, nullptr, a_star_search_state, a_star_algo_state, q};

    auto found_journey = a_star_searcher.execute().journeys_->els_.front();
    found_journey.legs_.pop_back();
    EXPECT_EQ(results[i], found_journey);
  }
}

loader::mem_dir simple_pareto_verification_files() {
  return loader::mem_dir::read(R"(
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
R0,DTA,R0,R0,"S0 -> S3",2
R1,DTA,R1,R1,"S0 -> S4",2
R2,DTA,R2,R2,"S2 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1
R2,MON,R2_MON,R2_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,03:00:00,03:00:00,S3,3,0,0
R1_MON,00:00:00,00:00:00,S0,0,0,0
R1_MON,01:00:00,01:00:00,S2,1,0,0
R1_MON,02:00:00,02:00:00,S4,2,0,0
R2_MON,01:10:00,01:10:00,S2,0,0,0
R2_MON,01:30:00,01:30:00,S3,1,0,0
)");
}
TEST(tb_a_star, pareto_verification) {
  verify_pareto_set(simple_pareto_verification_files, "S0", "S3",
                    sys_days{year{2021} / March / day{1}}, 2);
}

TEST(tb_a_star, pareto_verification_with_early_start_time) {
  verify_pareto_set(simple_pareto_verification_files, "S0", "S3",
                    sys_days{year{2021} / February / day{28}} + 23h, 2);
}

loader::mem_dir later_pareto_verification_files() {
  return loader::mem_dir::read(R"(
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
R0,DTA,R0,R0,"S0 -> S3",2
R1,DTA,R1,R1,"S0 -> S4",2
R2,DTA,R2,R2,"S2 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1
R2,MON,R2_MON,R2_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,01:00:00,01:00:00,S0,0,0,0
R0_MON,02:00:00,02:00:00,S1,1,0,0
R0_MON,04:00:00,04:00:00,S3,3,0,0
R1_MON,01:00:00,01:00:00,S0,0,0,0
R1_MON,02:00:00,02:00:00,S2,1,0,0
R1_MON,03:00:00,03:00:00,S4,2,0,0
R2_MON,02:10:00,02:10:00,S2,0,0,0
R2_MON,02:30:00,02:30:00,S3,1,0,0
)");
}
TEST(tb_a_star, pareto_verification_early_start_time_same_day) {
  verify_pareto_set(simple_pareto_verification_files, "S0", "S3",
                    sys_days{year{2021} / March / day{1}}, 2);
}
loader::mem_dir overnight_pareto_verification_files() {
  return loader::mem_dir::read(R"(
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
R0,DTA,R0,R0,"S0 -> S3",2
R1,DTA,R1,R1,"S0 -> S4",2
R2,DTA,R2,R2,"S2 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1
R2,MON,R2_MON,R2_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,10:00:00,10:00:00,S0,0,0,0
R0_MON,11:00:00,11:00:00,S1,1,0,0
R0_MON,26:00:00,26:00:00,S3,3,0,0
R1_MON,10:00:00,10:00:00,S0,0,0,0
R1_MON,11:00:00,11:00:00,S2,1,0,0
R1_MON,12:00:00,12:00:00,S4,2,0,0
R2_MON,11:10:00,11:10:00,S2,0,0,0
R2_MON,24:30:00,24:30:00,S3,1,0,0
)");
}
TEST(tb_a_star, overnight_pareto_verification) {
  verify_pareto_set(overnight_pareto_verification_files, "S0", "S3",
                    sys_days{year{2021} / March / day{1}} + 10h, 2);
}

loader::mem_dir three_journeys_pareto_verification_files() {
  return loader::mem_dir::read(R"(
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

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307


# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S3",2
R1,DTA,R1,R1,"S0 -> S4",2
R2,DTA,R2,R2,"S2 -> S3",2
R3,DTA,R2,R2,"S2 -> S3",2
R4,DTA,R2,R2,"S2 -> S3",2
R5,DTA,R2,R2,"S2 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1
R2,MON,R2_MON,R2_MON,1
R3,MON,R3_MON,R3_MON,1
R4,MON,R4_MON,R4_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,03:00:00,03:00:00,S3,3,0,0
R1_MON,00:00:00,00:00:00,S0,0,0,0
R1_MON,01:00:00,01:00:00,S2,1,0,0
R1_MON,02:00:00,02:00:00,S4,2,0,0
R2_MON,01:10:00,01:10:00,S2,0,0,0
R2_MON,02:30:00,02:30:00,S3,1,0,0
R3_MON,01:10:00,01:10:00,S2,0,0,0
R3_MON,01:30:00,01:30:00,S5,1,0,0
R3_MON,02:00:00,02:00:00,S6,2,0,0
R4_MON,01:40:00,01:40:00,S5,0,0,0
R4_MON,02:10:00,02:10:00,S3,0,0,0
)");
}
TEST(tb_a_star, three_journeys_pareto_verification) {
  verify_pareto_set(three_journeys_pareto_verification_files, "S0", "S3",
                    sys_days{year{2021} / March / day{1}}, 3);
}

TEST(tb_a_star, abc_pareto_verification) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});

  query q{.start_time_ = interval{unixtime_t{sys_days{March / 30 / 2020}} + 5h,
                                  unixtime_t{sys_days{March / 30 / 2020}} + 6h},
          .start_match_mode_ = location_match_mode::kEquivalent,
          .dest_match_mode_ = location_match_mode::kEquivalent,
          .start_ = {{tt.locations_.location_id_to_idx_.at(
                          {"0000001", source_idx_t{0}}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                {"0000003", source_idx_t{0}}),
                            0_minutes, 0U}}};

  static auto search_state = routing::search_state{};
  auto algo_state = query_state{tt, tbd};

  search<direction::kForward, tb::query_engine<false>> tb_searcher{
      tt, nullptr, search_state, algo_state, q};

  auto results = tb_searcher.execute().journeys_->els_;

  static auto a_star_search_state = routing::search_state{};
  auto a_star_algo_state = a_star::a_star_state{tbd};

  search<direction::kForward, a_star::tb_a_star> a_star_searcher{
      tt, nullptr, a_star_search_state, a_star_algo_state, q};

  auto result_a_star = a_star_searcher.execute().journeys_->els_;
  EXPECT_TRUE(result_a_star.front() == results.front() ||
              result_a_star.front() == results.back());
  EXPECT_TRUE(result_a_star.back() == results.front() ||
              result_a_star.back() == results.back());
  EXPECT_NE(result_a_star.back(), result_a_star.front());

  search_state = routing::search_state{};
  auto raptor_algo_state = raptor_state{};

  search<direction::kForward,
         raptor<direction::kForward, false, 0, search_mode::kOneToOne>>
      raptor_searcher{tt, nullptr,     search_state, raptor_algo_state,
                      q,  std::nullopt};

  results = raptor_searcher.execute().journeys_->els_;
  result_a_star.front().legs_.pop_back();
  result_a_star.back().legs_.pop_back();
  EXPECT_TRUE(result_a_star.front() == results.front() ||
              result_a_star.front() == results.back());
  EXPECT_TRUE(result_a_star.back() == results.front() ||
              result_a_star.back() == results.back());
  EXPECT_NE(result_a_star.back(), result_a_star.front());
}

mem_dir earlier_stop_transfer_files() {
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
TEST(tb_a_star, earlier_stop_transfer_verification) {
  verify_pareto_set(earlier_stop_transfer_files, "S3", "S2",
                    sys_days{year{2021} / March / day{1}}, 1);
}

mem_dir another_longer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
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
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S1 -> S3",2
R1,DTA,R1,R1,"S3 -> S7",2
R2,DTA,R2,R2,"S7 -> S4",2
R3,DTA,R3,R3,"S4 -> S10",2
R4,DTA,R4,R4,"S1 -> S7",2
R5,DTA,R5,R5,"S7 -> S9",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,0
R2,MON,R2_MON,R2_MON,0
R3,MON,R3_MON,R3_MON,0
R4,MON,R4_MON,R4_MON,0
R5,MON,R5_MON,R5_MON,0

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S1,0,0,0
R0_MON,01:00:00,01:00:00,S2,1,0,0
R0_MON,02:00:00,02:00:00,S3,2,0,0
R1_MON,02:10:00,02:10:00,S3,0,0,0
R1_MON,03:00:00,03:00:00,S7,1,0,0
R2_MON,03:10:00,03:10:00,S7,0,0,0
R2_MON,04:00:00,04:00:00,S8,1,0,0
R2_MON,05:00:00,05:00:00,S5,2,0,0
R2_MON,06:00:00,06:00:00,S4,3,0,0
R3_MON,06:10:00,06:10:00,S4,0,0,0
R3_MON,07:00:00,07:00:00,S10,1,0,0
R4_MON,00:00:00,00:00:00,S1,0,0,0
R4_MON,01:00:00,01:00:00,S6,1,0,0
R4_MON,03:00:00,03:00:00,S7,2,0,0
R5_MON,03:10:00,03:10:00,S7,0,0,0
R5_MON,04:00:00,04:00:00,S9,1,0,0
)");
}
TEST(tb_a_star, another_longer_gtfs) {
  verify_pareto_set(another_longer_files, "S1", "S10",
                    sys_days{year{2021} / March / day{1}}, 1);
}

mem_dir overnight_files() {
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
R0,DTA,R0,R0,"S0 -> S2",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,TUE,R1_TUE,R1_TUE,0


# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,23:30:00,23:30:00,S0,0,0,0
R0_MON,23:55:00,23:55:00,S1,1,0,0
R0_MON,25:30:00,25:30:00,S2,2,0,0
R1_TUE,00:05:00,00:05:00,S1,0,0,0
R1_TUE,01:00:00,01:00:00,S2,1,0,0
)");
}
TEST(tb_a_star, overnight_pareto_verification_files) {
  verify_pareto_set(overnight_files, "S0", "S2",
                    sys_days{year{2021} / March / day{1}} + 23h + 30min, 2);
}
}  // namespace nigiri::routing::tb
