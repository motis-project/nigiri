#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
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

pareto_set<routing::journey> tripbased_search(timetable const& tt,
                                              tb::tb_data const& tbd,
                                              routing::query q) {
  static auto search_state = routing::search_state{};
  auto algo_state = tb::query_state{tt, tbd};

  return *(routing::search<direction::kForward, tb::query_engine<false>>{
      tt, nullptr, search_state, algo_state, std::move(q)}
               .execute()
               .journeys_);
}

pareto_set<routing::journey> tripbased_search(
    timetable const& tt,
    tb::tb_data const& tbd,
    std::string_view from,
    std::string_view to,
    routing::start_time_t const time) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({to, src}), 0_minutes, 0U}}};
  return tripbased_search(tt, tbd, std::move(q));
}

pareto_set<routing::journey> tripbased_intermodal_search(
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
  return tripbased_search(tt, tbd, std::move(q));
}

std::string results_str(auto const& results, timetable const& tt) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n";
  }
  return ss.str();
}

mem_dir no_transfer_files() {
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
THU,0,0,0,1,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, no_transfer) {
  auto const tt = load_gtfs(no_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto const transfers : tbd.segment_transfers_) {
    EXPECT_TRUE(transfers.empty());
  }
}

mem_dir same_day_transfer_files() {
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

TEST(tb_preprocess, same_day_transfer) {
  auto const tt = load_gtfs(same_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
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

TEST(tb_query, same_day_transfer) {
  auto const tt = load_gtfs(same_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S0", "S2", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(std::string_view{same_day_transfer_journeys},
            results_str(results, tt));
}

mem_dir next_day_transfer_files() {
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
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,TUE,R1_TUE,R1_TUE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_TUE,06:00:00,06:00:00,S1,0,0,0
R1_TUE,08:00:00,08:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, next_day_transfer) {
  auto const tt = load_gtfs(next_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(1, t.day_offset_);
}

mem_dir long_transfer_files() {
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
THU,0,0,0,1,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,76:00:00,76:00:00,S1,1,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, long_transfer) {
  auto const tt = load_gtfs(long_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(3, t.day_offset_);
}

mem_dir weekday_transfer_files() {
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
WD,1,1,1,1,1,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,WD,R0_WD,R0_WD,1
R1,WD,R1_WD,R1_WD,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_WD,00:00:00,00:00:00,S0,0,0,0
R0_WD,02:00:00,02:00:00,S1,1,0,0
R1_WD,01:00:00,01:00:00,S1,0,0,0
R1_WD,03:00:00,03:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, weekday_transfer) {
  auto const tt = load_gtfs(weekday_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"0111100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(1, t.day_offset_);
}

mem_dir daily_transfer_files() {
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
DLY,1,1,1,1,1,1,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,DLY,R0_DLY,R0_DLY,1
R1,DLY,R1_DLY,R1_DLY,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_DLY,00:00:00,00:00:00,S0,0,0,0
R0_DLY,02:00:00,02:00:00,S1,1,0,0
R1_DLY,03:00:00,03:00:00,S1,0,0,0
R1_DLY,04:00:00,04:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, daily_transfer) {
  auto const tt = load_gtfs(daily_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"111111100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
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

TEST(tb_preprocess, earlier_stop_transfer) {
  auto const tt = load_gtfs(earlier_stop_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}] + 3U;
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}] + 1U,
            t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
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

TEST(tb_query, early_stop_transfer) {
  auto const tt = load_gtfs(earlier_stop_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S3", "S2", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(std::string_view{earlier_stop_transfer_journeys},
            results_str(results, tt));
}

TEST(tb_query, no_journey_possible) {
  auto const tt = load_gtfs(earlier_stop_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S4", "S0", unixtime_t{sys_days{February / 28 / 2021} + 23h});
  EXPECT_EQ(0, results.size());
}

mem_dir earlier_transport_transfer_files() {
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
R0_MON1,02:00:00,02:00:00,S0,0,0,0
R0_MON1,03:00:00,03:00:00,S1,1,0,0
R0_MON1,04:00:00,04:00:00,S2,2,0,0
R0_MON1,05:00:00,05:00:00,S3,3,0,0
R0_MON1,05:00:00,05:00:00,S1,4,0,0
R0_MON1,06:00:00,06:00:00,S4,5,0,0
)");
}

TEST(tb_preprocess, earlier_transport_transfer) {
  auto const tt = load_gtfs(earlier_transport_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{1U}];
  ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{0U}] + 4U,
            t.to_segment_);
  // EXPECT_EQ(transport_idx_t{0U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

mem_dir uturn_transfer_files() {
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
R0,DTA,R0,R0,"S0 -> S1 -> S2",2
R1,DTA,R1,R1,"S2 -> S1 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,03:00:00,03:00:00,S2,0,0,0
R1_MON,04:00:00,04:00:00,S1,1,0,0
R1_MON,05:00:00,05:00:00,S3,2,0,0
)");
}

TEST(tb_preprocess, uturn_transfer) {
  auto const tt = load_gtfs(uturn_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto s = tb::segment_idx_t{1U}; s < tbd.segment_transfers_.size(); ++s) {
    EXPECT_TRUE(tbd.segment_transfers_[s].empty());
  }
  ASSERT_EQ(1U, tbd.segment_transfers_[tb::segment_idx_t{0U}].size());
  auto const t = tbd.segment_transfers_[tb::segment_idx_t{0U}][0];
  EXPECT_EQ(tb::segment_idx_t{3U}, t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

mem_dir unnecessary_transfer_1_files() {
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
R0,DTA,R0,R0,"S0 -> S1 -> S2",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,01:00:00,01:10:00,S1,0,0,0
R1_MON,04:00:00,04:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, unnecessary_transfer_1) {
  auto const tt = load_gtfs(unnecessary_transfer_1_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto const transfers : tbd.segment_transfers_) {
    EXPECT_TRUE(transfers.empty());
  }
}

mem_dir unnecessary_transfer_2_files() {
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

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2 -> S3 -> S4",2
R1,DTA,R1,R1,"S1 -> S2 -> S3 -> S5",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S2,1,0,0
R0_MON,02:00:00,02:00:00,S3,2,0,0
R0_MON,03:00:00,03:00:00,S4,3,0,0
R1_MON,00:10:00,00:10:00,S1,0,0,0
R1_MON,01:10:00,01:10:00,S2,1,0,0
R1_MON,02:10:00,02:10:00,S3,2,0,0
R1_MON,03:10:00,03:10:00,S5,3,0,0
)");
}

TEST(tb_preprocess, unnecessary_transfer_2) {
  auto const tt = load_gtfs(unnecessary_transfer_2_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}] + 1U;
  ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
  auto const t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}] + 2U,
            t.to_segment_);
  // EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  // EXPECT_EQ(0, t.day_offset_);
}

mem_dir early_train_files() {
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

TEST(tb_preprocess, early_train_journeys) {
  auto const tt = load_gtfs(early_train_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  tbd.print(std::cout, tt);
  ASSERT_EQ("R1", tt.transport_name(transport_idx_t{1U}));
  // auto const s = tbd.transport_first_segment_[transport_idx_t{1U}];
  // ASSERT_EQ(1U, tbd.segment_transfers_[s].size());
}

TEST(tb_query, early_train) {
  auto const tt = load_gtfs(early_train_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_search(
      tt, tbd, "S1", "S3", unixtime_t{sys_days{March / 04 / 2021} + 5h});
  EXPECT_EQ(std::string_view{early_train_journeys}, results_str(results, tt));
}

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

TEST(tb_query, abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results =
      tripbased_search(tt, tbd, "0000001", "0000003",
                       unixtime_t{sys_days{March / 30 / 2020} + 5h});
  EXPECT_EQ(std::string_view{abc_journeys}, results_str(results, tt));
}

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

TEST(tb_query, profile_abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results =
      tripbased_search(tt, tbd, "0000001", "0000003",
                       interval{unixtime_t{sys_days{March / 30 / 2020}} + 5h,
                                unixtime_t{sys_days{March / 30 / 2020}} + 6h});
  EXPECT_EQ(std::string_view{profile_abc_journeys}, results_str(results, tt));
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

TEST(tb_query, intermodal_abc) {
  auto const tt = load_hrd(files_abc);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const results = tripbased_intermodal_search(
      tt, tbd,
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000001", .src_ = source_idx_t{0U}}),
        10_minutes, 99U}},
      {{tt.locations_.location_id_to_idx_.at(
            {.id_ = "0000003", .src_ = source_idx_t{0U}}),
        15_minutes, 77U}},
      interval{unixtime_t{sys_days{March / 30 / 2020}} + 5_hours,
               unixtime_t{sys_days{March / 30 / 2020}} + 6_hours});
  EXPECT_EQ(std::string_view{intermodal_abc_journeys},
            results_str(results, tt));
}