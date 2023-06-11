#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/timetable.h"

#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;
using std::chrono::operator""h;
using std::chrono::operator""min;

namespace {

mem_dir test_files() {
  using std::filesystem::path;
  return {
      {{path{kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
)"}},
       {path{kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
)"}},
       {path{kCalendarDatesFile}, std::string{R"(service_id,date,exception_type
S_RE1,20190503,1
S_RE2,20190504,1
)"}},
       {path{kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,RE 1,,,3
R_RE2,DB,RE 2,,,3
)"}},
       {path{kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,RE 1,
R_RE2,S_RE2,T_RE2,RE 2,
)"}},
       {path{kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,49:00:00,49:00:00,A,1,0,0
T_RE1,50:00:00,50:00:00,B,2,0,0
T_RE2,00:30:00,00:30:00,B,1,0,0
T_RE2,00:45:00,00:45:00,C,2,0,0
T_RE2,01:00:00,01:00:00,D,3,0,0
)"}}}};
}

}  // namespace

TEST(rt, gtfsrt_resolve_static_trip) {
  timetable tt;
  rt_timetable rtt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);

  {  // test start time >24:00:00
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_start_time() = "49:00:00";
    *td.mutable_start_date() = "20190503";
    *td.mutable_trip_id() = "T_RE1";

    auto const t = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3}, tt,
                                          rtt, source_idx_t{0}, td);
    ASSERT_TRUE(t.valid());
  }

  {  // test start time that's on the prev. day in UTC
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_start_time() = "00:30:00";
    *td.mutable_start_date() = "20190504";
    *td.mutable_trip_id() = "T_RE2";

    auto const t = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt,
                                          rtt, source_idx_t{0}, td);
    ASSERT_TRUE(t.valid());
  }

  {  // test without start_time and start_date (assuming "today" as date)
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_trip_id() = "T_RE2";

    // 2019-05-03 00:30 CEST is
    // 2019-05-02 21:30 UTC
    // -> we give "today" in UTC (start_day would be local days)
    auto const t = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 2}, tt,
                                          rtt, source_idx_t{0}, td);
    ASSERT_TRUE(t.valid());
  }
}

TEST(rt, gtfsrt_resolve_rt_trip) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  // Load static timetable.
  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);

  // Create empty RT timetable.
  auto rtt = rt_timetable{};
  rtt.transport_traffic_days_ = tt.transport_traffic_days_;
  rtt.bitfields_ = tt.bitfields_;
  rtt.base_day_ = date::sys_days{2019_y / May / 3};
  rtt.base_day_idx_ = tt.day_idx(rtt.base_day_);
  rtt.location_rt_transports_.resize(tt.n_locations());

  // Create basic update message.
  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 4} + 9h));

  auto const entity = msg.add_entity();
  entity->set_id("1");
  entity->set_is_deleted(false);

  auto const td = entity->mutable_trip_update()->mutable_trip();
  td->set_start_time("00:30:00");
  td->set_start_date("20190504");
  td->set_trip_id("T_RE2");

  // ** UPDATE 0: update first arrival, check propagation **
  transit_realtime::FeedMessage msg0;
  msg0.mutable_header()->CopyFrom(msg.header());
  auto const e0 = msg0.add_entity();
  e0->mutable_trip_update()->mutable_trip()->CopyFrom(*td);
  {
    auto const stop_update = e0->mutable_trip_update()->add_stop_time_update();
    stop_update->set_stop_sequence(1U);
    stop_update->mutable_arrival()->set_delay(900);
  }

  auto stats = rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg0);
  auto r = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt, rtt,
                                  source_idx_t{0}, *td);
  EXPECT_EQ(1U, stats.total_entities_success_);
  if (stats.total_entities_success_ != 1U) {
    std::cout << stats << "\n";
  }

  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r.rt_.has_value());
  ASSERT_TRUE(r.t_.has_value());
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 22h + 45min,  // assumed +15
            rtt.unix_event_time(*r.rt_, 0U, event_type::kDep));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 00min,  // propagated +15
            rtt.unix_event_time(*r.rt_, 1U, event_type::kArr));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 00min,  // propagated +15
            rtt.unix_event_time(*r.rt_, 1U, event_type::kDep));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 15min,  // propagated +15
            rtt.unix_event_time(*r.rt_, 2U, event_type::kArr));

  // ** UPDATE 1: reduce delay, check time/delay update and propagation **
  {
    auto const stop_update =
        entity->mutable_trip_update()->add_stop_time_update();
    stop_update->set_stop_sequence(1);
    stop_update->mutable_departure()->set_time(
        to_unix(date::sys_days{2019_y / May / 3} + 22h + 35min));
  }

  {
    auto const stop_update =
        entity->mutable_trip_update()->add_stop_time_update();
    stop_update->set_stop_id("D");
    stop_update->mutable_arrival()->set_delay(600);
  }

  stats = rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(1U, stats.total_entities_success_);

  r = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt, rtt,
                             source_idx_t{0}, *td);
  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r.rt_.has_value());
  ASSERT_TRUE(r.t_.has_value());
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 22h + 35min,  // absolute update
            rtt.unix_event_time(*r.rt_, 0U, event_type::kDep));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 22h + 50min,  // propagated +5
            rtt.unix_event_time(*r.rt_, 1U, event_type::kArr));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 22h + 50min,  // propagated +5
            rtt.unix_event_time(*r.rt_, 1U, event_type::kDep));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 10min,  // rel. update +10
            rtt.unix_event_time(*r.rt_, 2U, event_type::kArr));
}