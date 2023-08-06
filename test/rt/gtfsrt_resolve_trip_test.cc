#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;

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

constexpr auto const kTransportAfterUpdate = std::string_view{
    R"(   0: B       B...............................................                                                             d: 03.05 22:30 [04.05 00:30]  RT 03.05 22:35 [04.05 00:35]  [{name=Bus RE 2, day=2019-05-03, id=T_RE2, src=0}]
   1: C       C............................................... a: 03.05 22:45 [04.05 00:45]  RT 03.05 22:50 [04.05 00:50]  d: 03.05 22:45 [04.05 00:45]  RT 03.05 22:50 [04.05 00:50]  [{name=Bus RE 2, day=2019-05-03, id=T_RE2, src=0}]
   2: D       D............................................... a: 03.05 23:00 [04.05 01:00]  RT 03.05 23:10 [04.05 01:10]
)"};

}  // namespace

TEST(rt, gtfsrt_resolve_static_trip) {
  timetable tt;
  rt_timetable rtt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);

  {  // test start time >24:00:00

    // first dep:
    //      49:00     2019-05-03    GMT -02:00
    //   -> 01:00.2   2019-05-03    GMT -02:00
    //   -> 23:00.1   2019-05-03    UTC
    //   -> 23:00     2019-05-04    UTC

    auto td = transit_realtime::TripDescriptor();
    *td.mutable_start_time() = "49:00:00";
    *td.mutable_start_date() = "20190503";
    *td.mutable_trip_id() = "T_RE1";

    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, rtt, source_idx_t{0}, td);
    EXPECT_TRUE(r.valid());
  }

  {  // test start time that's on the prev. day in UTC
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_start_time() = "00:30:00";
    *td.mutable_start_date() = "20190504";
    *td.mutable_trip_id() = "T_RE2";

    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4},
                                               tt, rtt, source_idx_t{0}, td);
    EXPECT_TRUE(r.valid());
  }

  {  // test without start_time and start_date (assuming "today" as date)
    auto td = transit_realtime::TripDescriptor();
    *td.mutable_trip_id() = "T_RE2";

    // 2019-05-03 00:30 CEST is
    // 2019-05-02 21:30 UTC
    // -> we give "today" in UTC (start_day would be local days)
    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4},
                                               tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
  }
}

mem_dir miami_test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,agency_lang
Miami-Dade Transit,Miami-Dade Transit,http://www.miamidade.gov/transit,America/New_York,en

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,stop_code,stop_desc
1,Quail Roost Dr & SW 115 Ave,25.592450,-80.377950,115AQRSN,SW 107TH AVE & SW 112TH ST
3614,SW 107TH AVE @ SW 128TH ST,25.591810,-80.379840,QRSD17A3,
3213,SW 117 AV @ QUAIL ROOST DR,25.590330,-80.380730,C117QUA4,

# routes.txt
agency_id,route_id,route_short_name,route_long_name,route_type,route_color,route_text_color,eligibility_restricted
Miami-Dade Transit,27757,1,SO.MIAMI HTS-PERRINE VIA SOUTHLAND,3,FF00FF,FFFFFF,-999

# trips.txt
route_id,trip_id,service_id,trip_headsign,direction_id,block_id,shape_id,drt_advance_book_min,peak_offpeak
27757,5456914,1,1 - SW 168 St,0,1484970,196385,0.0,0
27757,5456915,1,1 - SW 168 St,0,1484970,196385,0.0,0

# stop_times.txt
trip_id,stop_id,arrival_time,departure_time,timepoint,stop_sequence,shape_dist_traveled,continuous_pickup,continuous_drop_off,start_service_area_radius,end_service_area_radius,departure_buffer
5456914,1,15:15:00,15:15:00,1,1,,-999,-999,-999.0,-999.0,0
5456914,3614,15:15:35,15:15:35,0,2,0.1548,-999,-999,-999.0,-999.0,0
5456914,3213,15:16:20,15:16:20,0,3,0.354,-999,-999,-999.0,-999.0,0
5456915,1,47:15:00,47:15:00,1,1,,-999,-999,-999.0,-999.0,0
5456915,3614,47:15:35,47:15:35,0,2,0.1548,-999,-999,-999.0,-999.0,0
5456915,3213,47:16:20,47:16:20,0,3,0.354,-999,-999,-999.0,-999.0,0

# calendar_dates.txt
service_id,date,exception_type
1,20230904,2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
1,1,1,1,1,1,0,0,20230724,20231112
)");
}

TEST(rt, resolve_tz) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2023_y / August / 3},
                    date::sys_days{2023_y / August / 5}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, miami_test_files(), tt);
  loader::finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 3});

  {
    auto td = transit_realtime::TripDescriptor();
    td.set_start_date("20230803");
    td.set_trip_id("5456914");

    auto const r = gtfsrt_resolve_run(date::sys_days{2023_y / August / 3}, tt,
                                      rtt, source_idx_t{0U}, td);
    EXPECT_TRUE(r.first.valid());
  }

  {
    // first dep:
    //      47:00     2023-08-03    GMT +04:00
    //   -> 23:00.1   2023-08-03    GMT +04:00
    //   -> 03:00.2   2023-08-03    UTC
    //   -> 03:00     2023-08-05    UTC
    auto td = transit_realtime::TripDescriptor();
    td.set_start_date("20230803");
    td.set_trip_id("5456915");

    auto const r = gtfsrt_resolve_run(date::sys_days{2023_y / August / 3}, tt,
                                      rtt, source_idx_t{0U}, td);
    EXPECT_TRUE(r.first.valid());
  }
}

TEST(rt, gtfs_rt_update) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

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

  // Basic checks with rt_timetable=nullptr.
  auto const scheduled =
      std::array<unixtime_t, 4>{sys_days{2019_y / May / 3} + 22h + 30min,
                                sys_days{2019_y / May / 3} + 22h + 45min,
                                sys_days{2019_y / May / 3} + 22h + 45min,
                                sys_days{2019_y / May / 3} + 23h + 0min};
  auto const stop_ids = {"B", "C", "D"};
  auto i = 0U, j = 0U;
  auto fr = frun{tt, nullptr,
                 rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt,
                                        rtt, source_idx_t{0}, *td)
                     .first};
  ASSERT_TRUE(fr.valid());
  for (auto const [from, to] : utl::pairwise(fr)) {
    EXPECT_EQ(scheduled[i++], from.scheduled_time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[i++], to.scheduled_time(nigiri::event_type::kArr));
    EXPECT_EQ(scheduled[j++], from.time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[j++], to.time(nigiri::event_type::kArr));
  }
  for (auto const [id, stop] : utl::zip(stop_ids, fr)) {
    EXPECT_EQ(tt.locations_.get({id, source_idx_t{0U}}).l_,
              stop.get_location().l_);
  }

  // Basic checks with rt_timetable!=nullptr.
  fr = frun{tt, &rtt,
            rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt, rtt,
                                   source_idx_t{0}, *td)
                .first};
  i = j = 0U;
  for (auto const [from, to] : utl::pairwise(fr)) {
    EXPECT_EQ(scheduled[i++], from.scheduled_time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[i++], to.scheduled_time(nigiri::event_type::kArr));
    EXPECT_EQ(scheduled[j++], from.time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[j++], to.time(nigiri::event_type::kArr));
  }
  for (auto const [id, stop] : utl::zip(stop_ids, fr)) {
    EXPECT_EQ(tt.locations_.get({id, source_idx_t{0U}}).l_,
              stop.get_location().l_);
  }

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
  auto [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt,
                                       rtt, source_idx_t{0}, *td);
  EXPECT_EQ(1U, stats.total_entities_success_);
  if (stats.total_entities_success_ != 1U) {
    std::cout << stats << "\n";
  }

  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r.is_rt());
  ASSERT_TRUE(r.t_.is_valid());
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 22h + 45min,  // assumed +15
            rtt.unix_event_time(r.rt_, 0U, event_type::kDep));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 00min,  // propagated +15
            rtt.unix_event_time(r.rt_, 1U, event_type::kArr));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 00min,  // propagated +15
            rtt.unix_event_time(r.rt_, 1U, event_type::kDep));
  EXPECT_EQ(date::sys_days{2019_y / May / 3} + 23h + 15min,  // propagated +15
            rtt.unix_event_time(r.rt_, 2U, event_type::kArr));

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

  std::tie(r, t) = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4}, tt,
                                          rtt, source_idx_t{0}, *td);
  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r.is_rt());
  ASSERT_TRUE(r.t_.is_valid());

  auto const expected_rt = std::array<unixtime_t, 4>{
      sys_days{2019_y / May / 3} + 22h + 35min,  // absolute update +5
      sys_days{2019_y / May / 3} + 22h + 50min,  // propagated +5
      sys_days{2019_y / May / 3} + 22h + 50min,  // propagated +5
      sys_days{2019_y / May / 3} + 23h + 10min  // rel. update +10
  };

  i = j = 0U;
  fr = frun{tt, &rtt, r};
  for (auto const [from, to] : utl::pairwise(fr)) {
    EXPECT_EQ(scheduled[i++], from.scheduled_time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[i++], to.scheduled_time(nigiri::event_type::kArr));
    EXPECT_EQ(expected_rt[j++], from.time(nigiri::event_type::kDep));
    EXPECT_EQ(expected_rt[j++], to.time(nigiri::event_type::kArr));
  }

  for (auto const [id, stop] : utl::zip(stop_ids, fr)) {
    EXPECT_EQ(tt.locations_.get({id, source_idx_t{0U}}).l_,
              stop.get_location().l_);
  }

  // Ignore delays.
  fr = frun{tt, nullptr, r};
  i = j = 0U;
  for (auto const [from, to] : utl::pairwise(fr)) {
    EXPECT_EQ(scheduled[i++], from.scheduled_time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[i++], to.scheduled_time(nigiri::event_type::kArr));
    EXPECT_EQ(scheduled[j++], from.time(nigiri::event_type::kDep));
    EXPECT_EQ(scheduled[j++], to.time(nigiri::event_type::kArr));
  }

  std::stringstream ss;
  ss << frun{tt, &rtt, r};
  EXPECT_EQ(ss.str(), kTransportAfterUpdate);
  std::cout << ss.str() << "\n";
}