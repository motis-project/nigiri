#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/util.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,
Z,Z,,-5.0,-5.0,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3
R9,DB,9,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,
R1,S1,T0,RE 1,
R9,S1,TZ,9,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
T0,08:00:00,08:00:00,A,1,0,0
T0,09:00:00,09:00:00,B,2,0,0
TZ,07:00:00,07:00:00,Z,1,0,0
TZ,07:55:00,07:55:00,A,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
A,A,3,,,,TZ,T1
)");
}

}  // namespace

TEST(rt, delay_test) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  auto tt =
      loader::load(test_files(), {2019_y / March / 25, 2019_y / November / 1});

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 1});

  transit_realtime::FeedMessage msg;
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 1} + 8h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_trip_id("T1");
  td->set_start_date("20190501");
  td->set_start_time("10:00:00");

  auto const stop_update_a = e->mutable_trip_update()->add_stop_time_update();
  stop_update_a->set_stop_sequence(1U);
  stop_update_a->mutable_departure()->set_delay(10 * 60);  // 10 minutes delay

  auto const stop_update_b = e->mutable_trip_update()->add_stop_time_update();
  stop_update_b->set_stop_sequence(2U);
  stop_update_b->mutable_arrival()->set_delay(0);  // on time

  // The forbidden transfer TZ -> T1 at A (TZ -> T0 stays allowed) moves T1 to
  // a split route (stop A becomes a virtual location of A).
  ASSERT_EQ(3U, tt.n_routes());

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);
  rtt.update_lbs(tt);

  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{sys_days{2019_y / May / 1} + 7h,
                                          sys_days{2019_y / May / 1} + 9h},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = true,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"A", source_idx_t{0}}),
                  0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {"B", source_idx_t{0}}),
                        0_minutes, 0U}},
      .max_transfers_ = 1,
      .min_connection_count_ = 1,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false,
      .prf_idx_ = 0,
      .allowed_claszes_ = routing::all_clasz_allowed(),
      .require_bike_transport_ = false,
      .require_car_transport_ = false,
      .via_stops_ = {}};

  auto const result =
      nigiri::test::search_pong(tt, &rtt, std::move(q), direction::kForward);

  ASSERT_FALSE(result.empty());
  auto const& journey = result.begin();

  auto const expected_arrival = sys_days{2019_y / May / 1} + 9h;
  EXPECT_EQ(to_unix(journey->dest_time_), to_unix(expected_arrival));
}

TEST(rt, stop_time_update_matching) {
  auto const tt = loader::load(mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.01,1.01,,
C,C,,0.02,1.03,,1,
C1,C,,0.02,1.03,,0,C
C2,C,,0.0201,1.03,,0,C
D,D,,0.04,1.06,,
E,E,,0.06,1.09,,
Z,Z,,-5.0,-5.0,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,
R1,S1,T2,RE 1,
R1,S1,T3,RE 1,
R1,S1,T3a,RE 1,
R1,S1,TZ,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:10:00,10:10:00,B,2,0,0
T1,10:20:00,10:20:00,A,3,0,0
T1,10:30:00,10:30:00,C1,4,0,0
T1,10:40:00,10:40:00,D,5,0,0
T1,10:50:00,10:50:00,E,6,0,0
T2,11:00:00,11:00:00,A,1,0,0
T2,11:10:00,11:10:00,B,2,0,0
T2,11:20:00,11:20:00,A,3,0,0
T3,12:00:00,12:00:00,A,1,0,0
T3,12:10:00,12:10:00,C1,2,0,0
T3,12:20:00,12:20:00,D,3,0,0
T3,12:30:00,12:30:00,E,4,0,0
T3a,11:45:00,11:45:00,A,1,0,0
T3a,11:55:00,11:55:00,C1,2,0,0
T3a,12:05:00,12:05:00,D,3,0,0
T3a,12:15:00,12:15:00,E,4,0,0
TZ,11:00:00,11:00:00,Z,1,0,0
TZ,11:50:00,11:50:00,C1,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
C1,C1,3,,,,TZ,T3
)"),
                               {2019_y / May / 1, 2019_y / May / 2});

  // The forbidden transfer TZ -> T3 at C1 (TZ -> T3a stays allowed) moves T3
  // to a split route (stop C1 becomes a virtual location of C1).
  {
    auto const route_of = [&](std::string_view const id) {
      for (auto t = trip_idx_t{0U}; t != tt.trip_id_strings_.size(); ++t) {
        if (tt.trip_id_strings_[t].view() == id) {
          return tt.transport_route_[tt.trip_transport_ranges_[t][0].first];
        }
      }
      return route_idx_t::invalid();
    };
    ASSERT_NE(route_of("T3"), route_of("T3a"));
  }

  auto rtt = rt::create_rt_timetable(tt, sys_days{2019_y / May / 1});
  rt::gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "tag",
                        rt::json_to_protobuf(R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1556697600"
 },
 "entity": [{
  "id": "1",
  "tripUpdate": {
   "trip": {"tripId": "T1", "startDate": "20190501"},
   "stopTimeUpdate": [
    {"stopSequence": 3, "stopId": "A", "arrival": {"delay": 300}},
    {"stopSequence": 7, "stopId": "C1",
     "stopTimeProperties": {"assignedStopId": "C2"}},
    {"stopSequence": 8, "stopId": "D", "scheduleRelationship": "SKIPPED"},
    {"stopSequence": 9, "stopId": "E", "arrival": {"delay": 420}}
   ]
  }
 }, {
  "id": "2",
  "tripUpdate": {
   "trip": {"tripId": "T2", "startDate": "20190501"},
   "stopTimeUpdate": [
    {"stopSequence": 2, "stopId": "B", "arrival": {"delay": 60}},
    {"stopSequence": 9, "stopId": "A", "arrival": {"delay": 180}}
   ]
  }
 }, {
  "id": "3",
  "tripUpdate": {
   "trip": {"tripId": "T3", "startDate": "20190501"},
   "stopTimeUpdate": [
    {"stopSequence": 2, "stopId": "C1", "scheduleRelationship": "SKIPPED"},
    {"stopSequence": 2, "stopId": "C2", "arrival": {"delay": 120}},
    {"stopSequence": 3, "stopId": "D", "arrival": {"delay": 240}},
    {"stopSequence": 3, "stopId": "D", "scheduleRelationship": "SKIPPED"}
   ]
  }
 }]
})"));

  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id("T1");
  td.set_start_date("20190501");
  auto const [r, _] = rt::gtfsrt_resolve_run(sys_days{2019_y / May / 1}, tt,
                                             &rtt, source_idx_t{0}, td);
  auto const fr = rt::frun{tt, &rtt, r};
  auto const base = sys_days{2019_y / May / 1} + 8h;
  EXPECT_EQ(base, fr[0].time(event_type::kDep));
  EXPECT_EQ(base + 10min, fr[1].time(event_type::kArr));
  EXPECT_EQ(base + 25min, fr[2].time(event_type::kArr));
  EXPECT_EQ(std::string_view{"C2"}, fr[3].get_location_id());
  EXPECT_TRUE(fr[4].is_cancelled());
  EXPECT_EQ(base + 57min, fr[5].time(event_type::kArr));

  td.set_trip_id("T2");
  auto const [r2, _2] = rt::gtfsrt_resolve_run(sys_days{2019_y / May / 1}, tt,
                                               &rtt, source_idx_t{0}, td);
  auto const fr2 = rt::frun{tt, &rtt, r2};
  EXPECT_EQ(base + 60min, fr2[0].time(event_type::kDep));
  EXPECT_EQ(base + 83min, fr2[2].time(event_type::kArr));

  td.set_trip_id("T3");
  auto const [r3, _3] = rt::gtfsrt_resolve_run(sys_days{2019_y / May / 1}, tt,
                                               &rtt, source_idx_t{0}, td);
  auto const fr3 = rt::frun{tt, &rtt, r3};
  EXPECT_FALSE(fr3[1].is_cancelled());
  EXPECT_EQ(std::string_view{"C2"}, fr3[1].get_location_id());
  EXPECT_EQ(base + 132min, fr3[1].time(event_type::kArr));
  EXPECT_FALSE(fr3[2].is_cancelled());
  EXPECT_EQ(base + 144min, fr3[2].time(event_type::kArr));
}
