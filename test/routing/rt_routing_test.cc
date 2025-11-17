#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "../../include/nigiri/routing/direct.h"

#include "../loader/hrd/hrd_timetable.h"

#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// T_RE1
// A | 01.05.  49:00 = 03.05. 01:00
// B | 01.05.  50:00 = 03.05. 02:00
//
// T_RE2
// B | 03.05.  00:30
// C | 03.05.  00:45
// D | 03.05.  01:00
//
// => delay T_RE2 at B  [03.05. 00:30]+2h = [03.05. 02:30]
// => search connection from A --> D (only possible if the transfer at B works!)
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,
C,C,,0.04,1.05,,
D,D,,0.06,1.07,,

# calendar_dates.txt
service_id,date,exception_type
S_RE1,20190501,1
S_RE2,20190503,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,RE 1,,,3
R_RE2,DB,RE 2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,RE 1,
R_RE2,S_RE2,T_RE2,RE 2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,49:00:00,49:00:00,A,1,0,0
T_RE1,50:00:00,50:00:00,B,2,0,0
T_RE2,00:30:00,00:30:00,B,1,0,0
T_RE2,00:45:00,00:45:00,C,2,0,0
T_RE2,01:00:00,01:00:00,D,3,0,0
)");
}

constexpr auto const kFwdJourneys = R"(
[2019-05-02 23:00, 2019-05-03 01:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-02 23:00]
       TO: (D, D) [2019-05-03 01:00]
leg 0: (A, A) [2019-05-02 23:00] -> (B, B) [2019-05-03 00:00]
   0: A       A...............................................                               d: 02.05 23:00 [03.05 01:00]  [{name=RE 1, day=2019-05-02, id=T_RE1, src=0}]
   1: B       B............................................... a: 03.05 00:00 [03.05 02:00]
leg 1: (B, B) [2019-05-03 00:00] -> (B, B) [2019-05-03 00:02]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-03 00:30] -> (D, D) [2019-05-03 01:00]
   0: B       B...............................................                                                             d: 02.05 22:30 [03.05 00:30]  RT 03.05 00:30 [03.05 02:30]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   1: C       C............................................... a: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  d: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   2: D       D............................................... a: 02.05 23:00 [03.05 01:00]  RT 03.05 01:00 [03.05 03:00]

)";

}  // namespace

TEST(routing, rt_raptor_forward) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 3});

  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 4} + 9h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_start_time("00:30:00");
  td->set_start_date("20190503");
  td->set_trip_id("T_RE2");
  {
    auto const stop_update = e->mutable_trip_update()->add_stop_time_update();
    stop_update->set_stop_sequence(1U);
    stop_update->mutable_departure()->set_delay(2 * 60 * 60 /* 2h */);
  }

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);

  auto const results =
      raptor_search(tt, &rtt, "A", "D", sys_days{May / 2 / 2019} + 23h);

  EXPECT_EQ(std::string_view{kFwdJourneys}, to_string(tt, &rtt, results));
  ASSERT_FALSE(results.empty());
  auto const& l = results.begin()->legs_.back();
  auto done = hash_set<std::pair<location_idx_t, location_idx_t>>{};
  auto direct = std::vector<routing::journey>{};
  get_direct(tt, &rtt, l.from_, l.to_, routing::query{},
             interval<unixtime_t>{l.dep_time_, l.dep_time_ + 1min},
             direction::kForward, done, direct);
  EXPECT_EQ(R"(
[2019-05-03 00:30, 2019-05-03 01:00]
TRANSFERS: 0
     FROM: (B, B) [2019-05-03 00:30]
       TO: (D, D) [2019-05-03 01:00]
leg 0: (B, B) [2019-05-03 00:30] -> (D, D) [2019-05-03 01:00]
   0: B       B...............................................                                                             d: 02.05 22:30 [03.05 00:30]  RT 03.05 00:30 [03.05 02:30]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   1: C       C............................................... a: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  d: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   2: D       D............................................... a: 02.05 23:00 [03.05 01:00]  RT 03.05 01:00 [03.05 03:00]

)",
            to_string(tt, &rtt, direct));
}

constexpr auto const kBwdJourneys = R"(
[2019-05-02 23:00, 2019-05-03 02:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-02 23:00]
       TO: (D, D) [2019-05-03 01:00]
leg 0: (A, A) [2019-05-02 23:00] -> (B, B) [2019-05-03 00:00]
   0: A       A...............................................                               d: 02.05 23:00 [03.05 01:00]  [{name=RE 1, day=2019-05-02, id=T_RE1, src=0}]
   1: B       B............................................... a: 03.05 00:00 [03.05 02:00]
leg 1: (B, B) [2019-05-03 00:00] -> (B, B) [2019-05-03 00:02]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-03 00:30] -> (D, D) [2019-05-03 01:00]
   0: B       B...............................................                                                             d: 02.05 22:30 [03.05 00:30]  RT 03.05 00:30 [03.05 02:30]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   1: C       C............................................... a: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  d: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   2: D       D............................................... a: 02.05 23:00 [03.05 01:00]  RT 03.05 01:00 [03.05 03:00]

)";

TEST(routing, rt_raptor_backward) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 3});

  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 4} + 9h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_start_time("00:30:00");
  td->set_start_date("20190503");
  td->set_trip_id("T_RE2");
  {
    auto const stop_update = e->mutable_trip_update()->add_stop_time_update();
    stop_update->set_stop_sequence(1U);
    stop_update->mutable_departure()->set_delay(2 * 60 * 60 /* 2h */);
  }

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);

  auto const results =
      raptor_search(tt, &rtt, "D", "A", sys_days{May / 3 / 2019} + 2h,
                    nigiri::direction::kBackward);

  EXPECT_EQ(std::string_view{kBwdJourneys}, to_string(tt, &rtt, results));
  ASSERT_FALSE(results.empty());
  auto const& l = results.begin()->legs_.back();
  auto done = hash_set<std::pair<location_idx_t, location_idx_t>>{};
  auto direct = std::vector<routing::journey>{};
  routing::get_direct(tt, &rtt, l.to_, l.from_, routing::query{},
                      interval<unixtime_t>{l.arr_time_, l.arr_time_ + 1min},
                      direction::kBackward, done, direct);
  EXPECT_EQ(R"(
[2019-05-03 00:30, 2019-05-03 01:00]
TRANSFERS: 0
     FROM: (B, B) [2019-05-03 00:30]
       TO: (D, D) [2019-05-03 01:00]
leg 0: (B, B) [2019-05-03 00:30] -> (D, D) [2019-05-03 01:00]
   0: B       B...............................................                                                             d: 02.05 22:30 [03.05 00:30]  RT 03.05 00:30 [03.05 02:30]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   1: C       C............................................... a: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  d: 02.05 22:45 [03.05 00:45]  RT 03.05 00:45 [03.05 02:45]  [{name=RE 2, day=2019-05-02, id=T_RE2, src=0}]
   2: D       D............................................... a: 02.05 23:00 [03.05 01:00]  RT 03.05 01:00 [03.05 03:00]

)",
            to_string(tt, &rtt, direct));
}

constexpr auto const unscheduled_journeys = R"(
[2019-05-02 23:00, 2019-05-03 01:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-02 23:00]
       TO: (D, D) [2019-05-03 01:00]
leg 0: (A, A) [2019-05-02 23:00] -> (B, B) [2019-05-03 00:00]
   0: A       A...............................................                               d: 02.05 23:00 [03.05 01:00]  [{name=RE 1, day=2019-05-02, id=T_RE1, src=0}]
   1: B       B............................................... a: 03.05 00:00 [03.05 02:00]
leg 1: (B, B) [2019-05-03 00:00] -> (B, B) [2019-05-03 00:02]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-03 00:30] -> (D, D) [2019-05-03 01:00]
   0: B       B...............................................                                                             d: 03.05 00:30 [03.05 02:30]  RT 03.05 00:30 [03.05 02:30]
   1: D       D............................................... a: 03.05 01:00 [03.05 03:00]  RT 03.05 01:00 [03.05 03:00]

)";

TEST(routing, rt_raptor_unscheduled) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 3});

  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2019_y / May / 4} + 9h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const tu = e->mutable_trip_update();
  auto const td = tu->mutable_trip();
  td->set_start_time("02:30:00");
  td->set_start_date("20190503");
  td->set_trip_id("NEW");
  td->set_schedule_relationship(
      transit_realtime::TripDescriptor_ScheduleRelationship_NEW);
  tu->mutable_trip_properties()->set_trip_short_name("Additional");

  {
    auto const stop_update = tu->add_stop_time_update();
    stop_update->set_stop_sequence(1U);
    stop_update->set_stop_id("B");
    stop_update->mutable_departure()->set_time(1556843400);
  }
  {
    auto const stop_update = tu->add_stop_time_update();
    stop_update->set_stop_sequence(2U);
    stop_update->set_stop_id("D");
    stop_update->mutable_arrival()->set_time(1556845200);
  }

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);

  auto const results =
      raptor_search(tt, &rtt, "A", "D", sys_days{May / 2 / 2019} + 23h);

  EXPECT_EQ(std::string_view{unscheduled_journeys},
            to_string(tt, &rtt, results));
}