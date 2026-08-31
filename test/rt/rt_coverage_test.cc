#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "../util.h"

// `rt_timetable::coverage_` - the interval real-time data is known for, so
// that routing can tell whether a query can be influenced by it at all. This
// file covers the maintenance of that interval by the GTFS-RT update path;
// the VDV path is asserted inside `vdv_aus.delay_propagation`
// (`test/rt/vdv_aus_test.cc`), which already has the fixture for it.
//
// What routing does with the coverage is tested in
// `test/routing/rt_drop_test.cc`.

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::to_unix;

namespace {

// Europe/Berlin on 2019-05-01 => UTC+2
//   T1: A 10:00 -> B 11:00  (=  8:00 ->  9:00 UTC)
//   T2: A 14:00 -> B 15:00  (= 12:00 -> 13:00 UTC)
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,

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

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B,2,0,0
T2,14:00:00,14:00:00,A,1,0,0
T2,15:00:00,15:00:00,B,2,0,0
)");
}

constexpr auto const kDay = 2019_y / May / 1;

unixtime_t t(auto&& x) { return unixtime_t{sys_days{kDay} + x}; }

timetable load_tt() {
  auto tt = timetable{};
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  return tt;
}

transit_realtime::FeedMessage msg_header() {
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{kDay} + 8h));
  return msg;
}

transit_realtime::TripUpdate* trip_update(transit_realtime::FeedMessage& msg,
                                          std::string_view trip_id,
                                          std::string_view start_time) {
  auto const e = msg.add_entity();
  e->set_id(std::to_string(msg.entity_size()));
  e->set_is_deleted(false);

  auto const tu = e->mutable_trip_update();
  auto const td = tu->mutable_trip();
  td->set_trip_id(std::string{trip_id});
  td->set_start_date("20190501");
  td->set_start_time(std::string{start_time});
  return tu;
}

// `delay` in minutes, `std::nullopt` = no update for this event.
void delay_msg(timetable const& tt,
               rt_timetable& rtt,
               std::string_view trip_id,
               std::string_view start_time,
               std::optional<int> const dep_delay,
               std::optional<int> const arr_delay) {
  auto msg = msg_header();
  auto const tu = trip_update(msg, trip_id, start_time);
  if (dep_delay.has_value()) {
    auto const stu = tu->add_stop_time_update();
    stu->set_stop_sequence(1U);
    stu->mutable_departure()->set_delay(*dep_delay * 60);
  }
  if (arr_delay.has_value()) {
    auto const stu = tu->add_stop_time_update();
    stu->set_stop_sequence(2U);
    stu->mutable_arrival()->set_delay(*arr_delay * 60);
  }
  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  ASSERT_EQ(1U, stats.total_entities_success_);
}

}  // namespace

TEST(rt, coverage_delay_and_cancel) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  // No real-time data yet.
  EXPECT_TRUE(rtt.coverage_.empty());
  EXPECT_TRUE(rtt.coverage_.from_ == rtt.coverage_.to_);

  // T1 departs 10 minutes late, arrives on time
  // => scheduled + real-time times of T1.
  delay_msg(tt, rtt, "T1", "10:00:00", 10, 0);
  EXPECT_FALSE(rtt.coverage_.empty());
  EXPECT_EQ((interval{t(8h), t(9h + 1min)}), rtt.coverage_);

  // T1 arrives 30 minutes late => extended to the back.
  delay_msg(tt, rtt, "T1", "10:00:00", std::nullopt, 30);
  EXPECT_EQ((interval{t(8h), t(9h + 31min)}), rtt.coverage_);

  // T1 is on time again => coverage is NOT shrunk (we still have real-time
  // information about the [9:00, 9:31) window: the trip is not there).
  delay_msg(tt, rtt, "T1", "10:00:00", 0, 0);
  EXPECT_EQ((interval{t(8h), t(9h + 31min)}), rtt.coverage_);

  // T2 is canceled => extended over its scheduled times (no RT transport).
  {
    auto msg = msg_header();
    trip_update(msg, "T2", "14:00:00")
        ->mutable_trip()
        ->set_schedule_relationship(
            transit_realtime::TripDescriptor_ScheduleRelationship_CANCELED);
    auto const stats =
        rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
    ASSERT_EQ(1U, stats.total_entities_success_);
  }
  EXPECT_EQ((interval{t(8h), t(13h + 1min)}), rtt.coverage_);
}

// A trip that is EARLIER than scheduled extends the coverage to the front.
TEST(rt, coverage_negative_delay) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  delay_msg(tt, rtt, "T1", "10:00:00", -10, -5);
  EXPECT_EQ((interval{t(7h + 50min), t(9h + 1min)}), rtt.coverage_);
}

// Cancellation of a run that already has an RT transport: both the RT and the
// scheduled times are covered.
TEST(rt, coverage_cancel_rt_run) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  delay_msg(tt, rtt, "T1", "10:00:00", 20, 20);
  EXPECT_EQ((interval{t(8h), t(9h + 21min)}), rtt.coverage_);

  {
    auto msg = msg_header();
    trip_update(msg, "T1", "10:00:00")
        ->mutable_trip()
        ->set_schedule_relationship(
            transit_realtime::TripDescriptor_ScheduleRelationship_CANCELED);
    auto const stats =
        rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
    ASSERT_EQ(1U, stats.total_entities_success_);
  }
  EXPECT_EQ((interval{t(8h), t(9h + 21min)}), rtt.coverage_);
}

// Added trips are created with placeholder (zero) times that are filled in
// afterwards. The coverage must not be anchored at the base day's midnight.
TEST(rt, coverage_added_trip) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  auto msg = msg_header();
  auto const tu = trip_update(msg, "T_ADDED", "22:00:00");
  tu->mutable_trip()->set_schedule_relationship(
      transit_realtime::TripDescriptor_ScheduleRelationship_ADDED);
  tu->mutable_trip()->set_route_id("R1");

  auto const dep = tu->add_stop_time_update();
  dep->set_stop_sequence(1U);
  dep->set_stop_id("A");
  dep->mutable_departure()->set_time(
      to_unix<std::int64_t>(sys_days{kDay} + 20h));

  auto const arr = tu->add_stop_time_update();
  arr->set_stop_sequence(2U);
  arr->set_stop_id("B");
  arr->mutable_arrival()->set_time(
      to_unix<std::int64_t>(sys_days{kDay} + 20h + 30min));

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  ASSERT_EQ(1U, stats.total_entities_success_);

  EXPECT_EQ((interval{t(20h), t(20h + 31min)}), rtt.coverage_);
}

TEST(rt, coverage_extend) {
  auto rtt = rt_timetable{};
  EXPECT_TRUE(rtt.coverage_.empty());

  // Empty intervals are ignored.
  rtt.extend_coverage(interval<unixtime_t>{});
  rtt.extend_coverage(interval{t(12h), t(10h)});  // reversed
  EXPECT_TRUE(rtt.coverage_.empty());

  rtt.extend_coverage(interval{t(10h), t(12h)});
  EXPECT_FALSE(rtt.coverage_.empty());
  EXPECT_EQ((interval{t(10h), t(12h)}), rtt.coverage_);

  // Contained intervals do not shrink the coverage.
  rtt.extend_coverage(interval{t(10h + 30min), t(11h)});
  rtt.extend_coverage(t(11h));
  EXPECT_EQ((interval{t(10h), t(12h)}), rtt.coverage_);

  // Overlapping + disjoint intervals extend it in both directions.
  rtt.extend_coverage(interval{t(9h), t(11h)});
  rtt.extend_coverage(interval{t(20h), t(21h)});
  EXPECT_EQ((interval{t(9h), t(21h)}), rtt.coverage_);

  // A single point covers exactly one minute.
  auto rtt1 = rt_timetable{};
  rtt1.extend_coverage(t(6h));
  EXPECT_EQ((interval{t(6h), t(6h + 1min)}), rtt1.coverage_);
  EXPECT_TRUE(rtt1.coverage_.contains(t(6h)));
  EXPECT_FALSE(rtt1.coverage_.contains(t(6h + 1min)));
}

// `rt_timetable::affects()` is the single predicate both real-time drop
// decisions go through: the drivers (`raptor_search.cc` / `pong.cc`) and
// `raptor<>::execute()` per start time. `test/routing/rt_drop_test.cc` covers
// the intervals they feed it. Both intervals are half-open.
TEST(rt, coverage_needs_rt) {
  auto rtt = rt_timetable{};

  // No real-time data at all: nothing can be affected ...
  EXPECT_FALSE(rtt.affects(interval{t(0h), t(24h)}, 0U));
  // ... except a profile reading time dependent footpaths, which `coverage_`
  // does not track.
  EXPECT_TRUE(rtt.affects(interval{t(0h), t(24h)}, 1U));

  rtt.extend_coverage(interval{t(10h), t(12h)});

  // Disjoint on either side.
  EXPECT_FALSE(rtt.affects(interval{t(8h), t(10h)}, 0U));
  EXPECT_FALSE(rtt.affects(interval{t(12h), t(14h)}, 0U));

  // Touching by exactly one minute on either side.
  EXPECT_TRUE(rtt.affects(interval{t(8h), t(10h + 1min)}, 0U));
  EXPECT_TRUE(rtt.affects(interval{t(11h + 59min), t(14h)}, 0U));

  // Contained, containing, overlapping.
  EXPECT_TRUE(rtt.affects(interval{t(10h + 30min), t(11h)}, 0U));
  EXPECT_TRUE(rtt.affects(interval{t(0h), t(24h)}, 0U));
  EXPECT_TRUE(rtt.affects(interval{t(9h), t(11h)}, 0U));

  // Coverage that starts exactly where the search stops looking, and vice
  // versa: half-open on both sides, so neither touches.
  EXPECT_FALSE(rtt.affects(interval{t(12h), t(12h + 1min)}, 0U));
  EXPECT_FALSE(rtt.affects(interval{t(9h), t(10h)}, 0U));
}
