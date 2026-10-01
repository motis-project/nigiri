#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "../raptor_search.h"
#include "../util.h"

// A GTFS-RT track change moves a stop of an rt transport to another platform,
// i.e. to another location. The routing builds its rt marks from
// `location_rt_scan_` (see rt_timetable::register_scan), so the transport has
// to be listed under the platform it actually uses, otherwise a search that
// only seeds that platform cannot board it. Range queries seed the exact
// platform of the real-time departure (get_starts), which is what these tests
// use; a single-start query seeds every platform of the station and hides the
// problem.

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;
using nigiri::test::to_unix;

namespace {

// Europe/Berlin on 2019-05-01 => UTC+2
//   T1: A 10:00 -> S1 10:30/10:31 -> B 11:00 (S1, S2 are platforms of S)
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,,
S,S,,0.02,1.03,,1,
S1,S platform 1,,0.02,1.03,,0,S
S2,S platform 2,,0.0201,1.0301,,0,S
B,B,,0.04,1.05,,,

# calendar_dates.txt
service_id,date,exception_type
SRV,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,SRV,T1,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:31:00,S1,2,0,0
T1,11:00:00,11:00:00,B,3,0,0
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

location_idx_t location_of(timetable const& tt, std::string_view const id) {
  return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
}

// One trip update for T1: `dep_delay` minutes at A (stop_sequence 1) and,
// at S (stop_sequence 2), either a platform assignment or a plain delay-only
// update (which resets a previous assignment, see gtfsrt_update.cc).
void update(timetable const& tt,
            rt_timetable& rtt,
            std::optional<int> const dep_delay,
            std::optional<std::string> const assigned_stop) {
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{kDay} + 7h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);
  auto const tu = e->mutable_trip_update();
  auto const td = tu->mutable_trip();
  td->set_trip_id("T1");
  td->set_start_date("20190501");
  td->set_start_time("10:00:00");

  if (dep_delay.has_value()) {
    auto const stu = tu->add_stop_time_update();
    stu->set_stop_sequence(1U);
    stu->mutable_departure()->set_delay(*dep_delay * 60);
  }
  {
    auto const stu = tu->add_stop_time_update();
    stu->set_stop_sequence(2U);
    if (assigned_stop.has_value()) {
      stu->mutable_stop_time_properties()->set_assigned_stop_id(*assigned_stop);
    } else {
      stu->mutable_arrival()->set_delay(dep_delay.value_or(0) * 60);
    }
  }

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  ASSERT_EQ(1U, stats.total_entities_success_);
}

bool on_scan_list(rt_timetable const& rtt,
                  location_idx_t const l,
                  rt_transport_idx_t const rt_t) {
  for (auto const x : rtt.location_rt_scan_[l]) {
    if (x == rt_t) {
      return true;
    }
  }
  return false;
}

// Range query from the exact platform `from` to B over [08:00, 09:00) UTC
// (= 10:00-11:00 local, the hour T1 departs in). Returns the departure times
// of the journeys found.
std::vector<unixtime_t> departures_from(timetable const& tt,
                                        rt_timetable const& rtt,
                                        std::string_view const from) {
  auto const results =
      raptor_search(tt, &rtt, from, "B", interval<unixtime_t>{t(8h), t(9h)},
                    direction::kForward);
  auto v = std::vector<unixtime_t>{};
  for (auto const& j : results) {
    EXPECT_EQ(0U, j.transfers_);
    EXPECT_EQ(location_of(tt, from), j.legs_.front().from_);
    v.push_back(j.start_time_);
  }
  return v;
}

}  // namespace

// Delay at A and the platform change at S arrive in the same update: the
// first delayed event registers the transport for the real-time scan, the
// platform change afterwards must reach the scan list as well.
TEST(rt, scan_track_change_same_update) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  update(tt, rtt, 5, "S2");

  auto const rt_t = rt_transport_idx_t{0U};
  ASSERT_EQ(1U, rtt.n_rt_transports());
  EXPECT_FALSE(rtt.is_unchanged(rt_t));
  EXPECT_TRUE(on_scan_list(rtt, location_of(tt, "S2"), rt_t));

  // boards at S2 at 10:36 local = 08:36 UTC, nothing departs from S1
  EXPECT_EQ((std::vector{t(8h + 36min)}), departures_from(tt, rtt, "S2"));
  EXPECT_TRUE(departures_from(tt, rtt, "S1").empty());
}

// The delay registers the transport in one update, the platform change
// arrives in a later one.
TEST(rt, scan_track_change_later_update) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  update(tt, rtt, 5, std::nullopt);
  EXPECT_EQ((std::vector{t(8h + 36min)}), departures_from(tt, rtt, "S1"));

  update(tt, rtt, 5, "S2");
  EXPECT_TRUE(on_scan_list(rtt, location_of(tt, "S2"), rt_transport_idx_t{0U}));
  EXPECT_EQ((std::vector{t(8h + 36min)}), departures_from(tt, rtt, "S2"));
  EXPECT_TRUE(departures_from(tt, rtt, "S1").empty());
}

// The platform change is reverted: the stop goes back to its scheduled
// platform, which has to be on the scan list even if the transport was first
// registered while it used the other one.
TEST(rt, scan_track_change_reset) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  update(tt, rtt, 5, "S2");
  EXPECT_EQ((std::vector{t(8h + 36min)}), departures_from(tt, rtt, "S2"));

  update(tt, rtt, 5, std::nullopt);  // delay-only update at S => reset
  EXPECT_TRUE(on_scan_list(rtt, location_of(tt, "S1"), rt_transport_idx_t{0U}));
  EXPECT_EQ((std::vector{t(8h + 36min)}), departures_from(tt, rtt, "S1"));
  EXPECT_TRUE(departures_from(tt, rtt, "S2").empty());
}

// A platform change without any delay: the transport only becomes deviating
// in finalize_rt_transport() (stop sequence differs from the schedule).
TEST(rt, scan_track_change_without_delay) {
  auto const tt = load_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{kDay});

  update(tt, rtt, std::nullopt, "S2");
  EXPECT_FALSE(rtt.is_unchanged(rt_transport_idx_t{0U}));
  EXPECT_TRUE(on_scan_list(rtt, location_of(tt, "S2"), rt_transport_idx_t{0U}));
  EXPECT_EQ((std::vector{t(8h + 31min)}), departures_from(tt, rtt, "S2"));
  EXPECT_TRUE(departures_from(tt, rtt, "S1").empty());
}
