#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_alert.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "./util.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
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
B,B,,0.01,1.01,,
C,C,,0.02,1.02,,
D,D,,0.03,1.03,,
E,E,,0.04,1.04,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:10:00,10:10:00,B,2,0,0
T1,10:20:00,10:20:00,C,3,0,0
T1,10:30:00,10:30:00,D,4,0,0
T1,10:40:00,10:40:00,E,5,0,0
)");
}

constexpr auto const kBaseDay = date::sys_days{2019_y / May / 1};
constexpr auto const kArr = event_type::kArr;
constexpr auto const kDep = event_type::kDep;
constexpr auto const kNoRtData = rt_data_state::kNoRtData;
constexpr auto const kInconsistent = rt_data_state::kInconsistent;
constexpr auto const kPropagated = rt_data_state::kPropagated;
constexpr auto const kPredicted = rt_data_state::kPredicted;

using states_t = std::vector<rt_data_state>;

// Static times are given in Europe/Berlin (UTC+2).
unixtime_t scheduled(std::chrono::minutes const local_time) {
  return kBaseDay + local_time - 2h;
}

transit_realtime::FeedMessage to_msg(
    std::vector<test::trip::delay> const& delays) {
  return test::to_feed_msg({{.trip_id_ = "T1", .delays_ = delays}},
                           kBaseDay + 7h);
}

frun get_frun(timetable const& tt, rt_timetable const& rtt) {
  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id("T1");
  td.set_start_date("20190501");
  td.set_start_time("10:00:00");
  auto const [r, _] =
      rt::gtfsrt_resolve_run(kBaseDay, tt, &rtt, source_idx_t{0}, td);
  return frun{tt, &rtt, r};
}

// Event order: A dep, B arr, B dep, C arr, C dep, D arr, D dep, E arr
states_t get_states(timetable const& tt, rt_timetable const& rtt) {
  auto const fr = get_frun(tt, rtt);
  auto states = states_t{};
  for (auto const rs : fr) {
    if (rs.stop_idx_ != 0U) {
      states.push_back(rs.data_state(event_type::kArr));
    }
    if (rs.stop_idx_ != fr.size() - 1U) {
      states.push_back(rs.data_state(event_type::kDep));
    }
  }
  return states;
}

void expect_no_delay(frun const& fr) {
  for (auto const rs : fr) {
    if (rs.stop_idx_ != 0U) {
      EXPECT_EQ(0_minutes, rs.delay(event_type::kArr));
    }
    if (rs.stop_idx_ != fr.size() - 1U) {
      EXPECT_EQ(0_minutes, rs.delay(event_type::kDep));
    }
  }
}

}  // namespace

TEST(rt, data_state_static_run) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const rtt = rt::create_rt_timetable(tt, kBaseDay);

  auto const fr = get_frun(tt, rtt);
  ASSERT_TRUE(fr.valid());
  ASSERT_FALSE(fr.is_rt());
  EXPECT_EQ((states_t{kNoRtData, kNoRtData, kNoRtData, kNoRtData, kNoRtData,
                      kNoRtData, kNoRtData, kNoRtData}),
            get_states(tt, rtt));
}

TEST(rt, data_state_rt_transport_from_alert) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto stats = statistics{.total_entities_ = 0, .feed_timestamp_ = {}};

  // An alert for the trip creates an RT transport without any times.
  auto a = transit_realtime::Alert{};
  auto const td = a.add_informed_entity()->mutable_trip();
  td->set_trip_id("T1");
  td->set_start_date("20190501");
  td->set_start_time("10:00:00");
  handle_alert(kBaseDay, tt, rtt, source_idx_t{0}, "tag", a, stats);

  auto const fr = get_frun(tt, rtt);
  ASSERT_TRUE(fr.is_rt());
  EXPECT_EQ((states_t{kNoRtData, kNoRtData, kNoRtData, kNoRtData, kNoRtData,
                      kNoRtData, kNoRtData, kNoRtData}),
            get_states(tt, rtt));
  expect_no_delay(fr);
}

TEST(rt, data_state_predicted_and_propagated) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto const stats = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .delay_minutes_ = 5},
              {.seq_ = 3U, .ev_type_ = kArr, .delay_minutes_ = 7}}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  EXPECT_EQ((states_t{kPredicted, kPropagated, kPropagated, kPredicted,
                      kPropagated, kPropagated, kPropagated, kPropagated}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h + 15min), fr[1].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 27min), fr[2].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 47min), fr[4].time(event_type::kArr));
}

TEST(rt, data_state_first_update_not_at_first_stop) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto const stats = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 3U, .ev_type_ = kArr, .delay_minutes_ = 5}}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  EXPECT_EQ((states_t{kNoRtData, kNoRtData, kNoRtData, kPredicted, kPropagated,
                      kPropagated, kPropagated, kPropagated}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h), fr[0].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 10min), fr[1].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 10min), fr[1].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 25min), fr[2].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 45min), fr[4].time(event_type::kArr));
}

TEST(rt, data_state_gtfsrt_no_data) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto const stats = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .delay_minutes_ = 5},
              {.seq_ = 3U, .ev_type_ = kArr, .no_data_ = true},
              {.seq_ = 5U, .ev_type_ = kArr, .delay_minutes_ = 3}}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  // NO_DATA stops propagation and is propagated itself to subsequent stops
  // without data. The times of NO_DATA stops are the scheduled times.
  EXPECT_EQ((states_t{kPredicted, kPropagated, kPropagated, kNoRtData,
                      kNoRtData, kNoRtData, kNoRtData, kPredicted}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h + 15min), fr[1].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 20min), fr[2].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 20min), fr[2].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 30min), fr[3].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 30min), fr[3].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 43min), fr[4].time(event_type::kArr));
}

TEST(rt, data_state_gtfsrt_no_data_resets_previous_update) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .delay_minutes_ = 5}}));
  EXPECT_EQ((states_t{kPredicted, kPropagated, kPropagated, kPropagated,
                      kPropagated, kPropagated, kPropagated, kPropagated}),
            get_states(tt, rtt));

  rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .no_data_ = true}}));
  EXPECT_EQ((states_t{kNoRtData, kNoRtData, kNoRtData, kNoRtData, kNoRtData,
                      kNoRtData, kNoRtData, kNoRtData}),
            get_states(tt, rtt));
  expect_no_delay(get_frun(tt, rtt));
}

TEST(rt, data_state_inconsistent_lower_bound) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // B arrival (on time, 10:10) would be before A departure (10:20).
  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto const stats = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .delay_minutes_ = 20},
              {.seq_ = 2U, .ev_type_ = kArr, .delay_minutes_ = 0}}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  EXPECT_EQ((states_t{kPredicted, kInconsistent, kInconsistent, kPropagated,
                      kPropagated, kPropagated, kPropagated, kPropagated}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h + 20min), fr[1].time(event_type::kArr));
}

TEST(rt, data_state_inconsistent_after_no_data) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // A +30min is propagated to B. C has NO_DATA -> scheduled time 10:20 is
  // before B's propagated departure 10:40 and has to be adjusted.
  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto const stats = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .delay_minutes_ = 30},
              {.seq_ = 3U, .ev_type_ = kArr, .no_data_ = true}}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  EXPECT_EQ((states_t{kPredicted, kPropagated, kPropagated, kInconsistent,
                      kInconsistent, kInconsistent, kInconsistent, kNoRtData}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h + 40min), fr[2].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 40min), fr[3].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 40min), fr[4].time(event_type::kArr));
}

TEST(rt, data_state_arrival_and_departure_differ) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // A +20min is propagated to B's arrival (10:30). B's departure (on time,
  // 10:10) is adjusted to 10:30.
  // C: on time (propagated delay 0) would be 10:20 -> adjusted to 10:30.
  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  auto const stats = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 1U, .ev_type_ = kDep, .delay_minutes_ = 20},
              {.seq_ = 2U, .ev_type_ = kDep, .delay_minutes_ = 0}}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  EXPECT_EQ((states_t{kPredicted, kPropagated, kInconsistent, kInconsistent,
                      kInconsistent, kPropagated, kPropagated, kPropagated}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h + 30min), fr[1].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 30min), fr[1].time(event_type::kDep));
  EXPECT_EQ(scheduled(10h + 30min), fr[2].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 30min), fr[3].time(event_type::kArr));
}

TEST(rt, data_state_incremental_update_keeps_previous_stops) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, kBaseDay);
  rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 3U, .ev_type_ = kArr, .delay_minutes_ = 5}}));
  rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      to_msg({{.seq_ = 4U, .ev_type_ = kArr, .delay_minutes_ = 10}}));

  // Stops before the first stop time update are left untouched.
  EXPECT_EQ((states_t{kNoRtData, kNoRtData, kNoRtData, kPredicted, kPropagated,
                      kPredicted, kPropagated, kPropagated}),
            get_states(tt, rtt));

  auto const fr = get_frun(tt, rtt);
  EXPECT_EQ(scheduled(10h + 25min), fr[2].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 40min), fr[3].time(event_type::kArr));
  EXPECT_EQ(scheduled(10h + 50min), fr[4].time(event_type::kArr));
}
