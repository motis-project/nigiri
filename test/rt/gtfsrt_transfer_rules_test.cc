#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "./util.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// Real-time updates in the presence of transfers.txt rules.
//
// Network (2019-05-01, Europe/Berlin). S is a station with the platforms
// S1..S4 a few meters apart (mutually equivalent -> valid track changes).
// Default change time at a stop: 2 min, S1 <-> S2 walk: a few minutes.
//
//   F  (RF): A  10:00 -> S1 10:30         feeder
//   F0 (RF): A2 09:00 -> S3 09:30         a second RF trip, scheduled at S3
//   G  (RG): S2 10:40 -> B 11:00          10 min after F: fine by default
//   GL (RG): S2 11:10 -> B 11:30          fallback if G cannot be reached
//   H  (RH): S1 10:31 -> C 11:00          1 min after F: only with a rule
//   HL (RH): S1 11:01 -> C 11:30          fallback if H cannot be reached
//   F2 (RF): S1 10:36 -> D 11:06          a second RF trip meeting F at S1
//
// P and Q meet twice, at S and at the station X (platforms X1, X2):
//   P  (RP): A  12:00 -> S1 12:30 -> X1 12:50
//   Q  (RQ): S2 12:40 -> X2 13:00 -> B 13:20
//
// Every test states its own transfers.txt rows and real-time messages.
constexpr auto const kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,50.0,6.0,,,
A2,A2,,50.0,6.1,,,
S,S,,50.0,6.5,,1,
S1,S1,,50.0001,6.5,,,S
S2,S2,,50.0002,6.5,,,S
S3,S3,,50.0003,6.5,,,S
S4,S4,,50.0004,6.5,,,S
B,B,,50.0,7.0,,,
C,C,,50.0,7.5,,,
D,D,,50.0,8.0,,,
X,X,,50.0,6.8,,1,
X1,X1,,50.0001,6.8,,,X
X2,X2,,50.0002,6.8,,,X

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RF,AG,RF,,,3
RG,AG,RG,,,3
RH,AG,RH,,,3
RP,AG,RP,,,3
RQ,AG,RQ,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RF,X,F,,
RF,X,F0,,
RF,X,F2,,
RG,X,G,,
RG,X,GL,,
RH,X,H,,
RH,X,HL,,
RP,X,P,,
RQ,X,Q,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
F,10:00:00,10:00:00,A,1,0,0
F,10:30:00,10:30:00,S1,2,0,0
F0,09:00:00,09:00:00,A2,1,0,0
F0,09:30:00,09:30:00,S3,2,0,0
F2,10:36:00,10:36:00,S1,1,0,0
F2,11:06:00,11:06:00,D,2,0,0
G,10:40:00,10:40:00,S2,1,0,0
G,11:00:00,11:00:00,B,2,0,0
GL,11:10:00,11:10:00,S2,1,0,0
GL,11:30:00,11:30:00,B,2,0,0
H,10:31:00,10:31:00,S1,1,0,0
H,11:00:00,11:00:00,C,2,0,0
HL,11:01:00,11:01:00,S1,1,0,0
HL,11:30:00,11:30:00,C,2,0,0
P,12:00:00,12:00:00,A,1,0,0
P,12:30:00,12:30:00,S1,2,0,0
P,12:50:00,12:50:00,X1,3,0,0
Q,12:40:00,12:40:00,S2,1,0,0
Q,13:00:00,13:00:00,X2,2,0,0
Q,13:20:00,13:20:00,B,3,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
{}
)";

constexpr auto const kDay = date::sys_days{2019_y / May / 1};

timetable load(std::string_view const transfers) {
  auto tt = timetable{};
  tt.date_range_ = {kDay, kDay + date::days{1}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable(
      {}, source_idx_t{0},
      loader::mem_dir::read(fmt::format(fmt::runtime(kFeed), transfers)), tt);
  loader::finalize(tt);
  return tt;
}

unixtime_t t(char const* s) { return parse_time_tz(s, "%Y-%m-%d %H:%M %Z"); }

// One stop time update, stated the way real feeds do it.
struct stu {
  unsigned seq_;
  std::string stop_id_;  // the scheduled stop id
  std::optional<std::string> assigned_{};  // track change
  std::optional<int> arr_delay_{};  // minutes
  std::optional<int> dep_delay_{};  // minutes
  bool send_seq_{true};
  bool send_stop_id_{true};
  // Track change stated the old way: stop_id differs from the schedule and
  // there is no stop_time_properties.
  bool assigned_as_stop_id_{false};
};

struct trip_update {
  std::string trip_id_;
  std::vector<stu> stus_;
};

transit_realtime::FeedMessage to_msg(std::vector<trip_update> const& updates) {
  auto msg = transit_realtime::FeedMessage{};
  auto* const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(test::to_unix(kDay + 8h));

  auto id = 0U;
  for (auto const& u : updates) {
    auto* const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));
    auto* const tu = e->mutable_trip_update();
    tu->mutable_trip()->set_trip_id(u.trip_id_);
    tu->mutable_trip()->set_start_date("20190501");
    for (auto const& s : u.stus_) {
      auto* const x = tu->add_stop_time_update();
      if (s.send_seq_) {
        x->set_stop_sequence(s.seq_);
      }
      if (s.assigned_as_stop_id_) {
        x->set_stop_id(*s.assigned_);
      } else {
        if (s.send_stop_id_) {
          x->set_stop_id(s.stop_id_);
        }
        if (s.assigned_.has_value()) {
          x->mutable_stop_time_properties()->set_assigned_stop_id(*s.assigned_);
        }
      }
      if (s.arr_delay_.has_value()) {
        x->mutable_arrival()->set_delay(*s.arr_delay_ * 60);
      }
      if (s.dep_delay_.has_value()) {
        x->mutable_departure()->set_delay(*s.dep_delay_ * 60);
      }
    }
  }
  return msg;
}

void update(timetable const& tt,
            rt_timetable& rtt,
            std::vector<trip_update> const& updates) {
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", to_msg(updates));
}

// The platform a trip stops at, as everything outside the routing sees it.
std::string platform(timetable const& tt,
                     rt_timetable const& rtt,
                     std::string const& trip_id,
                     stop_idx_t const stop_idx) {
  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id(trip_id);
  td.set_start_date("20190501");
  auto const [r, _] =
      rt::gtfsrt_resolve_run(kDay, tt, &rtt, source_idx_t{0}, td);
  if (!r.valid()) {
    return "?";
  }
  auto const fr = rt::frun{tt, &rtt, r};
  auto const l = fr[stop_idx].get_location_idx();
  if (l >= tt.n_locations()) {
    return "out-of-range";
  }
  return std::string{
      tt.locations_.ids_[tt.locations_.get_attribute_idx(l)].view()};
}

std::string leg_stop_id(timetable const& tt, location_idx_t const l) {
  if (l >= tt.n_locations()) {
    return "out-of-range";
  }
  return std::string{
      tt.locations_.ids_[tt.locations_.get_attribute_idx(l)].view()};
}

constexpr auto const kAtoB = std::pair{"A", "B"};
constexpr auto const kAtoC = std::pair{"A", "C"};

// Stop-to-stop query the way a server asks it: a stop stands for its station,
// so everything below it (platforms, virtual locations) is a start / target.
pareto_set<routing::journey> search(
    timetable const& tt,
    rt_timetable const& rtt,
    std::pair<char const*, char const*> const& od,
    char const* at,
    direction const dir = direction::kForward) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = t(at),
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .start_ = {{tt.locations_.location_id_to_idx_.at({od.first, src}),
                  0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({od.second, src}),
                        0_minutes, 0U}}};
  return raptor_search(tt, &rtt, std::move(q), dir);
}

unixtime_t arrival(timetable const& tt,
                   rt_timetable const& rtt,
                   std::pair<char const*, char const*> const& od,
                   char const* at = "2019-05-01 10:00 Europe/Berlin") {
  auto const res = search(tt, rtt, od, at);
  EXPECT_EQ(1U, res.size());
  return res.size() == 0U ? unixtime_t{} : begin(res)->dest_time_;
}

// not evaluated at static init time: the time zone database is not up yet
#define kGFromF t("2019-05-01 11:00 Europe/Berlin") /* A -F-> S -G-> B */
#define kGLFromF t("2019-05-01 11:30 Europe/Berlin") /* A -F-> S -GL-> B */
#define kHFromF t("2019-05-01 11:00 Europe/Berlin") /* A -F-> S -H-> C */
#define kHLFromF t("2019-05-01 11:30 Europe/Berlin") /* A -F-> S -HL-> C */

}  // namespace

// ===========================================================================
// 1. Delays at stops that carry a qualified rule (= virtual locations).
//    No track change involved: the rule has to keep applying and the update
//    has to arrive, however the feed names the stop.
// ===========================================================================

// Sanity: the schedule alone behaves as the rule says.
TEST(rt_transfer_rules, guard_schedule_baseline) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto const rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));  // 10 min < 15 min rule
}

// G waits: 10:40 -> 10:50. 20 min >= 15 min rule -> G is reachable, and its
// delay reaches B. The update names the stop by stop_id and stop_sequence.
TEST(rt_transfer_rules, delay_at_rule_stop_with_stop_id_and_sequence) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = 10}}}});
  EXPECT_EQ(t("2019-05-01 11:10 Europe/Berlin"), arrival(tt, rtt, kAtoB));
  EXPECT_EQ("S2", platform(tt, rtt, "G", 0U));
}

// Same, but the feed only states stop_id (no stop_sequence).
TEST(rt_transfer_rules, delay_at_rule_stop_with_stop_id_only) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"G",
           {{.seq_ = 1U,
             .stop_id_ = "S2",
             .dep_delay_ = 10,
             .send_seq_ = false}}}});
  EXPECT_EQ(t("2019-05-01 11:10 Europe/Berlin"), arrival(tt, rtt, kAtoB));
}

// Same, but the feed only states stop_sequence (no stop_id).
TEST(rt_transfer_rules, guard_delay_at_rule_stop_with_sequence_only) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"G",
           {{.seq_ = 1U,
             .stop_id_ = "S2",
             .dep_delay_ = 10,
             .send_stop_id_ = false}}}});
  EXPECT_EQ(t("2019-05-01 11:10 Europe/Berlin"), arrival(tt, rtt, kAtoB));
}

// A slower rule that the schedule satisfies (10 min >= 8 min) becomes binding
// once the feeder is late: 5 min < 8 min although the walk alone would do.
TEST(rt_transfer_rules, delay_makes_slower_rule_binding) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 5}}}});
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// A faster rule (0 min, route pair) keeps a connection the default change
// time (2 min) would break - also after both trips moved: F arrives 10:33,
// H leaves 10:33 and reaches C two minutes late.
TEST(rt_transfer_rules, delay_keeps_faster_rule) {
  auto const tt = load("S1,S1,2,120,,,,\nS1,S1,2,0,RF,RH,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 3}}},
          {"H", {{.seq_ = 1U, .stop_id_ = "S1", .dep_delay_ = 2}}}});
  EXPECT_EQ(t("2019-05-01 11:02 Europe/Berlin"), arrival(tt, rtt, kAtoC));
}

// A forbidden trip pair stays forbidden however late the connecting trip is.
TEST(rt_transfer_rules, delay_keeps_forbidden_pair) {
  auto const tt = load("S,S,2,120,,,,\nS,S,3,,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = 5}}}});
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  EXPECT_EQ("S2", platform(tt, rtt, "G", 0U));
  // ... and the delay arrived: G now leaves S2 at 10:45, reaches B at 11:05
  EXPECT_EQ(
      t("2019-05-01 11:05 Europe/Berlin"),
      arrival(tt, rtt, std::pair{"S2", "B"}, "2019-05-01 10:41 Europe/Berlin"));
}

// A skipped stop named by stop_id only.
TEST(rt_transfer_rules, skipped_rule_stop_with_stop_id_only) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,0,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto msg =
      to_msg({{"G", {{.seq_ = 1U, .stop_id_ = "S2", .send_seq_ = false}}}});
  msg.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->set_schedule_relationship(
          transit_realtime::
              TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", msg);
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));  // G cannot be boarded at S2
}

// ===========================================================================
// 2. Track changes: rules bound to the TRIP (or its route) at station level
//    follow the trip to its new platform.
// ===========================================================================

// slower: F -> G takes 15 min, also from the new platform
TEST(rt_transfer_rules, track_change_keeps_slower_trip_rule) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// faster: a timed transfer F -> H (H waits, 0 min) survives F moving away
// from H's platform, where the walk alone would take longer than 1 min
TEST(rt_transfer_rules, track_change_keeps_timed_trip_rule) {
  auto const tt = load("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
}

// Without the rule the same track change does break the connection.
TEST(rt_transfer_rules,
     guard_track_change_without_rule_breaks_tight_connection) {
  auto const tt = load("S1,S1,2,0,,,,");  // 0 min at S1 for everyone
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kHLFromF, arrival(tt, rtt, kAtoC));
}

// forbidden: the trip pair stays forbidden from the new platform
TEST(rt_transfer_rules, track_change_keeps_forbidden_trip_pair) {
  auto const tt = load("S,S,2,120,,,,\nS,S,3,,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// The rule also holds if its TARGET trip changes platform.
TEST(rt_transfer_rules, track_change_of_target_trip_keeps_rule) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "G", 0U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  // G itself is still usable from its new platform
  EXPECT_EQ(kGFromF, arrival(tt, rtt, std::pair{"S4", "B"},
                             "2019-05-01 10:35 Europe/Berlin"));
}

// ... and if both change platform (two real-time locations meet).
TEST(rt_transfer_rules, track_change_of_both_trips_keeps_rule) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ("S4", platform(tt, rtt, "G", 0U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// Timed transfer with both trips moved to platforms nobody was scheduled at.
TEST(rt_transfer_rules, track_change_of_both_trips_keeps_timed_rule) {
  auto const tt = load("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"H", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S2"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "F", 1U));
  EXPECT_EQ("S2", platform(tt, rtt, "H", 0U));
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
}

// Route-qualified station rule; another RF trip (F0) is scheduled at S3, so
// the location F needs there already exists.
TEST(rt_transfer_rules, track_change_keeps_route_rule_existing_location) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,RF,RG,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// Route-qualified station rule, new platform S4 never saw an RF trip.
TEST(rt_transfer_rules, track_change_keeps_route_rule_new_location) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,RF,RG,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// The track change stated the old way: a different stop_id, no
// stop_time_properties.
TEST(rt_transfer_rules, track_change_via_stop_id_keeps_rule) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F",
           {{.seq_ = 2U,
             .stop_id_ = "S1",
             .assigned_ = "S3",
             .assigned_as_stop_id_ = true}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// ===========================================================================
// 3. Track changes: rules bound to the PLATFORM stay with the platform.
// ===========================================================================

// leaving a slow platform: S1 -> S2 takes 15 min for F, S3 -> S2 does not
TEST(rt_transfer_rules, track_change_leaves_platform_rule_behind) {
  auto const tt = load("S1,S2,2,120,,,,\nS1,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
}

// entering a slow platform (trip-qualified: nobody was split off at S3): the
// rule only becomes binding through the track change
TEST(rt_transfer_rules, track_change_makes_trip_platform_rule_binding) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// entering a slow platform (route-qualified: F0 is already split off at S3)
TEST(rt_transfer_rules, track_change_makes_route_platform_rule_binding) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,2,900,RF,,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// entering a forbidden platform pair
TEST(rt_transfer_rules, track_change_makes_forbidden_platform_rule_binding) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,3,,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  // no transfer S3 -> S2 for F at all, and G and GL both leave from S2: what
  // is left is the next feeder, P, two hours later
  EXPECT_EQ(t("2019-05-01 13:20 Europe/Berlin"), arrival(tt, rtt, kAtoB));
}

// entering a fast platform: 0 min at S3 for RF -> RH, and H moves there too
TEST(rt_transfer_rules, track_change_makes_faster_platform_rule_binding) {
  auto const tt = load("S3,S3,2,120,,,,\nS3,S3,2,0,RF,RH,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kHLFromF, arrival(tt, rtt, kAtoC));  // S1: 1 min < 2 min default
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"H", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ("S3", platform(tt, rtt, "H", 0U));
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
}

// The specific rule wins on the new platform as well: the station says 15 min
// for RF -> RG, the trip pair F -> G is timed.
TEST(rt_transfer_rules, track_change_keeps_rule_precedence) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,RF,RG,,\nS,S,1,,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));

  // and delaying G beyond reach of nothing: still G, 5 min later
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = 5}}}});
  EXPECT_EQ(t("2019-05-01 11:05 Europe/Berlin"), arrival(tt, rtt, kAtoB));
}

// ===========================================================================
// 4. Track change and delay together, reverting, searching the other way.
// ===========================================================================

// 8 min rule, satisfied by the schedule; F changes platform AND is 5 min late
TEST(rt_transfer_rules, track_change_and_delay_make_rule_binding) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));  // 10 min >= 8 min
  update(
      tt, rtt,
      {{"F",
        {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3", .arr_delay_ = 5}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));  // 5 min < 8 min
}

// A track change that is taken back restores the scheduled rules.
TEST(rt_transfer_rules, track_change_reverted) {
  auto const tt = load("S1,S2,2,120,,,,\nS1,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 0}}}});
  EXPECT_EQ("S1", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
}

// A second track change replaces the first one.
TEST(rt_transfer_rules, track_change_twice) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
}

// Backward search uses the rule of the new platform too: arriving at B by
// 11:05 needs G, which F cannot reach from S3.
TEST(rt_transfer_rules, track_change_backward_search) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const before =
      search(tt, rtt, std::pair{"B", "A"}, "2019-05-01 11:05 Europe/Berlin",
             direction::kBackward);
  EXPECT_EQ(1U, before.size());
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const after =
      search(tt, rtt, std::pair{"B", "A"}, "2019-05-01 11:05 Europe/Berlin",
             direction::kBackward);
  EXPECT_EQ(0U, after.size());
  auto const later =
      search(tt, rtt, std::pair{"B", "A"}, "2019-05-01 11:35 Europe/Berlin",
             direction::kBackward);
  ASSERT_EQ(1U, later.size());
  EXPECT_EQ(t("2019-05-01 10:00 Europe/Berlin"), begin(later)->dest_time_);
}

// Backward search across a kept timed transfer.
TEST(rt_transfer_rules, track_change_backward_search_timed) {
  auto const tt = load("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  auto const res =
      search(tt, rtt, std::pair{"C", "A"}, "2019-05-01 11:00 Europe/Berlin",
             direction::kBackward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 10:00 Europe/Berlin"), begin(res)->dest_time_);
}

// ===========================================================================
// 5. What a journey and a query see of a moved trip.
// ===========================================================================

// The journey names the platform the trip really stops at.
TEST(rt_transfer_rules, journey_shows_new_platform) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const res =
      search(tt, rtt, std::pair{"A", "B"}, "2019-05-01 10:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  auto const& legs = begin(res)->legs_;
  ASSERT_LE(2U, legs.size());
  EXPECT_EQ("S3", leg_stop_id(tt, legs.front().to_));
  EXPECT_EQ("S2", leg_stop_id(tt, legs.back().from_));
  for (auto const& l : legs) {
    EXPECT_LT(l.from_, tt.n_locations());
    EXPECT_LT(l.to_, tt.n_locations());
  }
}

// A query from the station finds the moved trip at its new platform.
TEST(rt_transfer_rules, start_at_station_with_moved_trip) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "G", 0U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, std::pair{"S", "B"},
                             "2019-05-01 10:35 Europe/Berlin"));
}

// A query to the station arrives with the moved trip.
TEST(rt_transfer_rules, destination_at_station_with_moved_trip) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(t("2019-05-01 10:30 Europe/Berlin"),
            arrival(tt, rtt, std::pair{"A", "S"}));
  // ... and to the new platform itself, but no longer to the old one by train
  EXPECT_EQ(t("2019-05-01 10:30 Europe/Berlin"),
            arrival(tt, rtt, std::pair{"A", "S3"}));
}

// A departure interval instead of one departure time (range search).
TEST(rt_transfer_rules, track_change_interval_search) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const src = source_idx_t{0};
  auto const run = [&](direction const dir, char const* from, char const* to) {
    return raptor_search(
        tt, &rtt,
        routing::query{
            .start_time_ = interval{t("2019-05-01 09:30 Europe/Berlin"),
                                    t("2019-05-01 11:40 Europe/Berlin")},
            .start_match_mode_ = routing::location_match_mode::kEquivalent,
            .dest_match_mode_ = routing::location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                              0_minutes, 0U}}},
        dir);
  };
  auto const fwd = run(direction::kForward, "A", "B");
  ASSERT_EQ(1U, fwd.size());
  EXPECT_EQ(t("2019-05-01 10:00 Europe/Berlin"), begin(fwd)->start_time_);
  EXPECT_EQ(kGLFromF, begin(fwd)->dest_time_);

  auto const bwd = run(direction::kBackward, "B", "A");
  ASSERT_EQ(1U, bwd.size());
  EXPECT_EQ(t("2019-05-01 10:00 Europe/Berlin"), begin(bwd)->dest_time_);
  EXPECT_EQ(kGLFromF, begin(bwd)->start_time_);
}

// P -> Q can change at S or at X. Once P moved to S3, where P -> S2 is not
// possible, the change has to happen at X - and the pass that moves changes
// to nicer places afterwards must not move it back to S.
TEST(rt_transfer_rules, track_change_transfer_not_moved_to_forbidden_pair) {
  auto const tt = load("S3,S2,2,120,,,,\nS3,S2,3,,,,P,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"P", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "P", 1U));
  auto const res =
      search(tt, rtt, std::pair{"A", "B"}, "2019-05-01 12:00 Europe/Berlin");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 13:20 Europe/Berlin"), begin(res)->dest_time_);
  auto const& legs = begin(res)->legs_;
  ASSERT_LE(2U, legs.size());
  EXPECT_EQ("X1", leg_stop_id(tt, legs.front().to_));
  EXPECT_EQ("X2", leg_stop_id(tt, legs.back().from_));
}

// ... while without the rule the same journey may change at either station.
TEST(rt_transfer_rules, guard_transfer_at_either_station) {
  auto const tt = load("S3,S2,2,120,,,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"P", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(t("2019-05-01 13:20 Europe/Berlin"),
            arrival(tt, rtt, kAtoB, "2019-05-01 12:00 Europe/Berlin"));
}

// ===========================================================================
// 6. Situations around the real-time locations themselves.
// ===========================================================================

// Two trips of one route move to the same new platform: they share one
// real-time location, and the route's own rule (RF -> RF takes 10 min) is its
// change time - F -> F2 has 6 min, 11 min once F2 is late.
TEST(rt_transfer_rules, track_change_shared_location_own_change_time) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,600,RF,RF,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(
      0U, search(tt, rtt, std::pair{"A", "D"}, "2019-05-01 10:00 Europe/Berlin")
              .size());
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"F2", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "F", 1U));
  EXPECT_EQ("S4", platform(tt, rtt, "F2", 0U));
  EXPECT_EQ(1U, rtt.rt_virts_.size());
  EXPECT_EQ(
      0U, search(tt, rtt, std::pair{"A", "D"}, "2019-05-01 10:00 Europe/Berlin")
              .size());
  update(
      tt, rtt,
      {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
       {"F2",
        {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4", .dep_delay_ = 5}}}});
  EXPECT_EQ(1U, rtt.rt_virts_.size());  // found again, not created again
  EXPECT_EQ(t("2019-05-01 11:11 Europe/Berlin"),
            arrival(tt, rtt, std::pair{"A", "D"}));
}

// No change of vehicles at S4 at all (same-stop ban): that also holds between
// the platform and a location split off for a moved trip.
TEST(rt_transfer_rules, track_change_onto_platform_without_transfers) {
  auto const rules = std::string{"S,S,2,120,,,,\nS,S,2,900,,,F,G"};
  auto const moves = std::vector<trip_update>{
      {"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
      {"HL", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4"}}}};
  {
    auto const tt = load(rules);
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(kHLFromF, arrival(tt, rtt, kAtoC));  // F -> HL at S4
  }
  {
    auto const tt = load(rules + "\nS4,S4,3,,,,,");
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(0U,
              search(tt, rtt, kAtoC, "2019-05-01 10:00 Europe/Berlin").size());
  }
}

// A stop that is routed at a real-time location gets skipped afterwards.
TEST(rt_transfer_rules, skipped_stop_at_real_time_location) {
  auto const tt = load("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(kHFromF, arrival(tt, rtt, kAtoC));
  auto msg = to_msg({{"F", {{.seq_ = 2U, .stop_id_ = "S1"}}}});
  msg.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->set_schedule_relationship(
          transit_realtime::
              TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", msg);
  EXPECT_EQ(0U,
            search(tt, rtt, kAtoC, "2019-05-01 10:00 Europe/Berlin").size());
}

// Precedence against a rule that names the new platform without qualifying
// the trip: "anything from S4 to an RG trip at S2: 0 min" names both stops
// exactly and beats "RF trips at the station: 15 min".
TEST(rt_transfer_rules, track_change_platform_rule_beats_station_rule) {
  auto const tt = load(
      "S,S,2,120,,,,\nS,S,2,900,RF,,,\n"
      "S4,S2,2,120,,,,\nS4,S2,2,0,,RG,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));  // from S1: 15 min
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));  // from S4: 0 min
}

// An assignment that names the scheduled platform is no stop change - and
// must not swallow the delay of the stop either.
TEST(rt_transfer_rules, assignment_to_scheduled_platform) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(
      tt, rtt,
      {{"F",
        {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S1", .arr_delay_ = 5}}}});
  EXPECT_EQ("S1", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(0U, rtt.rt_virts_.size());
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));  // 5 min < 8 min
}

// Departure interval at the station of a trip that was moved.
TEST(rt_transfer_rules, interval_search_starting_at_moved_trip) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.rt_virts_.size());
  auto const src = source_idx_t{0};
  auto const res = raptor_search(
      tt, &rtt,
      routing::query{
          .start_time_ = interval{t("2019-05-01 10:35 Europe/Berlin"),
                                  t("2019-05-01 10:50 Europe/Berlin")},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .start_ = {{tt.locations_.location_id_to_idx_.at({"S", src}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at({"B", src}),
                            0_minutes, 0U}}},
      direction::kForward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 10:40 Europe/Berlin"), begin(res)->start_time_);
  EXPECT_EQ(kGFromF, begin(res)->dest_time_);
}

// A profile that ignores transfers.txt routes on the platforms (virtual
// locations are projected away): F -> HL is forbidden for the default profile
// but fine for this one - also once HL is a real-time transport, which is
// registered at its virtual location and has to be found from the platform.
TEST(rt_transfer_rules, other_profile_finds_real_time_trips_at_rule_stops) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load("S,S,2,120,,,,\nS,S,3,,,,F,HL");
  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  loader::build_lb_graph<direction::kForward>(tt, kProfile);
  loader::build_lb_graph<direction::kBackward>(tt, kProfile);

  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const run = [&](profile_idx_t const prf) {
    auto const src = source_idx_t{0};
    return raptor_search(
        tt, &rtt,
        routing::query{
            .start_time_ = t("2019-05-01 10:00 Europe/Berlin"),
            .start_match_mode_ = routing::location_match_mode::kEquivalent,
            .dest_match_mode_ = routing::location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at({"A", src}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at({"C", src}),
                              0_minutes, 0U}},
            .prf_idx_ = prf},
        direction::kForward);
  };
  EXPECT_EQ(0U, run(kDefaultProfile).size());
  auto const scheduled = run(kProfile);
  ASSERT_EQ(1U, scheduled.size());
  EXPECT_EQ(kHLFromF, begin(scheduled)->dest_time_);

  update(tt, rtt, {{"HL", {{.seq_ = 1U, .stop_id_ = "S1", .dep_delay_ = 1}}}});
  EXPECT_EQ(0U, run(kDefaultProfile).size());
  auto const delayed = run(kProfile);
  ASSERT_EQ(1U, delayed.size());
  EXPECT_EQ(t("2019-05-01 11:31 Europe/Berlin"), begin(delayed)->dest_time_);
}

// Door to door: offsets to the platforms the moved trips use now.
TEST(rt_transfer_rules, intermodal_with_moved_trips) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  auto const src = source_idx_t{0};
  auto const at = [&](char const* id) {
    return tt.locations_.location_id_to_idx_.at({id, src});
  };
  // ... to a door 4 min from S3, arriving with F
  auto const to_s3 = nigiri::test::raptor_intermodal_search(
      tt, &rtt, {{at("A"), 3_minutes, 0U}}, {{at("S3"), 4_minutes, 0U}},
      t("2019-05-01 09:50 Europe/Berlin"));
  ASSERT_EQ(1U, to_s3.size());
  EXPECT_EQ(t("2019-05-01 10:34 Europe/Berlin"), begin(to_s3)->dest_time_);
  // ... from a door 5 min from S4, leaving with G
  auto const from_s4 = nigiri::test::raptor_intermodal_search(
      tt, &rtt, {{at("S4"), 5_minutes, 0U}}, {{at("B"), 2_minutes, 0U}},
      t("2019-05-01 10:30 Europe/Berlin"));
  ASSERT_EQ(1U, from_s4.size());
  EXPECT_EQ(t("2019-05-01 11:02 Europe/Berlin"), begin(from_s4)->dest_time_);
}

// What a stop is called must not depend on whether a rule split it off to a
// virtual location: F stops at platform S1 of station S either way.
TEST(rt_transfer_rules, stop_name_and_id_at_rule_stop) {
  auto const with_rule = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto const without_rule = load("");
  for (auto const* tt : {&without_rule, &with_rule}) {
    auto const rtt = rt::create_rt_timetable(*tt, kDay);
    auto td = transit_realtime::TripDescriptor{};
    td.set_trip_id("F");
    td.set_start_date("20190501");
    auto const [r, _] =
        rt::gtfsrt_resolve_run(kDay, *tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());
    auto const fr = rt::frun{*tt, &rtt, r};  // run_stop points into it
    auto const stop = fr[1U];
    EXPECT_EQ("S", stop.name(lang_t{}));
    EXPECT_EQ("S", stop.id());
    EXPECT_EQ("S1", stop.get_location_id());
  }
}

// An alert for the station reaches a trip whose stop was split off to a
// virtual location below one of the station's platforms (F at S1).
TEST(rt_transfer_rules, station_alert_at_rule_stop) {
  auto const with_rule = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto const without_rule = load("");
  for (auto const* tt : {&without_rule, &with_rule}) {
    auto rtt = rt::create_rt_timetable(*tt, kDay);
    auto const src = source_idx_t{0};
    auto const station = tt->locations_.location_id_to_idx_.at({"S", src});
    rtt.alerts_.location_[station].push_back(alert_idx_t{7U});

    auto td = transit_realtime::TripDescriptor{};
    td.set_trip_id("F");
    td.set_start_date("20190501");
    auto const [r, trip] = rt::gtfsrt_resolve_run(kDay, *tt, &rtt, src, td);
    ASSERT_TRUE(r.valid());
    auto const fr = rt::frun{*tt, &rtt, r};
    auto const alerts =
        rtt.alerts_.get_alerts(*tt, src, trip, rt_transport_idx_t::invalid(),
                               fr[1U].get_location_idx(), false);
    EXPECT_TRUE(alerts.contains(alert_idx_t{7U}));
  }
}

// ===========================================================================
// 7. Guards: behavior that must not change (these pass before and after).
// ===========================================================================

// An unqualified platform rule never needed a split: it stays with S1.
TEST(rt_transfer_rules, guard_unqualified_platform_rule) {
  auto const tt = load("S1,S2,2,900,,,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", platform(tt, rtt, "F", 1U));
  EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));
}

// Delays at stops without any rule, in a feed that has rules elsewhere.
TEST(rt_transfer_rules, guard_delay_at_plain_stop) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"GL", {{.seq_ = 2U, .stop_id_ = "B", .arr_delay_ = 5}}}});
  EXPECT_EQ(t("2019-05-01 11:35 Europe/Berlin"), arrival(tt, rtt, kAtoB));
}

// The walks of the default profile can be replaced after the import (street
// routing, loader::rebuild_default_profile): a trip that moves to another
// platform walks like that platform does then. F moves to S4, where a rule
// names it, so it is routed at a real-time location.
TEST(rt_transfer_rules, track_change_inherits_rebuilt_walks) {
  auto const moves = std::vector<trip_update>{
      {"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}};
  auto tt = load("S4,S4,2,120,,,,\nS4,S4,2,900,,,F,HL");
  {
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(1U, rtt.rt_virts_.size());
    EXPECT_EQ(kGFromF, arrival(tt, rtt, kAtoB));  // beeline S4 -> S2: 2 min
  }

  auto const s2 = tt.locations_.location_id_to_idx_.at({"S2", source_idx_t{0}});
  auto const s4 = tt.locations_.location_id_to_idx_.at({"S4", source_idx_t{0}});
  auto walks = vector_map<location_idx_t, std::vector<footpath>>{};
  walks.resize(tt.n_locations());
  walks[s4].emplace_back(s2, duration_t{12});
  walks[s2].emplace_back(s4, duration_t{12});
  loader::rebuild_default_profile(tt, walks);
  {
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(1U, rtt.rt_virts_.size());
    EXPECT_EQ(kGLFromF, arrival(tt, rtt, kAtoB));
  }
}
