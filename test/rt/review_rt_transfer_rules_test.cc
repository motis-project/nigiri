#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/helpers/algorithm.h"

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

// Regression tests for real-time defects found in the t2t-rt review. Every
// test states the correct behaviour, so it fails while the defect is present.

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// The network of gtfsrt_transfer_rules_test.cc (S: station with platforms
// S1..S4, X: station with platforms X1, X2), plus:
//   T0  (R0): A0 09:30 -> A 09:50       feeder to F
//   DAB (RD): A0 10:00 -> B 10:45       direct, no change
//   XB  (RX): X1 10:31 -> B 10:40       leaves X right after F arrives at S
//   ST1 (RS1): A2 10:00 -> S1 10:30     first half of a stay-seated chain
//   ST2 (RS2): S1 10:30 -> C 11:00      second half
//   Z: a stop about 180 m north of S4, not walkable except through rules
constexpr auto const kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,50.0,6.0,,,
A0,A0,,50.0,5.9,,,
A2,A2,,50.0,6.1,,,
S,S,,50.0,6.5,,1,
S1,S1,,50.0001,6.5,,,S
S2,S2,,50.0002,6.5,,,S
S3,S3,,50.0003,6.5,,,S
S4,S4,,50.0004,6.5,,,S
Z,Z,,50.0020,6.5,,,
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
R0,AG,R0,,,3
RD,AG,RD,,,3
RX,AG,RX,,,3
RS1,AG,RS1,,,3
RS2,AG,RS2,,,3

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
R0,X,T0,,
RD,X,DAB,,
RX,X,XB,,
RS1,X,ST1,,
RS2,X,ST2,,

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
T0,09:30:00,09:30:00,A0,1,0,0
T0,09:50:00,09:50:00,A,2,0,0
DAB,10:00:00,10:00:00,A0,1,0,0
DAB,10:45:00,10:45:00,B,2,0,0
XB,10:31:00,10:31:00,X1,1,0,0
XB,10:40:00,10:40:00,B,2,0,0
ST1,10:00:00,10:00:00,A2,1,0,0
ST1,10:30:00,10:30:00,S1,2,0,0
ST2,10:30:00,10:30:00,S1,1,0,0
ST2,11:00:00,11:00:00,C,2,0,0

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

location_idx_t lidx(timetable const& tt, char const* id) {
  return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
}

struct stu {
  unsigned seq_;
  std::string stop_id_;
  std::optional<std::string> assigned_{};
  std::optional<int> arr_delay_{};
  bool skipped_{false};
};

struct trip_update {
  std::string trip_id_;
  std::vector<stu> stus_;
};

void update(timetable const& tt,
            rt_timetable& rtt,
            std::vector<trip_update> const& updates) {
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
      x->set_stop_sequence(s.seq_);
      x->set_stop_id(s.stop_id_);
      if (s.assigned_.has_value()) {
        x->mutable_stop_time_properties()->set_assigned_stop_id(*s.assigned_);
      }
      if (s.arr_delay_.has_value()) {
        x->mutable_arrival()->set_delay(*s.arr_delay_ * 60);
      }
      if (s.skipped_) {
        x->set_schedule_relationship(
            transit_realtime::
                TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
      }
    }
  }
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", msg);
  rtt.update_lbs(tt);  // as motis does after every update
}

// Stop-to-stop, the way a server asks: a stop stands for its station.
routing::query query(timetable const& tt,
                     char const* from,
                     char const* to,
                     routing::start_time_t const time) {
  return routing::query{
      .start_time_ = time,
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .start_ = {{lidx(tt, from), 0_minutes, 0U}},
      .destination_ = {{lidx(tt, to), 0_minutes, 0U}}};
}

}  // namespace

// Two real-time virtual locations exist (F moved to S3, G moved to S4). An
// interval search from S puts both into its start set and must not throw.
TEST(t2t_review_rt, interval_search_with_two_real_time_virtual_locations) {
  auto const tt = load("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_GE(rtt.rt_virts_.size(), 2U) << "precondition";

  auto res = pareto_set<routing::journey>{};
  EXPECT_NO_THROW(
      res = raptor_search(tt, &rtt,
                          query(tt, "S", "B",
                                interval{t("2019-05-01 10:35 Europe/Berlin"),
                                         t("2019-05-01 10:50 Europe/Berlin")}),
                          direction::kForward));
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 11:00 Europe/Berlin"), begin(res)->dest_time_);
}

// The timed rule S4 -> X1 for route RF only binds F once F is moved to S4 -
// then F's real-time virtual location has a 0 min edge to X1, and
// T0 + F + XB arrives at B 10:40, before the direct DAB (10:45) but with two
// changes. When F arrives (round 2), DAB's 10:45 is already known: the lower
// bound of F's location must know the rule edge, or the arrival is pruned.
TEST(t2t_review_rt, lower_bound_knows_real_time_rule_edges) {
  auto const tt = load("S4,X1,1,,RF,,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.rt_virts_.size()) << "precondition";

  // control: from A, F is the first vehicle and nothing is known about B yet
  // when it arrives - the rule edge takes F + XB to B at 10:40
  auto const control = raptor_search(
      tt, &rtt, query(tt, "A", "B", t("2019-05-01 10:00 Europe/Berlin")),
      direction::kForward);
  ASSERT_TRUE(utl::any_of(control, [](routing::journey const& j) {
    return j.dest_time_ == t("2019-05-01 10:40 Europe/Berlin");
  })) << "precondition: the real-time rule edge works";

  auto const res = raptor_search(
      tt, &rtt, query(tt, "A0", "B", t("2019-05-01 09:30 Europe/Berlin")),
      direction::kForward);
  EXPECT_TRUE(utl::any_of(res, [](routing::journey const& j) {
    return j.dest_time_ == t("2019-05-01 10:40 Europe/Berlin");
  }));
}

// Z -> S4 is a 5 min rule, Z -> S4 for the trip G a 10 min one. Once G and GL
// are moved to S4, a start at Z 10:32 cannot make G (10:40) but makes GL
// (11:10): the start offset of G's real-time virtual location is the trip
// rule's, not its platform's. (Starting G at the platform's offset, the
// search finds a journey it cannot reconstruct and drops it - GL, which it
// dominated, is gone as well.)
TEST(t2t_review_rt, start_offset_of_real_time_location_honours_rule) {
  auto const tt = load("Z,S4,2,300,,,,\nZ,S4,2,600,,,,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}},
          {"GL", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.rt_virts_.size()) << "precondition";

  auto q = query(tt, "Z", "B", t("2019-05-01 10:32 Europe/Berlin"));
  q.start_match_mode_ = routing::location_match_mode::kExact;
  q.use_start_footpaths_ = true;
  auto const res = raptor_search(tt, &rtt, std::move(q), direction::kForward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(t("2019-05-01 11:30 Europe/Berlin"), begin(res)->dest_time_);
}

// ST1 -> ST2 stay seated at S1; a rule names ST2 there. Moving the joint stop
// to S3 and taking it back restores the schedule, and a repeated revert
// message changes nothing anymore.
TEST(t2t_review_rt, revert_at_seated_joint_restores_schedule) {
  auto const tt = load("S1,S1,4,,,,ST1,ST2\nS,S,2,120,,,,\nS,S,2,900,,,F,ST2");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto n_location_changes = 0U;
  rtt.set_change_callback([&](transport, stop_idx_t, event_type,
                              std::optional<location_idx_t> const l,
                              std::optional<bool>, std::optional<duration_t>) {
    n_location_changes += l.has_value() ? 1U : 0U;
  });

  auto const joint_location = [&]() {
    auto td = transit_realtime::TripDescriptor{};
    td.set_trip_id("ST2");
    td.set_start_date("20190501");
    auto const [r, _] =
        rt::gtfsrt_resolve_run(kDay, tt, &rtt, source_idx_t{0}, td);
    return r.valid() ? rt::frun{tt, &rtt, r}[0U].get_location_idx()
                     : location_idx_t::invalid();
  };
  auto const scheduled = joint_location();
  ASSERT_NE(location_idx_t::invalid(), scheduled) << "precondition";

  update(tt, rtt,
         {{"ST2", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  ASSERT_NE(0U, n_location_changes) << "precondition: the move applied";

  auto const revert =
      trip_update{"ST2", {{.seq_ = 1U, .stop_id_ = "S1", .arr_delay_ = 0}}};
  update(tt, rtt, {revert});
  EXPECT_EQ(scheduled, joint_location());

  n_location_changes = 0U;
  update(tt, rtt, {revert});
  EXPECT_EQ(0U, n_location_changes);
}

// F is named by a trip rule (15 min, one trip), F0 and F2 only by the route
// rules (RF: 15 min, RF -> RG: 3 min). For F -> G the trip rule is the most
// specific one. Moved to S3, F lands on a location that states the same
// values as F0's - but there the route pair is ranked highest (3 min), so the
// key must keep what ranks F's rules.
TEST(t2t_review_rt, real_time_key_keeps_specificity) {
  auto const tt =
      load("S,S,2,120,,,,\nS,S,2,900,,,F,\nS,S,2,900,RF,,,\nS,S,2,180,RF,RG,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const before = raptor_search(
      tt, &rtt, query(tt, "A", "B", t("2019-05-01 10:00 Europe/Berlin")),
      direction::kForward);
  ASSERT_EQ(1U, before.size());
  ASSERT_EQ(t("2019-05-01 11:30 Europe/Berlin"), begin(before)->dest_time_)
      << "precondition: F -> G takes 15 min in the schedule";

  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const after = raptor_search(
      tt, &rtt, query(tt, "A", "B", t("2019-05-01 10:00 Europe/Berlin")),
      direction::kForward);
  ASSERT_EQ(1U, after.size());
  EXPECT_EQ(t("2019-05-01 11:30 Europe/Berlin"), begin(after)->dest_time_);
}
