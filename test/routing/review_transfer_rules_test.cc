#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fmt/format.h"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

// Regression tests for defects found in the t2t-rt review (transfers.txt rules,
// virtual locations and hubs). Every test states the correct behaviour, so it
// fails as long as the defect is present. All feeds run on 2019-05-01 in
// Europe/Berlin; the default change time at a stop is 2 min.

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

struct stop_def {
  std::string_view id_;
  double lat_;
  double lng_;
  std::string_view parent_{};
  bool station_{false};
};

struct trip_def {
  std::string_view id_;
  std::string_view route_;
  std::vector<std::pair<std::string_view, std::string_view>> times_;
  std::string_view block_{};
};

// Routes are taken from the trips, plus the ones rules name without a trip.
std::string feed(std::vector<stop_def> const& stops,
                 std::vector<trip_def> const& trips,
                 std::string_view const transfers,
                 std::vector<std::string_view> const& extra_routes = {}) {
  auto s = std::string{
      "# agency.txt\n"
      "agency_id,agency_name,agency_url,agency_timezone\n"
      "AG,Agency,https://example.com,Europe/Berlin\n\n"
      "# calendar_dates.txt\n"
      "service_id,date,exception_type\n"
      "S1,20190501,1\n\n"
      "# stops.txt\n"
      "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"};
  for (auto const& x : stops) {
    s += fmt::format("{},{} stop,{},{},{},{}\n", x.id_, x.id_, x.lat_, x.lng_,
                     x.station_ ? 1 : 0, x.parent_);
  }

  auto routes = extra_routes;
  for (auto const& x : trips) {
    routes.push_back(x.route_);
  }
  utl::erase_duplicates(routes);
  s += "\n# routes.txt\n"
       "route_id,agency_id,route_short_name,route_long_name,route_type\n";
  for (auto const r : routes) {
    s += fmt::format("{},AG,{},,3\n", r, r);
  }

  s += "\n# trips.txt\nroute_id,service_id,trip_id,block_id\n";
  for (auto const& x : trips) {
    s += fmt::format("{},S1,{},{}\n", x.route_, x.id_, x.block_);
  }

  s += "\n# stop_times.txt\n"
       "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n";
  for (auto const& x : trips) {
    auto seq = 0U;
    for (auto const& [sid, hhmm] : x.times_) {
      s += fmt::format("{},{}:00,{}:00,{},{}\n", x.id_, hhmm, hhmm, sid, seq++);
    }
  }

  s += "\n# transfers.txt\n"
       "from_stop_id,to_stop_id,transfer_type,min_transfer_time,"
       "from_route_id,to_route_id,from_trip_id,to_trip_id\n";
  s += transfers;
  return s;
}

timetable load(std::vector<std::string> const& feeds) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  for (auto i = 0U; i != feeds.size(); ++i) {
    loader::gtfs::load_timetable(
        {}, source_idx_t{static_cast<source_idx_t::value_t>(i)},
        loader::mem_dir::read(feeds[i]), tt);
  }
  loader::finalize(tt);
  return tt;
}

unixtime_t at(std::string_view const hhmm) {
  return parse_time_tz(fmt::format("2019-05-01 {} Europe/Berlin", hhmm),
                       "%Y-%m-%d %H:%M %Z");
}

location_idx_t lidx(timetable const& tt,
                    std::string_view const id,
                    source_idx_t const src = source_idx_t{0U}) {
  return tt.locations_.location_id_to_idx_.at({id, src});
}

pareto_set<routing::journey> search(timetable const& tt,
                                    std::string_view const from,
                                    std::string_view const to,
                                    std::string_view const hhmm) {
  return raptor_search(tt, nullptr, from, to,
                       fmt::format("2019-05-01 {} Europe/Berlin", hhmm));
}

std::size_t n_transit_legs(routing::journey const& j) {
  return static_cast<std::size_t>(
      utl::count_if(j.legs_, [](routing::journey::leg const& l) {
        return std::holds_alternative<routing::journey::run_enter_exit>(
            l.uses_);
      }));
}

// The shared feed of the tests that change out of a trip bound to a virtual
// location: FA (route RF1) is split off at U by the RF1 -> RF3 rule, FB and
// FB2 (RF2) leave from U itself, FA -> FB is a plain change at U (2 min).
std::string split_feed() {
  return feed({{"U", 55.0, 13.0}, {"L", 55.1, 13.0}, {"M", 55.2, 13.0}},
              {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
               {"FB", "RF2", {{"U", "10:33"}, {"M", "11:00"}}},
               {"FB2", "RF2", {{"U", "10:50"}, {"M", "11:20"}}}},
              "U,U,2,120,,,,\n"
              "U,U,2,300,RF1,RF3,,\n",
              {"RF3"});
}

}  // namespace

// ===========================================================================
// GTFS: a rule naming a station applies to all its child stops - also to a
// change that stays at one child stop.
// ===========================================================================

// XS,XS,2,600: every change inside station XS needs 10 min. TA1 arrives at
// platform X1 10:30, TA2 leaves X1 10:35 (5 min) -> not reachable, TA3 10:45.
TEST(t2t_review, station_min_time_applies_within_one_platform) {
  auto const tt =
      load({feed({{"XS", 50.0, 8.0, "", true},
                  {"X1", 50.0, 8.0, "XS"},
                  {"X2", 50.0003, 8.0, "XS"},
                  {"A", 50.1, 8.0},
                  {"B", 50.2, 8.0}},
                 {{"TA1", "RA1", {{"A", "10:00"}, {"X1", "10:30"}}},
                  {"TA2", "RA2", {{"X1", "10:35"}, {"B", "11:00"}}},
                  {"TA3", "RA2", {{"X1", "10:45"}, {"B", "11:15"}}}},
                 "XS,XS,2,600,,,,\n")});
  auto const res = search(tt, "A", "B", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:15"), begin(res)->dest_time_);
}

// YS,YS,3: no change between routes anywhere in station YS - also not at the
// platform Y1 itself.
TEST(t2t_review, station_ban_applies_within_one_platform) {
  auto const tt =
      load({feed({{"YS", 50.5, 8.5, "", true},
                  {"Y1", 50.5, 8.5, "YS"},
                  {"Y2", 50.5003, 8.5, "YS"},
                  {"C", 50.6, 8.5},
                  {"D", 50.7, 8.5}},
                 {{"TE1", "RE1", {{"C", "10:00"}, {"Y1", "10:30"}}},
                  {"TE2", "RE2", {{"Y1", "10:40"}, {"D", "11:00"}}}},
                 "YS,YS,3,,,,,\n")});
  EXPECT_EQ(0U, search(tt, "C", "D", "10:00").size());
}

// ===========================================================================
// A virtual location's own change time: a rule qualified on one side only
// ("arriving on RB, whatever departs") also covers two RB trips that share
// the virtual location.
// ===========================================================================

TEST(t2t_review, one_sided_rule_applies_between_trips_of_its_route) {
  auto const tt =
      load({feed({{"Z", 52.0, 10.0},
                  {"E", 52.1, 10.0},
                  {"F", 52.2, 10.0},
                  {"E2", 52.1, 10.2},
                  {"F2", 52.2, 10.2}},
                 {{"TB1", "RB", {{"E", "12:00"}, {"Z", "12:30"}}},
                  {"TB2", "RB", {{"Z", "12:35"}, {"F", "13:00"}}},
                  {"TB3", "RB", {{"Z", "12:45"}, {"F", "13:15"}}},
                  {"TB4", "RB", {{"E2", "12:00"}, {"Z", "12:30"}}},
                  {"TB5", "RB2", {{"Z", "12:35"}, {"F2", "13:00"}}},
                  {"TB6", "RB2", {{"Z", "12:45"}, {"F2", "13:15"}}}},
                 "Z,Z,2,120,,,,\n"
                 "Z,Z,2,600,RB,,,\n")});

  // control: RB -> RB2 respects the 10 min
  auto const other_route = search(tt, "E2", "F2", "12:00");
  ASSERT_EQ(1U, other_route.size());
  EXPECT_EQ(at("13:15"), begin(other_route)->dest_time_);

  // RB -> RB: the same rule, so TB2 (5 min) cannot be reached either
  auto const same_route = search(tt, "E", "F", "12:00");
  ASSERT_EQ(1U, same_route.size());
  EXPECT_EQ(at("13:15"), begin(same_route)->dest_time_);
}

// ===========================================================================
// The virtual location key has to keep the rule's specificity: CA is named by
// a trip rule (5 min), CA2 only by the route rules. For CA2 -> CC2 the most
// specific rule is RC -> RC2 (10 min), not CA's trip rule.
// ===========================================================================

TEST(t2t_review, virtual_location_key_keeps_specificity) {
  auto const tt =
      load({feed({{"S", 53.0, 11.0},
                  {"S2", 53.0005, 11.0},
                  {"G", 53.1, 11.0},
                  {"H", 53.2, 11.0}},
                 {{"CA", "RC", {{"G", "13:00"}, {"S", "13:30"}}},
                  {"CA2", "RC", {{"G", "14:00"}, {"S", "14:30"}}},
                  {"CC", "RC2", {{"S2", "13:36"}, {"H", "14:00"}}},
                  {"CC2", "RC2", {{"S2", "14:36"}, {"H", "15:00"}}},
                  {"CC3", "RC2", {{"S2", "14:45"}, {"H", "15:10"}}}},
                 "S,S2,2,180,,,,\n"
                 "S,S2,2,300,,,CA,\n"
                 "S,S2,2,300,RC,,,\n"
                 "S,S2,2,600,RC,RC2,,\n")});

  // control: CA -> CC, the trip rule (one trip beats both routes) gives 5 min
  auto const ca = search(tt, "G", "H", "13:00");
  ASSERT_EQ(1U, ca.size());
  EXPECT_EQ(at("14:00"), begin(ca)->dest_time_);

  // CA2 -> CC2 (6 min) needs the route pair's 10 min -> CC3
  auto const ca2 = search(tt, "G", "H", "14:00");
  ASSERT_EQ(1U, ca2.size());
  EXPECT_EQ(at("15:10"), begin(ca2)->dest_time_);
}

// Without a rule of another duration in between, the rank does not decide:
// CA ("trip CA -> anything: 5 min") and CA2 ("route RC -> anything: 5 min")
// get 5 min for every partner either way, so they keep sharing one virtual
// location - and CA2 makes CC2 (6 min).
TEST(t2t_review, virtual_location_key_merges_equal_values_without_competition) {
  auto const tt =
      load({feed({{"S", 53.0, 11.0},
                  {"S2", 53.0005, 11.0},
                  {"G", 53.1, 11.0},
                  {"H", 53.2, 11.0}},
                 {{"CA", "RC", {{"G", "13:00"}, {"S", "13:30"}}},
                  {"CA2", "RC", {{"G", "14:00"}, {"S", "14:30"}}},
                  {"CC", "RC2", {{"S2", "13:36"}, {"H", "14:00"}}},
                  {"CC2", "RC2", {{"S2", "14:36"}, {"H", "15:00"}}},
                  {"CC3", "RC2", {{"S2", "14:45"}, {"H", "15:10"}}}},
                 "S,S2,2,180,,,,\n"
                 "S,S2,2,300,,,CA,\n"
                 "S,S2,2,300,RC,,,\n")});
  auto const s = lidx(tt, "S");
  EXPECT_EQ(
      1, utl::count_if(tt.locations_.children_[s], [&](location_idx_t const c) {
        return tt.locations_.types_[c] == location_type::kVirt;
      }));
  auto const ca2 = search(tt, "G", "H", "14:00");
  ASSERT_EQ(1U, ca2.size());
  EXPECT_EQ(at("15:00"), begin(ca2)->dest_time_);
}

// ===========================================================================
// The majority fold and the station hierarchy.
// ===========================================================================

// DA -> DB is named by a station level trip rule (2 min), the platforms T1/T2
// carry an RD1 -> RD2 route rule (8 min) and a plain 5 min row. The trip rule
// is the most specific one for DA -> DB: the 3 min gap is enough.
TEST(t2t_review, fold_keeps_station_trip_rule_over_platform_route_rule) {
  auto const tt =
      load({feed({{"T", 54.0, 12.0, "", true},
                  {"T1", 54.0, 12.0, "T"},
                  {"T2", 54.0003, 12.0, "T"},
                  {"J", 54.1, 12.0},
                  {"K", 54.2, 12.0}},
                 {{"DA", "RD1", {{"J", "16:00"}, {"T1", "16:30"}}},
                  {"DB", "RD2", {{"T2", "16:33"}, {"K", "17:00"}}},
                  {"DB2", "RD2", {{"T2", "16:45"}, {"K", "17:15"}}}},
                 "T,T,2,120,,,DA,DB\n"
                 "T1,T2,2,300,,,,\n"
                 "T1,T2,2,480,RD1,RD2,,\n")});
  auto const res = search(tt, "J", "K", "16:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("17:00"), begin(res)->dest_time_);
}

// SJ,SJ,2,600 states the default for every pair of SJ's platforms. The one
// route rule RJ1 -> RJ2 on J1 -> J2 must not replace it for other routes:
// RJ1 -> RJ3 (5 min gap) still needs 10 min.
TEST(t2t_review, fold_keeps_explicit_station_default) {
  auto const tt =
      load({feed({{"SJ", 59.0, 17.0, "", true},
                  {"J1", 59.0, 17.0, "SJ"},
                  {"J2", 59.0003, 17.0, "SJ"},
                  {"JA", 59.1, 17.0},
                  {"JB", 59.2, 17.0}},
                 {{"JT1", "RJ1", {{"JA", "10:00"}, {"J1", "10:30"}}},
                  {"JC", "RJ3", {{"J2", "10:35"}, {"JB", "11:00"}}},
                  {"JC2", "RJ3", {{"J2", "10:45"}, {"JB", "11:15"}}}},
                 "SJ,SJ,2,600,,,,\n"
                 "J1,J2,2,180,RJ1,RJ2,,\n",
                 {"RJ2"})});
  auto const res = search(tt, "JA", "JB", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:15"), begin(res)->dest_time_);
}

// The station states its default (LS,LS: 1 min), so the qualified rows at
// platform L1 are exceptions to it and nothing is folded: every other trip
// pair at L1 has the station's value - for a trip split off at L1 (LT5, by
// its 0 min trip rule) just like for a plain one (LT9).
TEST(t2t_review, folded_platform_default_holds_at_its_virtual_locations) {
  auto const tt =
      load({feed({{"LS", 61.0, 19.0, "", true},
                  {"L1", 61.0, 19.0, "LS"},
                  {"L2", 61.0003, 19.0, "LS"},
                  {"LO", 61.1, 19.0},
                  {"LO2", 61.1, 19.3},
                  {"LD", 61.2, 19.0},
                  {"LD2", 61.2, 19.2}},
                 {{"LT5", "RL5", {{"LO", "10:00"}, {"L1", "10:30"}}},
                  {"LT6", "RL6", {{"L1", "10:40"}, {"LD", "11:00"}}},
                  {"LT7", "RL7", {{"L1", "10:32"}, {"LD2", "11:00"}}},
                  {"LT8", "RL7", {{"L1", "10:36"}, {"LD2", "11:05"}}},
                  {"LT9", "RL9", {{"LO2", "10:00"}, {"L1", "10:30"}}}},
                 "LS,LS,2,60,,,,\n"
                 "L1,L1,2,300,RL1,RL2,,\n"
                 "L1,L1,2,300,RL3,RL4,,\n"
                 "L1,L1,2,0,,,LT5,LT6\n",
                 {"RL1", "RL2", "RL3", "RL4"})});

  // a plain arrival at L1 makes LT7 (2 min later)
  auto const plain = search(tt, "LO2", "LD2", "10:00");
  ASSERT_EQ(1U, plain.size());
  EXPECT_EQ(at("11:00"), begin(plain)->dest_time_);

  // LT5 arrives at its virtual location below L1: the same
  auto const split = search(tt, "LO", "LD2", "10:00");
  ASSERT_EQ(1U, split.size());
  EXPECT_EQ(at("11:00"), begin(split)->dest_time_);
}

// One guarantee (KT1 -> KT2) next to three 5 min trip pairs at K: the 5 min
// rows are the majority and become K's change time. The guarantee names other
// trips, so it applies to none of their pairs: the three rows are redundant
// and only KT1/KT2 get virtual locations. KT3 -> KT4 (4 min) still misses.
TEST(t2t_review, fold_drops_default_rows_next_to_rules_for_other_trips) {
  auto const tt =
      load({feed({{"K", 62.0, 20.0}, {"KA", 62.1, 20.0}, {"KB", 62.2, 20.0}},
                 {{"KT1", "RK1", {{"KA", "08:00"}, {"K", "08:30"}}},
                  {"KT2", "RK2", {{"K", "08:30"}, {"KB", "09:00"}}},
                  {"KT3", "RK1", {{"KA", "09:30"}, {"K", "10:00"}}},
                  {"KT4", "RK2", {{"K", "10:04"}, {"KB", "10:30"}}},
                  {"KT5", "RK1", {{"KA", "10:30"}, {"K", "11:00"}}},
                  {"KT6", "RK2", {{"K", "11:10"}, {"KB", "11:40"}}},
                  {"KT7", "RK1", {{"KA", "11:30"}, {"K", "12:00"}}},
                  {"KT8", "RK2", {{"K", "12:10"}, {"KB", "12:40"}}},
                  {"KT9", "RK2", {{"K", "10:10"}, {"KB", "10:40"}}}},
                 "K,K,1,,,,KT1,KT2\n"
                 "K,K,2,300,,,KT3,KT4\n"
                 "K,K,2,300,,,KT5,KT6\n"
                 "K,K,2,300,,,KT7,KT8\n")});
  auto const k = lidx(tt, "K");
  EXPECT_EQ(5, tt.locations_.transfer_time_[k].count());
  EXPECT_EQ(
      2, utl::count_if(tt.locations_.children_[k], [&](location_idx_t const c) {
        return tt.locations_.types_[c] == location_type::kVirt;
      }));

  auto const guaranteed = search(tt, "KA", "KB", "08:00");
  ASSERT_EQ(1U, guaranteed.size());
  EXPECT_EQ(at("09:00"), begin(guaranteed)->dest_time_);

  auto const missed = search(tt, "KA", "KB", "09:30");
  ASSERT_EQ(1U, missed.size());
  EXPECT_EQ(at("10:40"), begin(missed)->dest_time_);
}

// ===========================================================================
// Block through-services and stay-seated chains.
// ===========================================================================

// IT1 and IT2 form one block (through ride at BS). A guarantee from another
// trip into IT2 splits IT2's first stop off to a virtual location - the
// through ride has to stay, exactly like the control block CT1 + CT2.
TEST(t2t_review, block_through_service_survives_rule_at_junction) {
  auto const tt =
      load({feed({{"BS", 58.0, 16.0},
                  {"BA", 58.1, 16.0},
                  {"BB", 58.2, 16.0},
                  {"BZ", 58.1, 16.3},
                  {"CS", 58.5, 16.5},
                  {"CA", 58.6, 16.5},
                  {"CB", 58.7, 16.5}},
                 {{"IT1", "RI1", {{"BA", "10:00"}, {"BS", "10:30"}}, "K1"},
                  {"IT2", "RI2", {{"BS", "10:30"}, {"BB", "11:00"}}, "K1"},
                  {"IT0", "RI3", {{"BZ", "09:00"}, {"BS", "10:20"}}},
                  {"CT1", "RC1", {{"CA", "10:00"}, {"CS", "10:30"}}, "K2"},
                  {"CT2", "RC9", {{"CS", "10:30"}, {"CB", "11:00"}}, "K2"}},
                 "BS,BS,1,,,,IT0,IT2\n")});

  auto const control = search(tt, "CA", "CB", "10:00");
  ASSERT_EQ(1U, control.size());
  EXPECT_EQ(at("11:00"), begin(control)->dest_time_);
  EXPECT_EQ(1U, n_transit_legs(*begin(control)));

  auto const res = search(tt, "BA", "BB", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
  EXPECT_EQ(1U, n_transit_legs(*begin(res)));
}

// ST1 -> ST2 is a stay-seated chain (type 4) at SS. The rule SX -> ST2 needs
// 10 min, SX arrives 4 min before ST2 leaves: a passenger of SX cannot make
// ST2. The chain must not drop the rule of ST2's first stop.
TEST(t2t_review, stay_seated_chain_keeps_rule_of_second_trip) {
  auto const tt =
      load({feed({{"SS", 58.3, 16.3},
                  {"SA", 58.4, 16.3},
                  {"SB", 58.5, 16.3},
                  {"SX0", 58.4, 16.6}},
                 {{"ST1", "RS1", {{"SA", "10:00"}, {"SS", "10:30"}}},
                  {"ST2", "RS2", {{"SS", "10:30"}, {"SB", "11:00"}}},
                  {"SX", "RSX", {{"SX0", "09:00"}, {"SS", "10:26"}}}},
                 "SS,SS,4,,,,ST1,ST2\n"
                 "SS,SS,2,120,,,,\n"
                 "SS,SS,2,600,,,SX,ST2\n")});

  // control: the chain itself works
  auto const chain = search(tt, "SA", "SB", "10:00");
  ASSERT_EQ(1U, chain.size());
  EXPECT_EQ(at("11:00"), begin(chain)->dest_time_);

  EXPECT_EQ(0U, search(tt, "SX0", "SB", "09:00").size());
}

// ===========================================================================
// Transfer time settings and hubs with weight 0.
// ===========================================================================

// W,W,1 makes W's own change time 0 min, and the RG1 -> RG3 rule splits GA off
// to a virtual location, so W gets a per-stop hub of weight 0. With a minimum
// transfer time of 5 min, every change at W takes 5 min: GB (2 min after the
// arrival) is not reachable, GB2 (10 min) is.
namespace {

std::string zero_min_hub_feed() {
  return feed({{"W", 56.0, 14.0},
               {"N", 56.1, 14.0},
               {"N2", 56.1, 14.2},
               {"O", 56.2, 14.0}},
              {{"GA", "RG1", {{"N", "10:00"}, {"W", "10:30"}}},
               {"GB", "RG2", {{"W", "10:32"}, {"O", "11:00"}}},
               {"GB2", "RG2", {{"W", "10:40"}, {"O", "11:10"}}},
               {"GC", "RG4", {{"N2", "10:00"}, {"W", "10:30"}}}},
              "W,W,1,,,,,\n"
              "W,W,2,300,RG1,RG3,,\n",
              {"RG3"});
}

pareto_set<routing::journey> min_transfer_time_search(
    timetable const& tt, std::string_view const from) {
  return raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, from), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "O"), 0_minutes, 0U}},
                     .transfer_time_settings_ = {
                         .default_ = false, .min_transfer_time_ = 5_minutes}});
}

}  // namespace

// GC arrives at W itself: the change at W must not make the journey vanish.
TEST(t2t_review, min_transfer_time_keeps_change_at_stop_with_zero_min_hub) {
  auto const tt = load({zero_min_hub_feed()});
  auto const res = min_transfer_time_search(tt, "N2");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:10"), begin(res)->dest_time_);
}

// GA arrives at its virtual location: the minimum holds there too.
TEST(t2t_review, min_transfer_time_applies_to_zero_min_hub) {
  auto const tt = load({zero_min_hub_feed()});
  auto const res = min_transfer_time_search(tt, "N");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:10"), begin(res)->dest_time_);
}

// ===========================================================================
// Consumers of the transfer relation besides RAPTOR: the change FA (virtual
// location below U) -> FB (at U) exists only through U's per-stop hub.
// ===========================================================================

TEST(t2t_review, trip_based_routing_sees_hub_transfers) {
  auto const tt = load({split_feed()});

  // control: RAPTOR finds FA -> FB
  auto const raptor = search(tt, "L", "M", "10:00");
  ASSERT_EQ(1U, raptor.size());
  EXPECT_EQ(at("11:00"), begin(raptor)->dest_time_);

  auto const tbd = routing::tb::preprocess(tt, kDefaultProfile);
  auto search_state = routing::search_state{};
  auto algo_state = routing::tb::query_state{tt, tbd};
  auto const res =
      *(routing::search<direction::kForward, routing::tb::query_engine<false>>{
          tt, nullptr, search_state, algo_state,
          routing::query{.start_time_ = at("10:00"),
                         .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
                         .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}}}}
            .execute()
            .journeys_);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// Without hubs (finalize_options::hubs_ = false) every transfer is a footpath:
// trip-based routing sees FA -> FB, and RAPTOR finds the same journeys as on
// the timetable with hubs.
TEST(t2t_review, hub_free_timetable_serves_trip_based_routing) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(split_feed()), tt);
  loader::finalize(
      tt, loader::finalize_options{
              .adjust_footpaths_ = false,
              .merge_dupes_intra_src_ = false,
              .merge_dupes_inter_src_ = false,
              .max_footpath_length_ = std::numeric_limits<std::uint16_t>::max(),
              .hubs_ = false});
  for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
    ASSERT_EQ(0U, tt.locations_.hub_in_[p].size()) << "profile " << p;
  }

  auto const tbd = routing::tb::preprocess(tt, kDefaultProfile);
  auto search_state = routing::search_state{};
  auto algo_state = routing::tb::query_state{tt, tbd};
  auto const res =
      *(routing::search<direction::kForward, routing::tb::query_engine<false>>{
          tt, nullptr, search_state, algo_state,
          routing::query{.start_time_ = at("10:00"),
                         .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
                         .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}}}}
            .execute()
            .journeys_);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);

  auto const with_hubs = load({split_feed()});
  ASSERT_NE(0U, with_hubs.locations_.hub_in_[kDefaultProfile].size())
      << "precondition: the feed has hubs";
  EXPECT_EQ(test::print_results(with_hubs, nullptr,
                                search(with_hubs, "L", "M", "10:00")),
            test::print_results(tt, nullptr, search(tt, "L", "M", "10:00")));
}

// A time-dependent offset (motis: flex) names the stop. FA arrives at, and FC
// leaves from, virtual locations of U (the RF1 -> RF3 rule): the offset holds
// there too, like any other offset at U.
TEST(t2t_review, td_offsets_hold_at_virtual_locations) {
  auto const tt =
      load({feed({{"U", 55.0, 13.0}, {"L", 55.1, 13.0}, {"M", 55.2, 13.0}},
                 {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
                  {"FC", "RF3", {{"U", "10:35"}, {"M", "11:00"}}}},
                 "U,U,2,120,,,,\n"
                 "U,U,2,300,RF1,RF3,,\n")});
  auto const u = lidx(tt, "U");
  ASSERT_TRUE(utl::any_of(tt.locations_.children_[u],
                          [&](location_idx_t const c) {
                            return tt.locations_.types_[c] ==
                                   location_type::kVirt;
                          }))
      << "precondition: U has virtual locations";
  auto const td = [](std::string_view const hhmm, duration_t const d) {
    return std::vector<routing::td_offset>{
        {.valid_from_ = at(hhmm), .duration_ = d, .transport_mode_payload_ = 0},
        {.valid_from_ = at("12:00"),
         .duration_ = footpath::kMaxDuration,
         .transport_mode_payload_ = 0}};
  };

  // to U by flex (2 min), then FC from its virtual location
  auto const from_u = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = at("10:30"),
          .start_match_mode_ = routing::location_match_mode::kIntermodal,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}},
          .td_start_ = {{u, td("10:00", 2_minutes)}}},
      direction::kForward);
  ASSERT_EQ(1U, from_u.size());
  EXPECT_EQ(at("11:00"), begin(from_u)->dest_time_);
  // the start leg names the query's stop, not the virtual location FC leaves
  auto const& access = begin(from_u)->legs_.front();
  ASSERT_TRUE(std::holds_alternative<routing::offset>(access.uses_));
  EXPECT_EQ(u, std::get<routing::offset>(access.uses_).target());

  // FA to its virtual location of U, then on by flex (5 min)
  auto const to_u = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = at("10:00"),
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kIntermodal,
          .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
          .td_dest_ = {{u, td("10:00", 5_minutes)}}},
      direction::kForward);
  ASSERT_EQ(1U, to_u.size());
  EXPECT_EQ(at("10:35"), begin(to_u)->dest_time_);
  auto const& egress = begin(to_u)->legs_.back();
  ASSERT_TRUE(std::holds_alternative<routing::offset>(egress.uses_));
  EXPECT_EQ(u, std::get<routing::offset>(egress.uses_).target());
}

// FB2 (10:50) is an alternative for the FB leg of L -> FA -> FB -> M.
TEST(t2t_review, leg_alternatives_after_split_off_trip) {
  auto const tt = load({split_feed()});
  auto const q =
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}}};
  auto const res = raptor_search(tt, nullptr, routing::query{q});
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);

  auto fb_leg = j.legs_.size();
  for (auto i = 0U; i != j.legs_.size(); ++i) {
    if (std::holds_alternative<routing::journey::run_enter_exit>(
            j.legs_[i].uses_) &&
        j.legs_[i].dep_time_ == at("10:33")) {
      fb_leg = i;
    }
  }
  ASSERT_NE(j.legs_.size(), fb_leg);

  auto const alternatives =
      routing::get_leg_alternatives(tt, nullptr, q, j, fb_leg, 3U);
  EXPECT_TRUE(utl::any_of(alternatives, [&](routing::journey const& alt) {
    return utl::any_of(alt.legs_, [&](routing::journey::leg const& l) {
      return std::holds_alternative<routing::journey::run_enter_exit>(
                 l.uses_) &&
             l.dep_time_ == at("10:50");
    });
  }));
}

// The walk FA's virtual location -> U, as motis' itinerary refresh looks it
// up between two transit legs.
TEST(t2t_review, lookup_footpath_from_virtual_location_to_its_stop) {
  auto const tt = load({split_feed()});
  auto const res = search(tt, "L", "M", "10:00");
  ASSERT_EQ(1U, res.size());
  auto const& fa = begin(res)->legs_.front();
  ASSERT_TRUE(
      std::holds_alternative<routing::journey::run_enter_exit>(fa.uses_));
  ASSERT_EQ(location_type::kVirt, tt.locations_.types_[fa.to_]);

  auto const fp = routing::lookup_footpath(
      fa.to_, fa.arr_time_, routing::side::kAlighting, tt, nullptr,
      routing::query{}, {{lidx(tt, "U"), 0_minutes, 0U}},
      routing::location_match_mode::kExact, true);
  EXPECT_TRUE(fp.has_value());
}

// Door to door, the egress offset belongs to the query: its target is U, not
// the virtual location FA arrives at.
TEST(t2t_review, egress_offset_names_the_query_stop) {
  auto const tt = load({split_feed()});
  auto const res = nigiri::test::raptor_intermodal_search(
      tt, nullptr, {{lidx(tt, "L"), 0_minutes, 0U}},
      {{lidx(tt, "U"), 3_minutes, 0U}}, at("10:00"));
  ASSERT_EQ(1U, res.size());
  auto const& egress = begin(res)->legs_.back();
  ASSERT_TRUE(std::holds_alternative<routing::offset>(egress.uses_));
  EXPECT_EQ(lidx(tt, "U"), std::get<routing::offset>(egress.uses_).target());
}

// P and Q (44 m apart) both carry two virtual locations, so their walk is a
// walk hub and the stored footpath P -> Q is pruned. The fastest direct
// connection P -> Q is still that 2 min walk.
TEST(t2t_review, fastest_direct_sees_walk_hubs) {
  auto const tt =
      load({feed({{"P", 57.0, 15.0},
                  {"Q", 57.0004, 15.0},
                  {"PA", 57.1, 15.0},
                  {"QA", 57.1, 15.2}},
                 {{"TP1", "RP1", {{"PA", "09:00"}, {"P", "09:30"}}},
                  {"TP2", "RP2", {{"PA", "09:10"}, {"P", "09:40"}}},
                  {"TQ1", "RQ1", {{"QA", "09:00"}, {"Q", "09:30"}}},
                  {"TQ2", "RQ2", {{"QA", "09:10"}, {"Q", "09:40"}}}},
                 "P,P,2,120,,,,\n"
                 "P,P,2,300,RP1,RX,,\n"
                 "P,P,2,420,RP2,RX,,\n"
                 "Q,Q,2,120,,,,\n"
                 "Q,Q,2,300,RQ1,RX,,\n"
                 "Q,Q,2,420,RQ2,RX,,\n",
                 {"RX"})});
  auto const p = lidx(tt, "P");
  auto const q = lidx(tt, "Q");

  // precondition: P -> Q exists only through a hub
  EXPECT_TRUE(
      utl::none_of(tt.locations_.footpaths_out_[kDefaultProfile][p],
                   [&](footpath const fp) { return fp.target() == q; }));
  auto hub_walk = std::optional<duration_t>{};
  routing::for_each_hub_source<direction::kForward>(
      tt, kDefaultProfile, q, [&](footpath const fp) {
        if (fp.target() == p) {
          hub_walk = fp.duration();
        }
        return true;
      });
  ASSERT_TRUE(hub_walk.has_value());

  auto const direct = routing::get_fastest_direct(
      tt,
      routing::query{.start_ = {{p, 0_minutes, 0U}},
                     .destination_ = {{q, 0_minutes, 0U}}},
      direction::kForward);
  EXPECT_EQ(hub_walk->count(), direct.count());
}

// ===========================================================================
// Tables sized before the virtual locations exist.
// ===========================================================================

// motis' flex routing indexes location_location_groups_ with every location
// near a coordinate, virtual ones included.
TEST(t2t_review, location_group_table_covers_virtual_locations) {
  auto const tt = load({split_feed()});
  ASSERT_TRUE(utl::any_of(tt.locations_.types_, [](location_type const t) {
    return t == location_type::kVirt;
  }));
  EXPECT_EQ(tt.n_locations(), tt.location_location_groups_.size());
}

// ===========================================================================
// Walks and the stops' change times.
// ===========================================================================

// E1 has a 5 min change time. E2 (same feed, 22 m) and E3 (other feed, 21 m)
// are equally close: leaving E1 on foot has to cost the same either way. The
// cross-feed link takes max(change times, walk), the same-feed one does not.
TEST(t2t_review, same_feed_walk_respects_change_time_like_cross_feed) {
  auto const tt = load(
      {feed({{"E1", 60.5, 20.0}, {"E2", 60.5002, 20.0}, {"E0", 60.6, 20.0}},
            {{"ET1", "RE1", {{"E0", "10:00"}, {"E1", "10:30"}}},
             {"ET2", "RE2", {{"E2", "10:40"}, {"E0", "11:00"}}}},
            "E1,E1,2,300,,,,\n"),
       feed({{"E3", 60.5, 20.0004}, {"E9", 60.6, 20.3}},
            {{"ET3", "RE3", {{"E3", "10:40"}, {"E9", "11:00"}}}}, "")});
  auto const e1 = lidx(tt, "E1");
  auto const walk_to = [&](location_idx_t const target) {
    auto best = std::optional<duration_t>{};
    for (auto const fp : tt.locations_.footpaths_out_[kDefaultProfile][e1]) {
      if (fp.target() == target) {
        best = fp.duration();
      }
    }
    return best;
  };
  auto const to_e2 = walk_to(lidx(tt, "E2"));
  auto const to_e3 = walk_to(lidx(tt, "E3", source_idx_t{1U}));
  ASSERT_TRUE(to_e2.has_value());
  ASSERT_TRUE(to_e3.has_value());
  EXPECT_EQ(to_e3->count(), to_e2->count());
}

// ===========================================================================
// GTFS transfers.txt semantics.
// ===========================================================================

// transfer_type 5: no in-seat transfer between K5T1 and K5T2 although they
// share a block - the rider has to alight, and a 0 min change is not enough.
TEST(t2t_review, type5_prevents_block_through_service) {
  auto const tt = load(
      {feed({{"K5S", 62.0, 21.0}, {"K5A", 62.1, 21.0}, {"K5B", 62.2, 21.0}},
            {{"K5T1", "RK1", {{"K5A", "10:00"}, {"K5S", "10:30"}}, "K5"},
             {"K5T2", "RK2", {{"K5S", "10:30"}, {"K5B", "11:00"}}, "K5"}},
            "K5S,K5S,5,,,,K5T1,K5T2\n")});
  EXPECT_EQ(0U, search(tt, "K5A", "K5B", "10:00").size());
}

// transfer_type 1: the departing vehicle waits and leaves sufficient time,
// so TT1 -> TT2 is guaranteed although the scheduled gap (2 min) is shorter
// than the stated min_transfer_time (5 min).
TEST(t2t_review, timed_transfer_with_min_time_is_guaranteed) {
  auto const tt = load(
      {feed({{"TTS", 62.5, 21.5}, {"TTA", 62.6, 21.5}, {"TTB", 62.7, 21.5}},
            {{"TT1", "RT1", {{"TTA", "10:00"}, {"TTS", "10:30"}}},
             {"TT2", "RT2", {{"TTS", "10:32"}, {"TTB", "11:00"}}},
             {"TT2L", "RT2", {{"TTS", "10:50"}, {"TTB", "11:20"}}}},
            "TTS,TTS,2,120,,,,\n"
            "TTS,TTS,1,300,,,TT1,TT2\n")});
  auto const res = search(tt, "TTA", "TTB", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// A profile that ignores the qualified rules keeps the stop's plain change
// time (2 min). The 0 s row for the trip pair Q1 -> Q2 says nothing about
// Q3 -> Q4 (1 min gap).
TEST(t2t_review, qualified_row_does_not_set_stop_time_for_other_profiles) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load({feed({{"QS", 63.0, 22.0},
                        {"QA", 63.1, 22.0},
                        {"QB", 63.2, 22.0},
                        {"QC", 63.1, 22.3},
                        {"QD", 63.2, 22.3}},
                       {{"Q1", "RQ1", {{"QA", "10:00"}, {"QS", "10:30"}}},
                        {"Q2", "RQ2", {{"QS", "10:40"}, {"QB", "11:00"}}},
                        {"Q3", "RQ3", {{"QC", "10:00"}, {"QS", "10:30"}}},
                        {"Q4", "RQ4", {{"QS", "10:31"}, {"QD", "11:00"}}},
                        {"Q4L", "RQ4", {{"QS", "10:40"}, {"QD", "11:10"}}}},
                       "QS,QS,2,120,,,,\n"
                       "QS,QS,2,0,,,Q1,Q2\n")});
  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  loader::build_lb_graph<direction::kForward>(tt, kProfile);
  loader::build_lb_graph<direction::kBackward>(tt, kProfile);

  auto const res = raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "QC"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "QD"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:10"), begin(res)->dest_time_);
}

// min_transfer_time is a non-negative number of seconds. A negative value is
// invalid input: at most it means "no time", never "no transfer".
TEST(t2t_review, negative_min_transfer_time_is_not_a_ban) {
  auto const tt =
      load({feed({{"NA1", 64.0, 23.0},
                  {"NA2", 64.0003, 23.0},
                  {"N0", 64.1, 23.0},
                  {"N9", 64.2, 23.0}},
                 {{"NT1", "RN1", {{"N0", "10:00"}, {"NA1", "10:30"}}},
                  {"NT2", "RN2", {{"NA2", "10:40"}, {"N9", "11:00"}}}},
                 "NA1,NA2,2,-120,,,,\n")});
  auto const res = search(tt, "N0", "N9", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// 36000 s = 600 min is a valid (if long) minimum transfer time. It exceeds
// what a footpath can hold (511 min), but it is no ban: the 21:00 departure
// 10.5 h after the arrival is reachable.
TEST(t2t_review, very_large_min_transfer_time_is_not_a_ban) {
  auto const tt =
      load({feed({{"NB1", 64.5, 23.5},
                  {"NB2", 64.5003, 23.5},
                  {"NB0", 64.6, 23.5},
                  {"NB9", 64.7, 23.5}},
                 {{"NBT1", "RNB1", {{"NB0", "10:00"}, {"NB1", "10:30"}}},
                  {"NBT2", "RNB2", {{"NB2", "21:00"}, {"NB9", "21:30"}}}},
                 "NB1,NB2,2,36000,,,,\n")});
  auto const res = search(tt, "NB0", "NB9", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("21:30"), begin(res)->dest_time_);
}

// ===========================================================================
// Walk hubs and rule hubs. Latent: no GTFS input produces it, because the
// fold adds an explicit default at every stop pair with voting rules (which
// makes the pair "ruled" and keeps its walk out of the walk hubs). So this
// test builds the rules directly, without the GTFS loader.
// ===========================================================================

// A rule states 10 min from A's virtual locations to everything at B: a cross
// product of 2 x 3 pairs, so it becomes a rule hub instead of cells. The walk
// A -> B takes 3 min. The walk hub A -> B takes the virtual locations along,
// which is only right where no slower rule speaks about the pair - vA1 -> B
// and vA2 -> vB1 have to stay at 10 min.
// Disabled: fails today (3 min) - rule_index (build_footpaths.cc) reads only
// transfer_rule_fps_ and does not see the pairs of rule hubs. Enable together
// with the fix (any_slower_at consulting the rule hubs [0, n_rule_hubs_)).
TEST(t2t_review, DISABLED_walk_hub_respects_slower_rule_hub) {
  auto tt = timetable{};
  loader::register_special_stations(tt);
  auto const add = [&](std::string_view const id, geo::latlng const pos,
                       location_type const type, location_idx_t const parent) {
    auto l = loader::location{};
    l.src_ = source_idx_t{0U};
    l.id_ = id;
    l.pos_ = pos;
    l.type_ = type;
    l.parent_ = parent;
    l.transfer_time_ = duration_t{2};
    auto const idx = loader::register_location(tt, l);
    if (parent != location_idx_t::invalid()) {
      tt.locations_.children_[parent].emplace_back(idx);
    }
    return idx;
  };
  auto const a =
      add("A", {50.0, 8.0}, location_type::kStation, location_idx_t::invalid());
  auto const b = add("B", {50.01, 8.0}, location_type::kStation,
                     location_idx_t::invalid());
  auto const first_virt = location_idx_t{tt.n_locations()};
  auto const va1 = add("", {50.0, 8.0}, location_type::kVirt, a);
  auto const va2 = add("", {50.0, 8.0}, location_type::kVirt, a);
  auto const vb1 = add("", {50.01, 8.0}, location_type::kVirt, b);
  auto const vb2 = add("", {50.01, 8.0}, location_type::kVirt, b);

  auto most_specific = hash_map<loader::transfer_pair, loader::candidate>{};
  for (auto const x : {va1, va2}) {
    for (auto const y : {b, vb1, vb2}) {
      most_specific[loader::transfer_pair{x, y}] =
          loader::candidate{.rank_ = 1U, .rule_idx_ = loader::rule_idx_t{0U}};
    }
  }
  auto durations = vector_map<loader::rule_idx_t, duration_t>{};
  durations.push_back(duration_t{10});
  loader::write_transfer_rules(tt, most_specific, durations, first_virt, true);

  // precondition: the rule is a hub, its pairs are no cells
  EXPECT_TRUE(
      utl::none_of(tt.locations_.transfer_rule_fps_[va1],
                   [&](footpath const fp) { return fp.target() == b; }));

  tt.locations_.preprocessing_footpaths_out_[a].emplace_back(b, duration_t{3});
  loader::build_footpaths(tt, {.adjust_footpaths_ = false,
                               .merge_dupes_intra_src_ = false,
                               .merge_dupes_inter_src_ = false});

  auto const fastest = [&](location_idx_t const from, location_idx_t const to) {
    auto best = footpath::kMaxDuration;
    routing::for_each_transfer<direction::kForward>(
        tt, nullptr, kDefaultProfile, from, [&](footpath const fp) {
          if (fp.target() == to) {
            best = std::min(best, fp.duration());
          }
          return true;
        });
    return best;
  };
  EXPECT_EQ(duration_t{3}, fastest(a, b));
  EXPECT_EQ(duration_t{10}, fastest(va1, b));
  EXPECT_EQ(duration_t{10}, fastest(va2, vb1));
}
