#include "gtest/gtest.h"

#include <optional>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"

using namespace nigiri;
using namespace date;
using nigiri::test::raptor_search;

namespace {

// Folding the pair defaults (loader::fold_pair_defaults) must not change what
// the rules state. Network: see test/rt/gtfsrt_transfer_rules_test.cc -
// F arrives S1 10:30; G leaves S2 10:40 (fallback GL 11:10), H leaves S1 10:31
// (fallback HL 11:01).
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

unixtime_t arrival(timetable const& tt, char const* from, char const* to) {
  auto const src = source_idx_t{0};
  auto const res = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = t("2019-05-01 10:00 Europe/Berlin"),
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                            0_minutes, 0U}}},
      direction::kForward);
  EXPECT_EQ(1U, res.size());
  return res.size() == 0U ? unixtime_t{} : begin(res)->dest_time_;
}

}  // namespace

// A trip rule that restates the default (2 min) is NOT redundant if a less
// specific rule says something else (route pair: 0 min): F -> H has 1 min.
TEST(gtfs, transfer_rules_fold_keeps_exception_to_faster_rule) {
  auto const tt = load("S1,S1,2,120,,,,\nS1,S1,2,0,RF,RH,,\nS1,S1,2,120,,,F,H");
  EXPECT_EQ(t("2019-05-01 11:30 Europe/Berlin"), arrival(tt, "A", "C"));
}

// ... same with a ban: RF may not change S1 -> S2, but the trip F may, at the
// default time.
TEST(gtfs, transfer_rules_fold_keeps_exception_to_ban) {
  auto const tt = load("S1,S2,2,120,,,,\nS1,S2,3,,RF,,,\nS1,S2,2,120,,,F,");
  EXPECT_EQ(t("2019-05-01 11:00 Europe/Berlin"), arrival(tt, "A", "B"));
}

// Rules stated for a station: their majority (15 min) is the default for the
// platform pairs below it as well, not only for the station itself.
TEST(gtfs, transfer_rules_fold_station_majority_reaches_platforms) {
  auto const tt = load("S,S,2,900,RF,RG,,\nS,S,2,900,RF,RH,,");
  EXPECT_EQ(t("2019-05-01 11:30 Europe/Berlin"), arrival(tt, "A", "B"));
}

// ... and an exception to that majority survives: F -> G is quick.
TEST(gtfs, transfer_rules_fold_station_majority_with_exception) {
  auto const tt = load("S,S,2,900,RF,RG,,\nS,S,2,900,RF,RH,,\nS,S,2,60,,,F,G");
  EXPECT_EQ(t("2019-05-01 11:00 Europe/Berlin"), arrival(tt, "A", "B"));
}

// A virtual location must not get walks of its own into other feeds: it
// leaves through its stop. S1 takes 5 min to change, the RF trips among
// themselves 0 min - their virtual location must not reach the other feed's
// stop Z (50 m away) faster than S1 does.
TEST(gtfs, transfer_rules_virtual_location_not_linked_to_other_feeds) {
  constexpr auto const kOtherFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG2,Agency 2,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
Z,Z,,50.0005,6.5,,,
Y,Y,,50.5,6.5,,,

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RZ,AG2,RZ,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RZ,X,TZ,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TZ,12:00:00,12:00:00,Z,1,0,0
TZ,12:30:00,12:30:00,Y,2,0,0
)";
  auto tt = timetable{};
  tt.date_range_ = {kDay, kDay + date::days{1}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable(
      {}, source_idx_t{0},
      loader::mem_dir::read(fmt::format(fmt::runtime(kFeed),
                                        "S1,S1,2,300,,,,\nS1,S1,2,0,RF,RF,,")),
      tt);
  loader::gtfs::load_timetable({}, source_idx_t{1},
                               loader::mem_dir::read(kOtherFeed), tt);
  loader::finalize(tt);

  auto const s1 = tt.locations_.location_id_to_idx_.at({"S1", source_idx_t{0}});
  auto const z = tt.locations_.location_id_to_idx_.at({"Z", source_idx_t{1}});
  // the whole transfer relation: footpaths and what the hubs hand out
  auto const walk = [&](location_idx_t const from) {
    auto d = std::optional<duration_t>{};
    auto const take = [&](footpath const fp) {
      if (fp.target() == z && (!d.has_value() || fp.duration() < *d)) {
        d = fp.duration();
      }
      return true;
    };
    for (auto const fp : tt.locations_.footpaths_out_[kDefaultProfile][from]) {
      take(fp);
    }
    routing::for_each_hub_source<direction::kBackward>(tt, kDefaultProfile,
                                                       from, take);
    return d;
  };
  ASSERT_TRUE(walk(s1).has_value());
  auto n_virts = 0U;
  for (auto const c : tt.locations_.children_[s1]) {
    if (tt.locations_.types_[c] == location_type::kVirt) {
      ++n_virts;
      EXPECT_TRUE(!walk(c).has_value() || *walk(c) >= *walk(s1))
          << "virtual location reaches Z in " << walk(c)->count()
          << " min, its stop in " << walk(s1)->count();
    }
  }
  EXPECT_NE(0U, n_virts);
}

// The search and the reconstruction have to agree on what a hub costs. S1 has
// a change time of 0 min, F and H sit on virtual locations of S1 (each named by
// an unrelated rule), so F -> H is derived by S1's hub at 0 min - also when the
// query asks for transfer time settings.
TEST(gtfs, transfer_rules_zero_minute_hub_with_transfer_time_settings) {
  auto const tt =
      load("S1,S1,2,0,,,,\nS1,S1,2,600,RF,RG,,\nS1,S1,2,600,RG,RH,,");
  auto const src = source_idx_t{0};
  auto const run = [&](routing::transfer_time_settings const tts) {
    return raptor_search(
        tt, nullptr,
        routing::query{
            .start_time_ = t("2019-05-01 10:00 Europe/Berlin"),
            .start_match_mode_ = routing::location_match_mode::kEquivalent,
            .dest_match_mode_ = routing::location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at({"A", src}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at({"C", src}),
                              0_minutes, 0U}},
            .transfer_time_settings_ = tts},
        direction::kForward);
  };
  auto const plain = run({});
  ASSERT_EQ(1U, plain.size());
  EXPECT_EQ(t("2019-05-01 11:00 Europe/Berlin"), begin(plain)->dest_time_);

  // whatever the settings make of a 0 min change: a journey, not a label the
  // reconstruction cannot retrace
  auto const padded = run({.default_ = false, .additional_time_ = 3_minutes});
  ASSERT_EQ(1U, padded.size());
  EXPECT_TRUE(begin(padded)->dest_time_ ==
                  t("2019-05-01 11:00 Europe/Berlin") ||
              begin(padded)->dest_time_ == t("2019-05-01 11:30 Europe/Berlin"));
}

// A GTFS stop group is a dummy stop at (0,0) that is equivalent to its
// members. Its "walk" to a member in Germany is thousands of kilometers - far
// more minutes than a duration holds - and must not wrap into a footpath.
TEST(gtfs, stop_group_at_null_island_gets_no_footpaths) {
  constexpr auto const kGroupFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
GRP,Group,,0.0,0.0,,,
M1,M1,,50.0,6.5,,,
M2,M2,,50.0,7.5,,,

# stop_group_elements.txt
stop_group_id,stop_id
GRP,M1
GRP,M2

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,AG,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,X,T,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T,10:00:00,10:00:00,M1,1,0,0
T,10:30:00,10:30:00,M2,2,0,0
)";
  for (auto const adjust_footpaths : {true, false}) {
    auto tt = timetable{};
    tt.date_range_ = {kDay, kDay + date::days{1}};
    loader::register_special_stations(tt);
    loader::gtfs::load_timetable({}, source_idx_t{0},
                                 loader::mem_dir::read(kGroupFeed), tt);
    loader::finalize(tt, {.adjust_footpaths_ = adjust_footpaths});

    auto const grp =
        tt.locations_.location_id_to_idx_.at({"GRP", source_idx_t{0}});
    ASSERT_FALSE(tt.locations_.equivalences_[grp].empty());
    EXPECT_TRUE(tt.locations_.footpaths_out_[kDefaultProfile][grp].empty());
    EXPECT_TRUE(tt.locations_.footpaths_in_[kDefaultProfile][grp].empty());
  }
}
