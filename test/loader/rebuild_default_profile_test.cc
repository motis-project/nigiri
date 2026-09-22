#include "gtest/gtest.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace nigiri;
using namespace date;
using nigiri::test::raptor_search;

namespace {

// loader::rebuild_default_profile replaces the walks of the default profile
// (a street router knows better than the loader's beelines) - and nothing a
// rule states.
//
// F arrives S1 10:30. G leaves S2 10:40 (fallback GL 11:10) to B,
// H leaves S1 10:31 (fallback HL 11:01) to C, K leaves T 10:45 to E.
// T is 450 m from S: the same feed, not the same station, so the loader
// connects the two with nothing.
constexpr auto const kFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,50.0,6.0,,,
S,S,,50.0,6.5,,1,
S1,S1,,50.0001,6.5,,,S
S2,S2,,50.0002,6.5,,,S
S3,S3,,50.0003,6.5,,,S
S4,S4,,50.0004,6.5,,,S
T,T,,50.0045,6.5,,,
B,B,,50.0,7.0,,,
C,C,,50.0,7.5,,,
E,E,,50.0,8.0,,,

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RF,AG,RF,,,3
RG,AG,RG,,,3
RH,AG,RH,,,3
RK,AG,RK,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RF,X,F,,
RG,X,G,,
RG,X,GL,,
RH,X,H,,
RH,X,HL,,
RK,X,K,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
F,10:00:00,10:00:00,A,1,0,0
F,10:30:00,10:30:00,S1,2,0,0
G,10:40:00,10:40:00,S2,1,0,0
G,11:00:00,11:00:00,B,2,0,0
GL,11:10:00,11:10:00,S2,1,0,0
GL,11:30:00,11:30:00,B,2,0,0
H,10:31:00,10:31:00,S1,1,0,0
H,11:00:00,11:00:00,C,2,0,0
HL,11:01:00,11:01:00,S1,1,0,0
HL,11:30:00,11:30:00,C,2,0,0
K,10:45:00,10:45:00,T,1,0,0
K,11:05:00,11:05:00,E,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
{}
)";

constexpr auto const kDay = date::sys_days{2019_y / May / 1};
constexpr auto const kSrc = source_idx_t{0};

timetable load(std::string_view const transfers) {
  auto tt = timetable{};
  tt.date_range_ = {kDay, kDay + date::days{1}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable(
      {}, kSrc,
      loader::mem_dir::read(fmt::format(fmt::runtime(kFeed), transfers)), tt);
  loader::finalize(tt);
  return tt;
}

unixtime_t t(char const* s) { return parse_time_tz(s, "%Y-%m-%d %H:%M %Z"); }

std::optional<unixtime_t> arrival(timetable const& tt, char const* to) {
  auto const res = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = t("2019-05-01 10:00 Europe/Berlin"),
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .start_ = {{tt.locations_.location_id_to_idx_.at({"A", kSrc}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at({to, kSrc}),
                            0_minutes, 0U}}},
      direction::kForward);
  return res.size() == 0U ? std::nullopt
                          : std::optional{begin(res)->dest_time_};
}

// Walks as a street router hands them over: between stops, both ways.
struct walk {
  char const* a_;
  char const* b_;
  int minutes_;
};

void rebuild(timetable& tt, std::vector<walk> const& walks) {
  auto w = vector_map<location_idx_t, std::vector<footpath>>{};
  w.resize(tt.n_locations());
  for (auto const& x : walks) {
    auto const a = tt.locations_.location_id_to_idx_.at({x.a_, kSrc});
    auto const b = tt.locations_.location_id_to_idx_.at({x.b_, kSrc});
    w[a].emplace_back(b, duration_t{x.minutes_});
    w[b].emplace_back(a, duration_t{x.minutes_});
  }
  loader::rebuild_default_profile(tt, w);
}

unsigned n_virts(timetable const& tt) {
  return static_cast<unsigned>(std::count(begin(tt.locations_.types_),
                                          end(tt.locations_.types_),
                                          location_type::kVirt));
}

#define kViaG t("2019-05-01 11:00 Europe/Berlin")
#define kViaGL t("2019-05-01 11:30 Europe/Berlin")
#define kViaHL t("2019-05-01 11:30 Europe/Berlin")
#define kViaK t("2019-05-01 11:05 Europe/Berlin")

}  // namespace

TEST(rebuild_default_profile, walk_replaces_beeline) {
  auto tt = load("");
  EXPECT_EQ(kViaG, arrival(tt, "B"));  // beeline S1 -> S2: 2 min
  rebuild(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
  rebuild(tt, {{"S1", "S2", 9}});
  EXPECT_EQ(kViaG, arrival(tt, "B"));
}

// The pair is new to the layer and to the lower bound graph.
TEST(rebuild_default_profile, walk_the_loader_did_not_have) {
  auto tt = load("");
  EXPECT_EQ(std::nullopt, arrival(tt, "E"));
  rebuild(tt, {{"S1", "T", 6}});
  EXPECT_EQ(kViaK, arrival(tt, "E"));
}

TEST(rebuild_default_profile, walk_the_router_did_not_find) {
  auto tt = load("");
  rebuild(tt, {});
  EXPECT_EQ(std::nullopt, arrival(tt, "B"));
  EXPECT_EQ(kViaHL, arrival(tt, "C"));  // changing at S1 is no walk
}

TEST(rebuild_default_profile, rule_beats_slower_walk) {
  auto tt = load("S1,S2,2,180,,,,");
  rebuild(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(kViaG, arrival(tt, "B"));
}

TEST(rebuild_default_profile, rule_beats_faster_walk) {
  auto tt = load("S1,S2,2,900,,,,");
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
  rebuild(tt, {{"S1", "S2", 1}});
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
}

TEST(rebuild_default_profile, forbidden_trip_pair_stays_forbidden) {
  auto tt = load("S1,S2,2,120,,,,\nS1,S2,3,,,,F,G");
  ASSERT_NE(0U, n_virts(tt));
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
  rebuild(tt, {{"S1", "S2", 1}});
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
}

// A rule for a whole station is a hub, which never overrides: the routing takes
// the minimum of hub and footpath, so the walk underneath has to go.
TEST(rebuild_default_profile, station_rule_beats_faster_walk) {
  auto tt = load("S,S,2,900,,,,");
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
  EXPECT_EQ(kViaHL, arrival(tt, "C"));
  rebuild(tt, {{"S1", "S2", 1}});
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
  EXPECT_EQ(kViaHL, arrival(tt, "C"));
}

// F stops at a virtual location below S1 (the F -> H rule). It owns no walks:
// the walk hubs hand it those of S1, so they have to follow the new walks.
TEST(rebuild_default_profile, virtual_location_walks_like_its_stop) {
  auto tt = load("S1,S1,2,120,,,,\nS1,S1,2,1800,,,F,H");
  ASSERT_NE(0U, n_virts(tt));
  EXPECT_EQ(kViaHL, arrival(tt, "C"));  // the rule: 30 min
  EXPECT_EQ(kViaG, arrival(tt, "B"));  // beeline
  rebuild(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(kViaGL, arrival(tt, "B"));
  EXPECT_EQ(kViaHL, arrival(tt, "C"));
  rebuild(tt, {{"S1", "S2", 3}, {"S1", "T", 6}});
  EXPECT_EQ(kViaG, arrival(tt, "B"));
  EXPECT_EQ(kViaK, arrival(tt, "E"));
}

// The rule-derived hubs are kept, not derived a second time.
TEST(rebuild_default_profile, is_repeatable) {
  auto tt = load("S,S,2,120,,,,\nS,S,2,1800,,,F,H");
  ASSERT_NE(0U, n_virts(tt));
  auto const n_rule_hubs = tt.locations_.n_rule_hubs_;
  ASSERT_NE(0U, n_rule_hubs);

  rebuild(tt, {{"S1", "T", 6}});
  auto const n_hubs = tt.locations_.hub_in_[kDefaultProfile].size();
  auto const n_fps = tt.locations_.footpaths_out_[kDefaultProfile].data_.size();
  rebuild(tt, {{"S1", "T", 6}});

  EXPECT_EQ(n_rule_hubs, tt.locations_.n_rule_hubs_);
  EXPECT_EQ(n_hubs, tt.locations_.hub_in_[kDefaultProfile].size());
  EXPECT_EQ(n_fps, tt.locations_.footpaths_out_[kDefaultProfile].data_.size());
  EXPECT_EQ(kViaG, arrival(tt, "B"));  // the station rule: 2 min
  EXPECT_EQ(kViaHL, arrival(tt, "C"));
  EXPECT_EQ(kViaK, arrival(tt, "E"));
}
