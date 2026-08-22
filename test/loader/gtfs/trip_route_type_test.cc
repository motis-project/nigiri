#include "gtest/gtest.h"

#include <set>

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

namespace {

// One agency, one calendar service running every day in the test interval.
constexpr auto const agency = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,http://db.de,Europe/Berlin
)"};

constexpr auto const stops = std::string_view{
    R"(stop_id,stop_name,stop_lat,stop_lon
S1,Stop 1,49.880,8.664
S2,Stop 2,49.878,8.661
S3,Stop 3,49.875,8.658
)"};

constexpr auto const calendar = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DAILY,1,1,1,1,1,1,1,20060701,20060731
)"};

// Single rail route (route_type=2). Trips carry a `trip_route_type` override:
//   TRAIN  -> empty   -> keeps route-level rail class
//   TRAIN2 -> empty   -> keeps route-level rail class (merges with TRAIN)
//   BUS    -> 3       -> basic bus           -> kBus
//   REPL   -> 714     -> rail replacement bus -> kBus (merges with BUS)
// All trips share the same stop sequence, so class is the only differentiator.
constexpr auto const routes = std::string_view{
    R"(route_id,agency_id,route_short_name,route_long_name,route_type
R1,DB,R1,Rail Route,2
)"};

constexpr auto const trips = std::string_view{
    R"(route_id,service_id,trip_id,trip_route_type
R1,DAILY,TRAIN,
R1,DAILY,TRAIN2,
R1,DAILY,BUS,3
R1,DAILY,REPL,714
)"};

constexpr auto const stop_times = std::string_view{
    R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence
TRAIN,08:00:00,08:00:00,S1,1
TRAIN,08:10:00,08:10:00,S2,2
TRAIN,08:20:00,08:20:00,S3,3
TRAIN2,09:00:00,09:00:00,S1,1
TRAIN2,09:10:00,09:10:00,S2,2
TRAIN2,09:20:00,09:20:00,S3,3
BUS,10:00:00,10:00:00,S1,1
BUS,10:10:00,10:10:00,S2,2
BUS,10:20:00,10:20:00,S3,3
REPL,11:00:00,11:00:00,S1,1
REPL,11:10:00,11:10:00,S2,2
REPL,11:20:00,11:20:00,S3,3
)"};

mem_dir test_files() {
  using std::filesystem::path;
  return mem_dir{{{path{kAgencyFile}, std::string{agency}},
                  {path{kStopFile}, std::string{stops}},
                  {path{kCalenderFile}, std::string{calendar}},
                  {path{kRoutesFile}, std::string{routes}},
                  {path{kTripsFile}, std::string{trips}},
                  {path{kStopTimesFile}, std::string{stop_times}}}};
}

}  // namespace

// `trip_route_type` splits a single rail route into two router routes: the
// non-overridden trips stay rail (kRegional), the overridden trips become bus
// (kBus) -- even though they share route, stop sequence and calendar.
TEST(gtfs, trip_route_type_splits_route_by_class) {
  auto const files = test_files();
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  tt.date_range_ = {2006_y / 7 / 1, 2006_y / 8 / 1};
  load_timetable({}, source_idx_t{0U}, files, tt);
  finalize(tt);

  auto classes = std::multiset<clasz>{};
  for (auto const c : tt.route_clasz_) {
    classes.insert(c);
  }

  EXPECT_EQ(2U, classes.size());
  EXPECT_EQ(1U, classes.count(clasz::kRegional));
  EXPECT_EQ(1U, classes.count(clasz::kBus));
}

// Without any `trip_route_type` override, the same feed yields a single rail
// route -- guards against the override path changing default behaviour.
TEST(gtfs, trip_route_type_absent_keeps_single_route) {
  constexpr auto const trips_no_override = std::string_view{
      R"(route_id,service_id,trip_id
R1,DAILY,TRAIN
R1,DAILY,TRAIN2
)"};

  using std::filesystem::path;
  auto const files =
      mem_dir{{{path{kAgencyFile}, std::string{agency}},
               {path{kStopFile}, std::string{stops}},
               {path{kCalenderFile}, std::string{calendar}},
               {path{kRoutesFile}, std::string{routes}},
               {path{kTripsFile}, std::string{trips_no_override}},
               {path{kStopTimesFile}, std::string{stop_times}}}};
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  tt.date_range_ = {2006_y / 7 / 1, 2006_y / 8 / 1};
  load_timetable({}, source_idx_t{0U}, files, tt);
  finalize(tt);

  auto classes = std::multiset<clasz>{};
  for (auto const c : tt.route_clasz_) {
    classes.insert(c);
  }

  EXPECT_EQ(1U, classes.size());
  EXPECT_EQ(1U, classes.count(clasz::kRegional));
}
