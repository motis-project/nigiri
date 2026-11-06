#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor_n_to_all.h"
#include "nigiri/special_stations.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;

// ---------------------------------------------------------------------------
// n-to-all minimum travel time: raptor_n_to_all(tt, rtt, query).
// Result: n_to_all_result::at(stop), i.e. raptor_state::min_[stop].
//
// Network (local time, Europe/Berlin = UTC+2 on 2019-05-01)
//
//   T_C1:  C 12:00 -> B 12:25     (25 min ride, leaves first)
//   T_A1:  A 12:10 -> B 12:30     (20 min ride, leaves later)
//   T_D1:  D 12:05 -> M 12:15     \ one transfer at M (5 min slack)
//   T_M1:  M 12:20 -> E 12:30     /
//
// Origins: A, C, D.   Interval: 12:00 .. 12:15 local = 10:00 .. 10:15 UTC.
//
// A label in best_ is "ready to continue", i.e. vehicle arrival plus the
// stop's transfer time. ready() adds that, so the expected values do not
// depend on the default transfer time of the loader.
// ---------------------------------------------------------------------------

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
M,M,,8.0,9.0,,
E,E,,10.0,11.0,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_A,DB,RE A,,,3
R_C,DB,RE C,,,3
R_D,DB,RE D,,,3
R_M,DB,RE M,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_A,S1,T_A1,RE A,
R_C,S1,T_C1,RE C,
R_D,S1,T_D1,RE D,
R_M,S1,T_M1,RE M,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_A1,12:10:00,12:10:00,A,1,0,0
T_A1,12:30:00,12:30:00,B,2,0,0
T_C1,12:00:00,12:00:00,C,1,0,0
T_C1,12:25:00,12:25:00,B,2,0,0
T_D1,12:05:00,12:05:00,D,1,0,0
T_D1,12:15:00,12:15:00,M,2,0,0
T_M1,12:20:00,12:20:00,M,1,0,0
T_M1,12:30:00,12:30:00,E,2,0,0
)");
}

}  // namespace

class NToAllMinTest : public ::testing::Test {
protected:
  static constexpr auto const kSrc = source_idx_t{0U};

  void SetUp() override {
    tt_.date_range_ = {date::sys_days{2019_y / March / 25},
                       date::sys_days{2019_y / November / 1}};
    register_special_stations(tt_);
    load_timetable({}, kSrc, test_files(), tt_);
    finalize(tt_);
    a_ = find("A");
    b_ = find("B");
    c_ = find("C");
    d_ = find("D");
    e_ = find("E");
  }

  location_idx_t find(std::string_view id) const {
    return tt_.locations_.location_id_to_idx_.at({.id_ = id, .src_ = kSrc});
  }

  static unixtime_t from() {
    return unixtime_t{date::sys_days{2019_y / May / 1} + std::chrono::hours{10}};
  }
  static start_time_t window(std::int32_t const minutes) {
    return interval<unixtime_t>{from(),
                                from() + std::chrono::minutes{minutes}};
  }

  // vehicle ride in minutes -> value stored in best_ at l
  duration_t ready(location_idx_t const l, std::int32_t const ride) const {
    return duration_t{ride + tt_.locations_.transfer_time_[l].count()};
  }

  n_to_all_result run(start_time_t const start_time,
                 duration_t const max_travel = duration_t{120}) const {
    auto q = query{};
    q.start_time_ = start_time;
    q.start_match_mode_ = location_match_mode::kExact;
    q.use_start_footpaths_ = false;
    q.start_ = {offset{a_, 0_minutes, 0U}, offset{c_, 0_minutes, 0U},
                offset{d_, 0_minutes, 0U}};
    q.max_transfers_ = 3U;
    q.max_travel_time_ = max_travel;
    return raptor_n_to_all(tt_, nullptr, std::move(q));
  }

  timetable tt_{};
  location_idx_t a_{}, b_{}, c_{}, d_{}, e_{};
};

// C arrives at B first (12:25) but A has the shorter ride (20 min): the
// minimum belongs to A, and it departs 10 minutes into the window.
TEST_F(NToAllMinTest, FastestRideBeatsEarliestArrival) {
  auto const r = run(window(15));
  auto const& b = r.at(b_);
  ASSERT_TRUE(b.reached());

  EXPECT_EQ(b.owner_, a_);
  EXPECT_EQ(b.dur_, ready(b_, 20));
  EXPECT_EQ(b.departure_, from() + std::chrono::minutes{10});
  EXPECT_EQ(b.transfers_, 0U);
}

TEST_F(NToAllMinTest, OriginsOwnThemselvesWithZeroDuration) {
  auto const r = run(window(15));
  for (auto const o : {a_, c_, d_}) {
    ASSERT_TRUE(r.at(o).reached());
    EXPECT_EQ(r.at(o).owner_, o);
    EXPECT_EQ(r.at(o).dur_, duration_t{0});
    EXPECT_EQ(r.at(o).transfers_, 0U);
  }
}

// The fastest ride may need a transfer and start in the middle of the window.
TEST_F(NToAllMinTest, FastestRideWithTransfer) {
  auto const r = run(window(15));
  auto const& e = r.at(e_);
  ASSERT_TRUE(e.reached());

  EXPECT_EQ(e.owner_, d_);
  EXPECT_EQ(e.dur_, ready(e_, 25)) << "12:05 -> 12:30";
  EXPECT_EQ(e.departure_, from() + std::chrono::minutes{5});
  EXPECT_EQ(e.transfers_, 1U);
}

// A single point in time is a plain one-to-all: the owner is whoever is there
// first, the duration is counted from `from()`, nobody gets a head start.
TEST_F(NToAllMinTest, PointInTimeIsPlainOneToAll) {
  auto const r = run(from());
  EXPECT_EQ(r.searched_.from_, from());
  EXPECT_EQ(r.searched_.to_, from());

  auto const& b = r.at(b_);
  ASSERT_TRUE(b.reached());
  EXPECT_EQ(b.owner_, c_) << "C: 12:25, A: 12:30";
  EXPECT_EQ(b.dur_, ready(b_, 25));
  EXPECT_EQ(b.departure_, from());

  auto const& e = r.at(e_);
  ASSERT_TRUE(e.reached());
  EXPECT_EQ(e.owner_, d_);
  EXPECT_EQ(e.dur_, ready(e_, 30)) << "waits for the 12:05 train";
}

// max_travel_time bounds every run, measured from that run's departure.
// With the bound at A's ride, C's longer ride and D's 25 min trip are cut.
TEST_F(NToAllMinTest, MaxTravelTimeBoundsEveryRun) {
  auto const r = run(window(15), ready(b_, 20));
  auto const& b = r.at(b_);
  ASSERT_TRUE(b.reached());
  EXPECT_EQ(b.owner_, a_);
  EXPECT_EQ(b.dur_, ready(b_, 20));
  EXPECT_FALSE(r.at(e_).reached());

  auto const tight = run(window(15), duration_t{10});
  EXPECT_FALSE(tight.at(b_).reached());
  EXPECT_TRUE(tight.at(a_).reached());
}

TEST_F(NToAllMinTest, EveryReachedStopHasAValidOwner) {
  auto const r = run(window(15));
  auto n_reached = 0U;
  for (auto l = location_idx_t{0U}; to_idx(l) != tt_.n_locations(); ++l) {
    auto const& m = r.at(l);
    if (!m.reached()) {
      continue;
    }
    ++n_reached;
    EXPECT_TRUE(m.owner_ == a_ || m.owner_ == c_ || m.owner_ == d_)
        << to_idx(l);
    EXPECT_GE(m.dur_, duration_t{0}) << to_idx(l);
    EXPECT_GE(m.departure_, from()) << to_idx(l);
    EXPECT_LE(m.departure_, from() + std::chrono::minutes{15}) << to_idx(l);
  }
  EXPECT_EQ(n_reached, 6U) << "A, B, C, D, M and E";
}
//;
TEST_F(NToAllMinTest, SearchedIntervalIsTheRequestedOne) {
  auto const r = run(window(15));
  EXPECT_EQ(r.searched_.from_, from());
  EXPECT_EQ(r.searched_.to_, from() + std::chrono::minutes{15});
}
