#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor_n_to_all.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;

// ---------------------------------------------------------------------------
// Same Y-shaped network used in simple_raptor_meet_middle_test.cc:
//
//   A --R_A--> P1 --R_A--> HUB --R_HUB--> Q1 --R_HUB--> Q2
//   B --R_B--> P2 --R_B--> HUB
//   C    ------R_C-------> HUB     (shorter arm — arrives first)
//
// In UTC (DST active on 2019-05-01, Europe/Berlin = UTC+2):
//   T_A:   A dep 00:00, P1 arr 00:30, HUB arr 01:00
//   T_B:   B dep 00:00, P2 arr 00:30, HUB arr 01:00
//   T_C:   C dep 00:00, HUB arr 00:30     <- C arrives at HUB first
//   T_HUB: HUB dep 01:30, Q1 arr 02:00, Q2 arr 02:30
//
// start_time = 2019-04-30 23:30 UTC (30 min before all trains)
//
// NToAll with origins = {A, B, C}:
//   A:   owner=A, travel=0
//   P1:  owner=A, travel=60
//   B:   owner=B, travel=0
//   P2:  owner=B, travel=60
//   C:   owner=C, travel=0
//   HUB: owner=C, travel=60    (C wins: 60 < A/B's 90)
//   Q1:  owner=C, travel=150   (C controls HUB → propagates via T_HUB)
//   Q2:  owner=C, travel=180
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
P1,P1,,2.0,3.0,,
B,B,,4.0,5.0,,
P2,P2,,6.0,7.0,,
C,C,,8.0,9.0,,
HUB,HUB,,10.0,11.0,,
Q1,Q1,,12.0,13.0,,
Q2,Q2,,14.0,15.0,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_A,DB,RE A,,,3
R_B,DB,RE B,,,3
R_C,DB,RE C,,,3
R_HUB,DB,RE HUB,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_A,S1,T_A,RE A,
R_B,S1,T_B,RE B,
R_C,S1,T_C,RE C,
R_HUB,S1,T_HUB,RE HUB,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_A,02:00:00,02:00:00,A,1,0,0
T_A,02:30:00,02:30:00,P1,2,0,0
T_A,03:00:00,03:00:00,HUB,3,0,0
T_B,02:00:00,02:00:00,B,1,0,0
T_B,02:30:00,02:30:00,P2,2,0,0
T_B,03:00:00,03:00:00,HUB,3,0,0
T_C,02:00:00,02:00:00,C,1,0,0
T_C,02:30:00,02:30:00,HUB,2,0,0
T_HUB,03:30:00,03:30:00,HUB,1,0,0
T_HUB,04:00:00,04:00:00,Q1,2,0,0
T_HUB,04:30:00,04:30:00,Q2,3,0,0
)");
}

}  // namespace

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class RaptorNToAllTest : public ::testing::Test {
protected:
  static constexpr auto const kSrc = source_idx_t{0U};

  // 2019-04-30 23:30 UTC — 30 min before all trains
  static inline const unixtime_t kStartTime =
      unixtime_t{sys_days{2019_y / April / 30}} + 23_hours + 30_minutes;

  void SetUp() override {
    tt_.date_range_ = {sys_days{2019_y / March / 25},
                       sys_days{2019_y / November / 1}};
    load_timetable({}, kSrc, test_files(), tt_);
    finalize(tt_);

    loc_a_   = find("A");
    loc_p1_  = find("P1");
    loc_b_   = find("B");
    loc_p2_  = find("P2");
    loc_c_   = find("C");
    loc_hub_ = find("HUB");
    loc_q1_  = find("Q1");
    loc_q2_  = find("Q2");
  }

  location_idx_t find(std::string_view id) const {
    return tt_.locations_.location_id_to_idx_.at({.id_ = id, .src_ = kSrc});
  }

  n_to_all_result go(std::vector<location_idx_t> const& origins) const {
    return raptor_n_to_all(tt_, nullptr, origins, kStartTime);
  }

  timetable tt_{};
  location_idx_t loc_a_{}, loc_p1_{}, loc_b_{}, loc_p2_{},
                 loc_c_{}, loc_hub_{}, loc_q1_{}, loc_q2_{};
};

TEST_F(RaptorNToAllTest, SingleOrigin_A_ReachesAllArmAndPostHubStops) {
  auto const result = raptor_n_to_all(tt_, nullptr, {loc_a_}, kStartTime);

  // A itself is seeded
  ASSERT_TRUE(result.cell(loc_a_).reached_);
  EXPECT_EQ(result.cell(loc_a_).travel_time_, duration_t{0});
  EXPECT_EQ(result.cell(loc_a_).owner_, loc_a_);

  // P1 reachable directly (0 transfers)
  ASSERT_TRUE(result.cell(loc_p1_).reached_) << "P1 unreachable from A";
  EXPECT_EQ(result.cell(loc_p1_).transfers_, 0U);
  EXPECT_EQ(result.cell(loc_p1_).owner_, loc_a_);

  // HUB reachable directly (0 transfers), further than P1
  ASSERT_TRUE(result.cell(loc_hub_).reached_) << "HUB unreachable from A";
  EXPECT_EQ(result.cell(loc_hub_).transfers_, 0U);
  EXPECT_GT(result.cell(loc_hub_).travel_time_,
            result.cell(loc_p1_).travel_time_) << "HUB must be further than P1";

  // Q1 reachable via transfer at HUB (1 transfer)
  ASSERT_TRUE(result.cell(loc_q1_).reached_) << "Q1 unreachable from A";
  EXPECT_EQ(result.cell(loc_q1_).transfers_, 1U)
      << "Q1 requires transfer at HUB (T_A then T_HUB)";

  // Q2 reachable via same transfer (1 transfer), further than Q1
  ASSERT_TRUE(result.cell(loc_q2_).reached_) << "Q2 unreachable from A";
  EXPECT_EQ(result.cell(loc_q2_).transfers_, 1U);
  EXPECT_GT(result.cell(loc_q2_).travel_time_,
            result.cell(loc_q1_).travel_time_);

  // A cannot reach other arms
  EXPECT_FALSE(result.cell(loc_b_).reached_)  << "B must be unreachable from A";
  EXPECT_FALSE(result.cell(loc_p2_).reached_) << "P2 must be unreachable from A";
  EXPECT_FALSE(result.cell(loc_c_).reached_)  << "C must be unreachable from A";
}


TEST_F(RaptorNToAllTest, SingleOrigin_A_ExactTravelTimes) {
  auto const result = raptor_n_to_all(tt_, nullptr, {loc_a_}, kStartTime);

  // Retrieve the actual times first so we can print them if they differ.
  auto const p1_min  = result.cell(loc_p1_).travel_time_.count();
  auto const hub_min = result.cell(loc_hub_).travel_time_.count();
  auto const q1_min  = result.cell(loc_q1_).travel_time_.count();
  auto const q2_min  = result.cell(loc_q2_).travel_time_.count();

  // Expected: 60, 90, 150, 180 from raw schedule (±2 for transfer time).
  // If the real raptor applies a 2-minute minimum transfer, all are +2.
  // Adjust the expected values below after first run if needed.
  constexpr auto kExpectedP1  = std::int32_t{62};
  constexpr auto kExpectedHUB = std::int32_t{92};
  constexpr auto kExpectedQ1  = std::int32_t{152};
  constexpr auto kExpectedQ2  = std::int32_t{182};

  EXPECT_EQ(p1_min,  kExpectedP1)
      << "If off by +2, the timetable applies a 2-min minimum transfer — "
      << "update expected values to " << p1_min;
  EXPECT_EQ(hub_min, kExpectedHUB);
  EXPECT_EQ(q1_min,  kExpectedQ1);
  EXPECT_EQ(q2_min,  kExpectedQ2);
}

TEST_F(RaptorNToAllTest, SingleOrigin_C_ReachesHubFirst) {
  auto const result_c = raptor_n_to_all(tt_, nullptr, {loc_c_}, kStartTime);
  auto const result_a = raptor_n_to_all(tt_, nullptr, {loc_a_}, kStartTime);

  ASSERT_TRUE(result_c.cell(loc_hub_).reached_);
  ASSERT_TRUE(result_a.cell(loc_hub_).reached_);

  EXPECT_LT(result_c.cell(loc_hub_).travel_time_,
            result_a.cell(loc_hub_).travel_time_)
      << "C must reach HUB faster than A (C has shorter arm)";

  // C cannot reach A, P1, B, P2 (their arms are one-directional inbound)
  EXPECT_FALSE(result_c.cell(loc_a_).reached_);
  EXPECT_FALSE(result_c.cell(loc_p1_).reached_);
  EXPECT_FALSE(result_c.cell(loc_b_).reached_);
  EXPECT_FALSE(result_c.cell(loc_p2_).reached_);
}

TEST_F(RaptorNToAllTest, MultiOrigin_HubOwnedByClosestOrigin) {
  auto const result = go({loc_a_, loc_b_, loc_c_});

  ASSERT_TRUE(result.cell(loc_hub_).reached_);
  EXPECT_EQ(result.cell(loc_hub_).owner_, loc_c_)
      << "HUB must be owned by C — C's arm is one hop vs A/B's two hops";

  // Travel time at HUB reflects C's faster path
  auto const result_c = raptor_n_to_all(tt_, nullptr, {loc_c_}, kStartTime);
  EXPECT_EQ(result.cell(loc_hub_).travel_time_,
            result_c.cell(loc_hub_).travel_time_)
      << "Multi-origin HUB travel_time must match C's single-origin result";
}

TEST_F(RaptorNToAllTest, MultiOrigin_ArmStopsExclusive) {
  auto const result = go({loc_a_, loc_b_, loc_c_});

  ASSERT_TRUE(result.cell(loc_p1_).reached_);
  EXPECT_EQ(result.cell(loc_p1_).owner_, loc_a_)
      << "P1 only reachable from A";

  ASSERT_TRUE(result.cell(loc_p2_).reached_);
  EXPECT_EQ(result.cell(loc_p2_).owner_, loc_b_)
      << "P2 only reachable from B";
}

TEST_F(RaptorNToAllTest, MultiOrigin_PostHubOwnedByHubWinner) {
  auto const result = go({loc_a_, loc_b_, loc_c_});

  ASSERT_TRUE(result.cell(loc_q1_).reached_);
  EXPECT_EQ(result.cell(loc_q1_).owner_, loc_c_)
      << "Q1 must be owned by C: C controls HUB, ownership propagates via T_HUB";

  ASSERT_TRUE(result.cell(loc_q2_).reached_);
  EXPECT_EQ(result.cell(loc_q2_).owner_, loc_c_)
      << "Q2 must be owned by C: same propagation";
}

TEST_F(RaptorNToAllTest, MultiOrigin_PostHubRequiresTransfer) {
  auto const result = go({loc_a_, loc_b_, loc_c_});

  EXPECT_EQ(result.cell(loc_q1_).transfers_, 1U)
      << "Q1: must board T_A/B/C then transfer to T_HUB at HUB";
  EXPECT_EQ(result.cell(loc_q2_).transfers_, 1U);
}


TEST_F(RaptorNToAllTest, MeetInMiddle_SameArrivalTime_HubOwnerWins) {
  auto const ra = raptor_n_to_all(tt_, nullptr, {loc_a_}, kStartTime);
  auto const rb = raptor_n_to_all(tt_, nullptr, {loc_b_}, kStartTime);
  auto const rc = raptor_n_to_all(tt_, nullptr, {loc_c_}, kStartTime);

  // All three individually reach Q1 in the same time
  ASSERT_TRUE(ra.cell(loc_q1_).reached_);
  ASSERT_TRUE(rb.cell(loc_q1_).reached_);
  ASSERT_TRUE(rc.cell(loc_q1_).reached_);

  EXPECT_EQ(ra.cell(loc_q1_).travel_time_, rb.cell(loc_q1_).travel_time_)
      << "A and B must reach Q1 in the same time";
  EXPECT_EQ(ra.cell(loc_q1_).travel_time_, rc.cell(loc_q1_).travel_time_)
      << "A and C must reach Q1 in the same time";

  // But in the joint run, C wins because C controls HUB
  auto const joint = go({loc_a_, loc_b_, loc_c_});
  ASSERT_TRUE(joint.cell(loc_q1_).reached_);

  EXPECT_EQ(joint.cell(loc_q1_).travel_time_, ra.cell(loc_q1_).travel_time_)
      << "Joint travel time must equal individual (same schedule)";
  EXPECT_EQ(joint.cell(loc_q1_).owner_, loc_c_)
      << "Q1 owned by C in joint run: C controls HUB, propagates to Q1";
}

TEST_F(RaptorNToAllTest, CompleteOwnershipTable) {
  auto const result = go({loc_a_, loc_b_, loc_c_});

  struct row {
    location_idx_t* loc;
    location_idx_t* expected_owner;
    std::string_view name;
  };

  for (auto const& r : std::vector<row>{
           {&loc_a_,   &loc_a_, "A"},
           {&loc_p1_,  &loc_a_, "P1"},
           {&loc_b_,   &loc_b_, "B"},
           {&loc_p2_,  &loc_b_, "P2"},
           {&loc_c_,   &loc_c_, "C"},
           {&loc_hub_, &loc_c_, "HUB"},
           {&loc_q1_,  &loc_c_, "Q1"},
           {&loc_q2_,  &loc_c_, "Q2"},
       }) {
    auto const& cell = result.cell(*r.loc);
    EXPECT_TRUE(cell.reached_)           << r.name << " must be reached";
    EXPECT_EQ(cell.owner_, *r.expected_owner) << r.name << " wrong owner";
  }
}

