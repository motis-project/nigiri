#include "gtest/gtest.h"

#include "utl/enumerate.h"

#include "nigiri/timetable.h"

#include "../util.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;

mem_dir adjacency_test_tt() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
SID,20260227,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A1,A1,,,,,,
A2,A2,,,,,,
A3,A3,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,MTA,A,A,A1 -> A2 -> A3,0
F,MTA,F,F,A2 -> A3,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
A,SID,A,A,0
F,SID,F,F,0

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
A,08:00,08:00,A1,0,0,0
A,08:30,08:30,A2,1,0,0
A,08:45,08:45,A3,1,0,0
F,09:00,09:00,A2,0,0,0
F,09:05,09:05,A3,1,0,0
)__");
}

constexpr auto kGtfsDateRange = interval{sys_days{2026_y / February / 27},
                                         sys_days{2026_y / February / 28}};

TEST(loader, lb_adjacency) {
  auto const tt = load_gtfs(adjacency_test_tt, kGtfsDateRange);

  auto const A1 = tt.find(location_id{"A1", source_idx_t{}}).value();
  auto const A2 = tt.find(location_id{"A2", source_idx_t{}}).value();
  auto const A3 = tt.find(location_id{"A3", source_idx_t{}}).value();

  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A1].size(), 0U);
  ASSERT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A2].size(), 1U);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A2][0].l_, A1);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A2][0].pt_duration_, 30U);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A3].size(), 2U);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A3][0].l_, A1);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A3][0].pt_duration_, 45U);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A3][1].l_, A2);
  EXPECT_EQ(tt.fwd_lb_adjacency_[kDefaultProfile][A3][1].pt_duration_, 5U);

  ASSERT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A1].size(), 2U);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A1][0].l_, A2);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A1][0].pt_duration_, 30U);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A1][1].l_, A3);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A1][1].pt_duration_, 45U);
  ASSERT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A2].size(), 1U);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A2][0].l_, A3);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A2][0].pt_duration_, 5U);
  EXPECT_EQ(tt.bwd_lb_adjacency_[kDefaultProfile][A3].size(), 0U);
}