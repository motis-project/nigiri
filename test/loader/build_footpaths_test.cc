#include "boost/chrono/duration.hpp"

#include "gtest/gtest.h"

#include "utl/enumerate.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

// ROUTING CONNECTIONS:
// 10:00 - 11:00 A-C    airplane direct
// 10:00 - 12:00 A-B-C  train, one transfer
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type

# calendar_dates.txt
service_id,date,exception_type

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
B,A,2,180
C,B,2,180
B,C,2,300
)"sv;

}  // namespace

TEST(loader, build_footpaths) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({.default_tz_ = "Europe/Berlin"},
                               source_idx_t{0},
                               loader::mem_dir::read(test_files), tt);
  loader::finalize(tt);

  auto const a = tt.find({"A", source_idx_t{0}});
  ASSERT_TRUE(a);
  auto const b = tt.find({"B", source_idx_t{0}});
  ASSERT_TRUE(b);
  auto const c = tt.find({"C", source_idx_t{0}});
  ASSERT_TRUE(c);

  EXPECT_EQ(tt.locations_.footpaths_out_[kDefaultProfile][*a].size(), 2);
  EXPECT_NE(utl::find(tt.locations_.footpaths_out_[kDefaultProfile][*a],
                      footpath{*b, duration_t{3}}),
            end(tt.locations_.footpaths_out_[kDefaultProfile][*a]));
  EXPECT_NE(utl::find(tt.locations_.footpaths_out_[kDefaultProfile][*a],
                      footpath{*c, duration_t{6}}),
            end(tt.locations_.footpaths_out_[kDefaultProfile][*a]));

  EXPECT_EQ(tt.locations_.footpaths_out_[kDefaultProfile][*b].size(), 2);
  EXPECT_NE(utl::find(tt.locations_.footpaths_out_[kDefaultProfile][*b],
                      footpath{*a, duration_t{3}}),
            end(tt.locations_.footpaths_out_[kDefaultProfile][*b]));
  EXPECT_NE(utl::find(tt.locations_.footpaths_out_[kDefaultProfile][*b],
                      footpath{*c, duration_t{3}}),
            end(tt.locations_.footpaths_out_[kDefaultProfile][*b]));

  EXPECT_EQ(tt.locations_.footpaths_out_[kDefaultProfile][*c].size(), 2);
  EXPECT_NE(utl::find(tt.locations_.footpaths_out_[kDefaultProfile][*c],
                      footpath{*b, duration_t{3}}),
            end(tt.locations_.footpaths_out_[kDefaultProfile][*c]));
  EXPECT_NE(utl::find(tt.locations_.footpaths_out_[kDefaultProfile][*c],
                      footpath{*a, duration_t{6}}),
            end(tt.locations_.footpaths_out_[kDefaultProfile][*c]));
}