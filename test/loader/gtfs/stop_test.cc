#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

TEST(gtfs, read_stations_example_data) {
  timetable tt;
  tz_map timezones;

  auto const files = example_files();
  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  EXPECT_EQ(8, stops.size());

  auto const s1_it = stops.find("S1");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("Mission St. & Silver Ave.",
            tt.locations_.names_.at(s1_it->second).view());
  EXPECT_FLOAT_EQ(37.728631, tt.locations_.coordinates_.at(s1_it->second).lat_);
  EXPECT_FLOAT_EQ(-122.431282,
                  tt.locations_.coordinates_.at(s1_it->second).lng_);

  auto const s6_it = stops.find("S6");
  ASSERT_NE(s6_it, end(stops));
  EXPECT_EQ("Mission St. & 15th St.",
            tt.locations_.names_.at(s6_it->second).view());
  EXPECT_FLOAT_EQ(37.766629, tt.locations_.coordinates_.at(s6_it->second).lat_);
  EXPECT_FLOAT_EQ(-122.419782,
                  tt.locations_.coordinates_.at(s6_it->second).lng_);

  auto const s8_it = stops.find("S8");
  ASSERT_NE(s8_it, end(stops));
  EXPECT_EQ("24th St. Mission Station",
            tt.locations_.names_.at(s8_it->second).view());
  EXPECT_FLOAT_EQ(37.752240, tt.locations_.coordinates_.at(s8_it->second).lat_);
  EXPECT_FLOAT_EQ(-122.418450,
                  tt.locations_.coordinates_.at(s8_it->second).lng_);
}

TEST(gtfs, read_stations_berlin_data) {
  timetable tt;
  tz_map timezones;

  auto const files = berlin_files();
  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  EXPECT_EQ(3, stops.size());

  auto s0_it = stops.find("5100071");
  ASSERT_NE(s0_it, end(stops));
  EXPECT_EQ("Zbaszynek", tt.locations_.names_.at(s0_it->second).view());
  EXPECT_FLOAT_EQ(52.2425040,
                  tt.locations_.coordinates_.at(s0_it->second).lat_);
  EXPECT_FLOAT_EQ(15.8180870,
                  tt.locations_.coordinates_.at(s0_it->second).lng_);

  auto s1_it = stops.find("9230005");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("S Potsdam Hauptbahnhof Nord",
            tt.locations_.names_.at(s1_it->second).view());
  EXPECT_FLOAT_EQ(52.3927320,
                  tt.locations_.coordinates_.at(s1_it->second).lat_);
  EXPECT_FLOAT_EQ(13.0668480,
                  tt.locations_.coordinates_.at(s1_it->second).lng_);

  auto s2_it = stops.find("9230006");
  ASSERT_NE(s2_it, end(stops));
  EXPECT_EQ("Potsdam, Charlottenhof Bhf",
            tt.locations_.names_.at(s2_it->second).view());
  EXPECT_FLOAT_EQ(52.3930040,
                  tt.locations_.coordinates_.at(s2_it->second).lat_);
  EXPECT_FLOAT_EQ(13.0362980,
                  tt.locations_.coordinates_.at(s2_it->second).lng_);
}
