#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

TEST(gtfs, read_stations_example_data) {
  auto tt = timetable{};
  auto timezones = tz_map{};
  auto i18n = translator{.tt_ = tt};

  register_special_stations(tt);

  auto const files = example_files();
  auto const [stops, _transfers, _accessibility] = read_stops(
      source_idx_t{0}, tt, i18n, timezones, files.get_file(kStopFile).data(),
      files.get_file(kTransfersFile).data(), 0U);

  EXPECT_EQ(8, stops.size());

  auto const s1_it = stops.find("S1");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("Mission St. & Silver Ave.", tt.get_default_name(s1_it->second));
  EXPECT_EQ("The stop is located at the southwest corner of the intersection.",
            tt.get_default_translation(
                tt.locations_.descriptions_.at(s1_it->second)));
  EXPECT_FLOAT_EQ(37.728631, tt.locations_.coordinates_.at(s1_it->second).lat_);
  EXPECT_FLOAT_EQ(-122.431282,
                  tt.locations_.coordinates_.at(s1_it->second).lng_);

  auto const s6_it = stops.find("S6");
  ASSERT_NE(s6_it, end(stops));
  EXPECT_EQ("Mission St. & 15th St.",
            tt.get_default_translation(tt.locations_.names_.at(s6_it->second)));
  EXPECT_EQ("The stop is located 10 feet north of Mission St.",
            tt.get_default_translation(
                tt.locations_.descriptions_.at(s6_it->second)));
  EXPECT_FLOAT_EQ(37.766629, tt.locations_.coordinates_.at(s6_it->second).lat_);
  EXPECT_FLOAT_EQ(-122.419782,
                  tt.locations_.coordinates_.at(s6_it->second).lng_);

  auto const s8_it = stops.find("S8");
  ASSERT_NE(s8_it, end(stops));
  EXPECT_EQ("24th St. Mission Station",
            tt.get_default_translation(tt.locations_.names_.at(s8_it->second)));
  EXPECT_EQ("", tt.get_default_translation(
                    tt.locations_.descriptions_.at(s8_it->second)));
  EXPECT_FLOAT_EQ(37.752240, tt.locations_.coordinates_.at(s8_it->second).lat_);
  EXPECT_FLOAT_EQ(-122.418450,
                  tt.locations_.coordinates_.at(s8_it->second).lng_);

  auto const s7_it = stops.find("S7");
  ASSERT_NE(s7_it, end(stops));
  EXPECT_EQ(15_minutes, tt.locations_.transfer_time_.at(s7_it->second));
}

TEST(gtfs, read_stations_berlin_data) {
  auto tt = timetable{};
  auto timezones = tz_map{};
  auto i18n = translator{.tt_ = tt};

  auto const files = berlin_files();
  auto const [stops, _transfers, _accessibility] = read_stops(
      source_idx_t{0}, tt, i18n, timezones, files.get_file(kStopFile).data(),
      files.get_file(kTransfersFile).data(), 0U);

  EXPECT_EQ(3, stops.size());

  auto s0_it = stops.find("5100071");
  ASSERT_NE(s0_it, end(stops));
  EXPECT_EQ("Zbaszynek", tt.get_default_name(s0_it->second));
  EXPECT_FLOAT_EQ(52.2425040,
                  tt.locations_.coordinates_.at(s0_it->second).lat_);
  EXPECT_FLOAT_EQ(15.8180870,
                  tt.locations_.coordinates_.at(s0_it->second).lng_);

  auto s1_it = stops.find("9230005");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("S Potsdam Hauptbahnhof Nord", tt.get_default_name(s1_it->second));
  EXPECT_FLOAT_EQ(52.3927320,
                  tt.locations_.coordinates_.at(s1_it->second).lat_);
  EXPECT_FLOAT_EQ(13.0668480,
                  tt.locations_.coordinates_.at(s1_it->second).lng_);

  auto s2_it = stops.find("9230006");
  ASSERT_NE(s2_it, end(stops));
  EXPECT_EQ("Potsdam, Charlottenhof Bhf", tt.get_default_name(s2_it->second));
  EXPECT_FLOAT_EQ(52.3930040,
                  tt.locations_.coordinates_.at(s2_it->second).lat_);
  EXPECT_FLOAT_EQ(13.0362980,
                  tt.locations_.coordinates_.at(s2_it->second).lng_);
}

TEST(gtfs, read_stations_stop_code_platform_fallback) {
  auto tt = timetable{};
  auto timezones = tz_map{};
  auto i18n = translator{.tt_ = tt};

  register_special_stations(tt);

  // P:  station (no platform/track).
  // A:  platform_code and stop_code set -> platform_code wins.
  // B:  only stop_code set              -> falls back to stop_code.
  // C:  neither set                     -> empty.
  constexpr auto const stops_content = std::string_view{
      R"(stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station,platform_code
P,,Parent,,52.0,13.0,1,,
A,A_CODE,Platform A,,52.0,13.0,0,P,A_PLATFORM
B,B_CODE,Platform B,,52.0,13.0,0,P,
C,,Platform C,,52.0,13.0,0,P,
)"};

  auto const [stops, _transfers, _accessibility] =
      read_stops(source_idx_t{0}, tt, i18n, timezones, stops_content,
                 std::string_view{}, 0U);

  auto const track = [&](std::string_view const id) {
    return tt.get_default_translation(
        tt.locations_.platform_codes_.at(stops.at(std::string{id})));
  };

  EXPECT_EQ("A_PLATFORM", track("A"));
  EXPECT_EQ("B_CODE", track("B"));
  EXPECT_EQ("", track("C"));
}
