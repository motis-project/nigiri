#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/stop.h"

using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

constexpr auto const example_file_content = std::string_view{
    R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S1,Mission St. & Silver Ave.,The stop is located at the southwest corner of the intersection.,37.728631,-122.431282,,,
S2,Mission St. & Cortland Ave.,The stop is located 20 feet south of Mission St.,37.74103,-122.422482,,,
S3,Mission St. & 24th St.,The stop is located at the southwest corner of the intersection.,37.75223,-122.418581,,,
S4,Mission St. & 21st St.,The stop is located at the northwest corner of the intersection.,37.75713,-122.418982,,,
S5,Mission St. & 18th St.,The stop is located 25 feet west of 18th St.,37.761829,-122.419382,,,
S6,Mission St. & 15th St.,The stop is located 10 feet north of Mission St.,37.766629,-122.419782,,,
S7,24th St. Mission Station,,37.752240,-122.418450,,,S8
S8,24th St. Mission Station,,37.752240,-122.418450,http://www.bart.gov/stations/stationguide/stationoverview_24st.asp,1,
)"};

TEST(gtfs, read_stations_example_data) {
  auto stops = read_stops(example_file_content);

  EXPECT_EQ(8, stops.size());

  auto s1_it = stops.find("S1");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("Mission St. & Silver Ave.", s1_it->second->name_);
  EXPECT_FLOAT_EQ(37.728631, s1_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(-122.431282, s1_it->second->coord_.lng_);

  auto s6_it = stops.find("S6");
  ASSERT_NE(s6_it, end(stops));
  EXPECT_EQ("Mission St. & 15th St.", s6_it->second->name_);
  EXPECT_FLOAT_EQ(37.766629, s6_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(-122.419782, s6_it->second->coord_.lng_);

  auto s8_it = stops.find("S8");
  ASSERT_NE(s8_it, end(stops));
  EXPECT_EQ("24th St. Mission Station", s8_it->second->name_);
  EXPECT_FLOAT_EQ(37.752240, s8_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(-122.418450, s8_it->second->coord_.lng_);
}

constexpr auto const berlin_file_content = std::string_view(
    R"(stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station
5100071,,Zbaszynek,,52.2425040,15.8180870,,,0,
9230005,,S Potsdam Hauptbahnhof Nord,,52.3927320,13.0668480,,,0,
9230006,,"Potsdam, Charlottenhof Bhf",,52.3930040,13.0362980,,,0,
)");

TEST(gtfs, read_stations_berlin_data) {
  auto stops = read_stops(berlin_file_content);

  EXPECT_EQ(3, stops.size());

  auto s0_it = stops.find("5100071");
  ASSERT_NE(s0_it, end(stops));
  EXPECT_EQ("Zbaszynek", s0_it->second->name_);
  EXPECT_FLOAT_EQ(52.2425040, s0_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(15.8180870, s0_it->second->coord_.lng_);

  auto s1_it = stops.find("9230005");
  ASSERT_NE(s1_it, end(stops));
  EXPECT_EQ("S Potsdam Hauptbahnhof Nord", s1_it->second->name_);
  EXPECT_FLOAT_EQ(52.3927320, s1_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(13.0668480, s1_it->second->coord_.lng_);

  auto s2_it = stops.find("9230006");
  ASSERT_NE(s2_it, end(stops));
  EXPECT_EQ("Potsdam, Charlottenhof Bhf", s2_it->second->name_);
  EXPECT_FLOAT_EQ(52.3930040, s2_it->second->coord_.lat_);
  EXPECT_FLOAT_EQ(13.0362980, s2_it->second->coord_.lng_);
}
