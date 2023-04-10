#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

namespace nigiri::loader::gtfs {

TEST(loader_gtfs_route, read_stop_times_example_data) {
  auto const files = example_files();

  timetable tt;
  tz_map timezones;

  auto const agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(agencies, files.get_file(kRoutesFile).data());
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services = merge_traffic_days(calendar, dates);
  auto [trips, blocks] =
      read_trips(routes, services, files.get_file(kTripsFile).data());
  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data());

  read_stop_times(trips, stops, files.get_file(kStopTimesFile).data());

  for (auto& [_, trip] : trips) {
    trip->interpolate();
  }

  auto awe1_it = trips.find("AWE1");
  ASSERT_NE(end(trips), awe1_it);

  auto& awe1_stops = awe1_it->second->stop_times_;
  auto& stop = awe1_stops[1];
  EXPECT_EQ("S1", tt.locations_.ids_[stop.stop_].view());
  EXPECT_EQ(6_minutes, stop.arr_.time_);
  EXPECT_EQ(6_minutes, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[2];
  EXPECT_EQ("S2", tt.locations_.ids_[stop.stop_].view());
  EXPECT_EQ(6_minutes, stop.arr_.time_);
  EXPECT_EQ(6_minutes, stop.dep_.time_);
  EXPECT_FALSE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[3];
  EXPECT_EQ("S3", tt.locations_.ids_[stop.stop_].view());
  EXPECT_EQ(6_minutes, stop.arr_.time_);
  EXPECT_EQ(6_minutes, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[4];
  EXPECT_EQ("S5", tt.locations_.ids_[stop.stop_].view());
  EXPECT_EQ(6_minutes, stop.arr_.time_);
  EXPECT_EQ(6_minutes, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  stop = awe1_stops[5];
  EXPECT_EQ("S6", tt.locations_.ids_[stop.stop_].view());
  EXPECT_EQ(6_minutes, stop.arr_.time_);
  EXPECT_EQ(6_minutes, stop.dep_.time_);
  EXPECT_TRUE(stop.arr_.in_out_allowed_);
  EXPECT_TRUE(stop.dep_.in_out_allowed_);

  EXPECT_THROW(awe1_it->second->expand_frequencies(
                   [](trip const&, frequency::schedule_relationship) {}),
               std::runtime_error);

  read_frequencies(trips, files.get_file(kFrequenciesFile).data());
  auto i = 0U;
  awe1_it->second->expand_frequencies(
      [&](trip const&, frequency::schedule_relationship s) {
        EXPECT_EQ(frequency::schedule_relationship::kUnscheduled, s);
        ++i;
      });
  EXPECT_EQ(357U, i);
}

}  // namespace nigiri::loader::gtfs
