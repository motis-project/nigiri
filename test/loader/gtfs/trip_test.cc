#include "gtest/gtest.h"

#include <algorithm>

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri::loader;

namespace nigiri::loader::gtfs {

TEST(gtfs, read_trips_example_data) {
  auto const files = example_files();

  timetable tt;
  tz_map timezones;

  auto agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(tt, timezones, agencies,
                                  files.get_file(kRoutesFile).data(), "CET");
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services = merge_traffic_days(calendar, dates);
  auto const trip_data =
      read_trips(tt, routes, services, files.get_file(kTripsFile).data());

  EXPECT_EQ(2U, trip_data.data_.size());
  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("AWE1"));
  EXPECT_EQ("A", trip_data.data_.at(trip_data.trips_.at("AWE1")).route_->id_);

  EXPECT_EQ("Downtown",
            tt.trip_direction(
                trip_data.data_.at(trip_data.trips_.at("AWE1")).headsign_));
}

TEST(gtfs, read_trips_berlin_data) {
  auto const files = berlin_files();

  timetable tt;
  tz_map timezones;

  auto agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(tt, timezones, agencies,
                                  files.get_file(kRoutesFile).data(), "CET");
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services = merge_traffic_days(calendar, dates);
  auto const trip_data =
      read_trips(tt, routes, services, files.get_file(kTripsFile).data());

  EXPECT_EQ(3, trip_data.data_.size());

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("1"));
  EXPECT_EQ("1", trip_data.data_[trip_data.trips_.at("1")].route_->id_);
  EXPECT_EQ(
      "Flughafen Schönefeld Terminal (Airport)",
      tt.trip_direction(trip_data.data_[trip_data.trips_.at("1")].headsign_));

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("2"));
  EXPECT_EQ("1", trip_data.data_[trip_data.trips_.at("2")].route_->id_);
  EXPECT_EQ(
      "S Potsdam Hauptbahnhof",
      tt.trip_direction(trip_data.data_[trip_data.trips_.at("2")].headsign_));

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("3"));
  EXPECT_EQ("2", trip_data.data_[trip_data.trips_.at("3")].route_->id_);
  EXPECT_EQ(
      "Golzow (PM), Schule",
      tt.trip_direction(trip_data.data_[trip_data.trips_.at("3")].headsign_));
}

}  // namespace nigiri::loader::gtfs
