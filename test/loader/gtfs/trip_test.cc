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

  auto const agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(agencies, files.get_file(kRoutesFile).data());
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services = merge_traffic_days(calendar, dates);
  auto const [trips, blocks] =
      read_trips(routes, services, files.get_file(kTripsFile).data());

  EXPECT_EQ(2U, trips.size());
  EXPECT_NE(end(trips), trips.find("AWE1"));
  EXPECT_EQ("A", trips.at("AWE1")->route_->id_);
  EXPECT_EQ("Downtown", trips.at("AWE1")->headsign_);
}

TEST(gtfs, read_trips_berlin_data) {
  auto const files = berlin_files();

  timetable tt;
  tz_map timezones;

  auto const agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(agencies, files.get_file(kRoutesFile).data());
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services = merge_traffic_days(calendar, dates);
  auto const [trips, blocks] =
      read_trips(routes, services, files.get_file(kTripsFile).data());

  EXPECT_EQ(3, trips.size());

  EXPECT_NE(end(trips), trips.find("1"));
  EXPECT_EQ("1", trips.at("1")->route_->id_);
  EXPECT_EQ("Flughafen Schönefeld Terminal (Airport)",
            trips.at("1")->headsign_);

  EXPECT_NE(end(trips), trips.find("2"));
  EXPECT_EQ("1", trips.at("2")->route_->id_);
  EXPECT_EQ("S Potsdam Hauptbahnhof", trips.at("2")->headsign_);

  EXPECT_NE(end(trips), trips.find("3"));
  EXPECT_EQ("2", trips.at("3")->route_->id_);
  EXPECT_EQ("Golzow (PM), Schule", trips.at("3")->headsign_);
}

}  // namespace nigiri::loader::gtfs
