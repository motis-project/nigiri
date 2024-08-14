#include "gtest/gtest.h"

#include <algorithm>

#include "utl/raii.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/timetable.h"

#include "./shape_test.h"
#include "./test_data.h"

using namespace date;
using namespace nigiri::loader;

namespace nigiri::loader::gtfs {

TEST(gtfs, read_trips_example_data) {
  auto const files = example_files();
  auto mmap = shape_test_mmap{"trips-example-data"};

  timetable tt;
  tt.date_range_ = interval{date::sys_days{July / 1 / 2006},
                            date::sys_days{August / 1 / 2006}};
  tz_map timezones;

  auto const config = loader_config{};
  auto agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(tt, timezones, agencies,
                                  files.get_file(kRoutesFile).data(), "CET");
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services =
      merge_traffic_days(tt.internal_interval_days(), calendar, dates);
  auto const shapes =
      parse_shapes(files.get_file(kShapesFile).data(), &mmap.get_vecvec());
  auto const trip_data = read_trips(tt, routes, services, shapes,
                                    files.get_file(kTripsFile).data(),
                                    config.bikes_allowed_default_);

  EXPECT_EQ(2U, trip_data.data_.size());
  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("AWE1"));
  auto const& trip = trip_data.data_.at(trip_data.trips_.at("AWE1"));
  EXPECT_EQ("A", trip.route_->id_);
  EXPECT_EQ("Downtown", tt.trip_direction(trip.headsign_));
  EXPECT_EQ(shape_idx_t::invalid(), trip.shape_idx_);
}

TEST(gtfs, read_trips_berlin_data) {
  auto const files = berlin_files();
  auto mmap = shape_test_mmap{"trips-berlin-data"};

  timetable tt;
  tt.date_range_ = interval{date::sys_days{July / 1 / 2006},
                            date::sys_days{August / 1 / 2006}};
  tz_map timezones;

  auto const config = loader_config{};
  auto agencies =
      read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
  auto const routes = read_routes(tt, timezones, agencies,
                                  files.get_file(kRoutesFile).data(), "CET");
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services =
      merge_traffic_days(tt.internal_interval_days(), calendar, dates);
  auto const shapes =
      parse_shapes(files.get_file(kShapesFile).data(), &mmap.get_vecvec());
  auto const trip_data = read_trips(tt, routes, services, shapes,
                                    files.get_file(kTripsFile).data(),
                                    config.bikes_allowed_default_);

  EXPECT_EQ(3, trip_data.data_.size());

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("1"));
  auto const& trip1 = trip_data.data_[trip_data.trips_.at("1")];
  EXPECT_EQ("1", trip1.route_->id_);
  EXPECT_EQ("Flughafen Schönefeld Terminal (Airport)",
            tt.trip_direction(trip1.headsign_));
  EXPECT_NE(shape_idx_t::invalid(), trip1.shape_idx_);

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("2"));
  auto const& trip2 = trip_data.data_[trip_data.trips_.at("2")];
  EXPECT_EQ("1", trip2.route_->id_);
  EXPECT_EQ("S Potsdam Hauptbahnhof", tt.trip_direction(trip2.headsign_));
  EXPECT_NE(shape_idx_t::invalid(), trip2.shape_idx_);

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("3"));
  auto const& trip3 = trip_data.data_[trip_data.trips_.at("3")];
  EXPECT_EQ("2", trip3.route_->id_);
  EXPECT_EQ("Golzow (PM), Schule", tt.trip_direction(trip3.headsign_));
  EXPECT_NE(shape_idx_t::invalid(), trip3.shape_idx_);
}

}  // namespace nigiri::loader::gtfs
