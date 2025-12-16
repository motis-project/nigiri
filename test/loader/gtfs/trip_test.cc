#include "gtest/gtest.h"

#include <algorithm>

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/common/span_cmp.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;

namespace nigiri::loader::gtfs {

TEST(gtfs, read_trips_example_data) {
  auto const files = example_files();

  timetable tt;
  tt.date_range_ = interval{date::sys_days{July / 1 / 2006},
                            date::sys_days{August / 1 / 2006}};
  tz_map timezones;
  auto i18n = translator{.tt_ = tt};

  auto const config = loader_config{};
  auto agencies =
      read_agencies(source_idx_t{}, tt, i18n, timezones,
                    files.get_file(kAgencyFile).data(), "Europe/Berlin");
  auto const routes =
      read_routes({}, tt, i18n, timezones, agencies,
                  files.get_file(kRoutesFile).data(), "Europe/Berlin");
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services =
      merge_traffic_days(tt.internal_interval_days(), calendar, dates);
  auto shapes_store =
      shapes_storage{"example_shapes", cista::mmap::protection::WRITE};
  auto const shapes =
      parse_shapes(files.get_file(kShapesFile).data(), shapes_store);
  auto const trip_data =
      read_trips(source_idx_t{}, source_file_idx_t{}, tt, i18n, routes,
                 services, shapes, files.get_file(kTripsFile).data(),
                 config.bikes_allowed_default_, config.cars_allowed_default_);

  EXPECT_EQ(2U, trip_data.data_.size());
  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("AWE1"));
  auto const& trip = trip_data.data_.at(trip_data.trips_.at("AWE1"));
  EXPECT_EQ("A", tt.route_ids_[source_idx_t{}].ids_.get(trip.route_));
  EXPECT_EQ("Downtown", tt.get_default_translation(trip.headsign_));
  EXPECT_EQ(shape_idx_t::invalid(), trip.shape_idx_);
}

TEST(gtfs, read_trips_berlin_data) {
  auto const files = berlin_files();

  timetable tt;
  tt.date_range_ = interval{date::sys_days{July / 1 / 2006},
                            date::sys_days{August / 1 / 2006}};
  tz_map timezones;
  auto i18n = translator{.tt_ = tt};

  auto const config = loader_config{};
  auto agencies =
      read_agencies(source_idx_t{0}, tt, i18n, timezones,
                    files.get_file(kAgencyFile).data(), "Europe/Berlin");
  auto const routes =
      read_routes({}, tt, i18n, timezones, agencies,
                  files.get_file(kRoutesFile).data(), "Europe/Berlin");
  auto const dates =
      read_calendar_date(files.get_file(kCalendarDatesFile).data());
  auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
  auto const services =
      merge_traffic_days(tt.internal_interval_days(), calendar, dates);
  auto shapes_store =
      shapes_storage{"berlin_shapes", cista::mmap::protection::WRITE};
  auto const shapes =
      parse_shapes(files.get_file(kShapesFile).data(), shapes_store);
  auto const trip_data =
      read_trips(source_idx_t{}, source_file_idx_t{}, tt, i18n, routes,
                 services, shapes, files.get_file(kTripsFile).data(),
                 config.bikes_allowed_default_, config.cars_allowed_default_);

  EXPECT_EQ(3U, trip_data.data_.size());

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("1"));
  auto const& trip1 = trip_data.data_[trip_data.trips_.at("1")];
  EXPECT_EQ("1", tt.route_ids_[source_idx_t{}].ids_.get(trip1.route_));
  EXPECT_EQ("Flughafen Schönefeld Terminal (Airport)",
            tt.get_default_translation(trip1.headsign_));
  EXPECT_NE(shape_idx_t::invalid(), trip1.shape_idx_);

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("2"));
  auto const& trip2 = trip_data.data_[trip_data.trips_.at("2")];
  EXPECT_EQ("1", tt.route_ids_[source_idx_t{}].ids_.get(trip2.route_));
  EXPECT_EQ("S Potsdam Hauptbahnhof",
            tt.get_default_translation(trip2.headsign_));
  EXPECT_NE(shape_idx_t::invalid(), trip2.shape_idx_);

  EXPECT_NE(end(trip_data.trips_), trip_data.trips_.find("3"));
  auto const& trip3 = trip_data.data_[trip_data.trips_.at("3")];
  EXPECT_EQ("2", tt.route_ids_[source_idx_t{}].ids_.get(trip3.route_));
  EXPECT_EQ("Golzow (PM), Schule", tt.get_default_translation(trip3.headsign_));
  EXPECT_NE(shape_idx_t::invalid(), trip3.shape_idx_);
}

}  // namespace nigiri::loader::gtfs
