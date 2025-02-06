#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop_time.h"

#include <nigiri/loader/gtfs/area.h>

#include "nigiri/loader/loader_interface.h"

#include "nigiri/common/sort_by.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "./test_data.h"

using namespace date;

namespace nigiri::loader::gtfs {

TEST(gtfs, read_stop_times_example_data) {
  auto const files = example_files();

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
  auto trip_data =
      read_trips(tt, routes, services, {}, files.get_file(kTripsFile).data(),
                 config.bikes_allowed_default_);
  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  read_stop_times(tt, trip_data, stops, files.get_file(kStopTimesFile).data(),
                  true);

  for (auto& t : trip_data.data_) {
    if (t.requires_sorting_) {
      t.stop_headsigns_.resize(t.seq_numbers_.size());
      std::tie(t.seq_numbers_, t.stop_seq_, t.event_times_, t.stop_headsigns_,
               t.distance_traveled_) =
          sort_by(t.seq_numbers_, t.stop_seq_, t.event_times_,
                  t.stop_headsigns_, t.distance_traveled_);
    }
  }

  for (auto& t : trip_data.data_) {
    t.interpolate();
  }

  auto awe1_it = trip_data.trips_.find("AWE1");
  ASSERT_NE(end(trip_data.trips_), awe1_it);

  EXPECT_EQ(shape_idx_t::invalid(),
            trip_data.data_[awe1_it->second].shape_idx_);

  auto& awe1_stop_times = trip_data.data_[awe1_it->second].event_times_[0];
  auto stp = stop{trip_data.data_[awe1_it->second].stop_seq_[0]};
  EXPECT_EQ("S1", tt.locations_.ids_[stp.location_idx()].view());
  EXPECT_EQ(6_hours + 10_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 10_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stp.out_allowed());
  EXPECT_TRUE(stp.in_allowed());

  awe1_stop_times = trip_data.data_[awe1_it->second].event_times_[1];
  stp = stop{trip_data.data_[awe1_it->second].stop_seq_[1]};
  EXPECT_EQ("S2", tt.locations_.ids_[stp.location_idx()].view());
  EXPECT_EQ(6_hours + 15_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 15_minutes, awe1_stop_times.dep_);
  EXPECT_FALSE(stp.in_allowed());
  EXPECT_TRUE(stp.out_allowed());

  awe1_stop_times = trip_data.data_[awe1_it->second].event_times_[2];
  stp = stop{trip_data.data_[awe1_it->second].stop_seq_[2]};
  EXPECT_EQ("S3", tt.locations_.ids_[stp.location_idx()].view());
  EXPECT_EQ(6_hours + 20_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 30_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stp.out_allowed());
  EXPECT_TRUE(stp.in_allowed());

  awe1_stop_times = trip_data.data_[awe1_it->second].event_times_[3];
  stp = stop{trip_data.data_[awe1_it->second].stop_seq_[3]};
  EXPECT_EQ("S5", tt.locations_.ids_[stp.location_idx()].view());
  EXPECT_EQ(6_hours + 38_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 38_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stp.out_allowed());
  EXPECT_TRUE(stp.in_allowed());

  awe1_stop_times = trip_data.data_[awe1_it->second].event_times_[4];
  stp = stop{trip_data.data_[awe1_it->second].stop_seq_[4]};
  EXPECT_EQ("S6", tt.locations_.ids_[stp.location_idx()].view());
  EXPECT_EQ(6_hours + 45_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 45_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stp.out_allowed());
  EXPECT_TRUE(stp.in_allowed());

  // Check distances are stored iff at least 1 entry is != 0.0
  EXPECT_EQ((std::vector{0.0, 3.14, 5.0, 0.0, 0.0}),
            trip_data.data_[awe1_it->second].distance_traveled_);
  // Check distances are not stored if column is 0.0
  auto awd1_it = trip_data.trips_.find("AWD1");
  ASSERT_NE(end(trip_data.trips_), awd1_it);
  EXPECT_TRUE(trip_data.data_[awd1_it->second].distance_traveled_.empty());

  read_frequencies(trip_data, files.get_file(kFrequenciesFile).data());
}

TEST(gtfs, read_stop_times_gtfs_flex_example_data) {
  auto const files = example_files();

  auto const src = source_idx_t{0};

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
  auto trip_data =
      read_trips(tt, routes, services, {}, files.get_file(kTripsFile).data(),
                 config.bikes_allowed_default_);
  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  auto const location_geojsons =
      read_location_geojson(tt, files.get_file(kLocationGeojsonFile).data());

  auto booking_rule_calendar =
      read_calendar(files.get_file(kBookingRuleCalendarFile).data());
  auto booking_rule_calendar_dates =
      read_calendar_date(files.get_file(kBookingRuleCalendarDatesFile).data());
  auto booking_rule_services =
      merge_traffic_days(tt.internal_interval_days(), booking_rule_calendar,
                         booking_rule_calendar_dates);

  auto const booking_rules = read_booking_rules(
      booking_rule_services, tt, files.get_file(kBookingRulesFile).data());

  read_stop_times(tt, src, trip_data, location_geojsons, stops, booking_rules,
                  files.get_file(kStopTimesGTFSFlexFile).data(), false);

  auto const test_stop_time =
      [&](std::string const& geo_id, std::string const& trip_id,
          std::initializer_list<std::string> const expected_trips,
          std::initializer_list<std::string> const expected_geos,
          stop_window&& expected_window, booking_rule_idx_t expected_pickup,
          booking_rule_idx_t expected_dropoff,
          pickup_dropoff_type expected_pickup_type,
          pickup_dropoff_type expected_dropoff_type) {
        auto const gtfs_trip_idx = trip_data.trips_[trip_id];
        auto const trip_idx = trip_data.data_[gtfs_trip_idx].trip_idx_;
        auto const geo_idx = location_geojsons.at(geo_id);

        ASSERT_LT(geo_idx.v_, tt.geometry_idx_to_trip_idxs_.size());
        ASSERT_EQ(tt.geometry_idx_to_trip_idxs_[geo_idx].size(),
                  expected_trips.size());
        for (auto i = 0; i < expected_trips.size(); ++i) {
          auto const id = *(expected_trips.begin() + i);
          auto const gtfs_t_idx = trip_data.trips_[id];
          auto t_idx = trip_data.data_[gtfs_t_idx].trip_idx_;
          EXPECT_EQ(tt.geometry_idx_to_trip_idxs_[geo_idx][i], t_idx);
        }

        for (auto i = 0; i < expected_geos.size(); ++i) {
          auto const id = *(expected_geos.begin() + i);
          auto g_idx = location_geojsons.at(id);
          EXPECT_EQ(tt.trip_idx_to_geometry_idxs_[trip_idx][i], g_idx);
        }

        geometry_trip_idx gt_idx;
        gt_idx = geometry_trip_idx{trip_idx, geo_idx};
        auto idx = tt.geometry_trip_idxs_[gt_idx];

        EXPECT_EQ(tt.window_times_.at(idx)[0].start_, expected_window.start_);
        EXPECT_EQ(tt.window_times_.at(idx)[0].end_, expected_window.end_);
        EXPECT_EQ(tt.pickup_booking_rules_.at(idx)[0], expected_pickup);
        EXPECT_EQ(tt.dropoff_booking_rules_.at(idx)[0], expected_dropoff);
        EXPECT_EQ(tt.pickup_types_.at(idx)[0], expected_pickup_type);
        EXPECT_EQ(tt.dropoff_types_.at(idx)[0], expected_dropoff_type);
      };
  auto const br_idx_3 = booking_rules.at("3");
  auto const br_idx_4 = booking_rules.at("4");
  auto const br_idx_5 = booking_rules.at("5");

  test_stop_time("l_geo_1", "AWE1", {"AWE1", "AWD1"}, {"l_geo_1", "l_geo_2"},
                 stop_window{hhmm_to_min("06:00:00"), hhmm_to_min("19:00:00")},
                 booking_rule_idx_t::invalid(), booking_rule_idx_t::invalid(),
                 kPhoneAgencyType, kCoordinateWithDriverType);
  test_stop_time("l_geo_2", "AWE1", {"AWE1"}, {"l_geo_1", "l_geo_2"},
                 stop_window{hhmm_to_min("08:00:00"), hhmm_to_min("20:00:00")},
                 br_idx_3, br_idx_3, kPhoneAgencyType, kPhoneAgencyType);
  test_stop_time("l_geo_3", "AWD1", {"AWD1"}, {"l_geo_3", "l_geo_1"},
                 stop_window{hhmm_to_min("11:00:00"), hhmm_to_min("17:00:00")},
                 br_idx_3, br_idx_3, kPhoneAgencyType, kPhoneAgencyType);
  test_stop_time("l_geo_1", "AWD1", {"AWE1", "AWD1"}, {"l_geo_3", "l_geo_1"},
                 stop_window{hhmm_to_min("10:00:00"), hhmm_to_min("19:00:00")},
                 br_idx_4, br_idx_5, kPhoneAgencyType, kUnavailableType);
}

}  // namespace nigiri::loader::gtfs
