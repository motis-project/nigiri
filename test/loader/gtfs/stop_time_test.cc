#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/loader_interface.h"

#include "nigiri/common/sort_by.h"
#include "nigiri/timetable.h"

#include "nigiri/loader/load.h"
#include "nigiri/resolve.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"

#include "./test_data.h"

using namespace date;
using namespace std::chrono_literals;

namespace nigiri::loader::gtfs {

TEST(gtfs, quoted_interpolate) {
  constexpr auto const kTimetable = R"(
# agency.txt

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
101255-6,Stop 1,52.0,13.0
101255-48,Stop 2,52.01,13.01
101255-23,Stop 3,52.02,13.02

# calendar_dates.txt
service_id,date,exception_type
1,20251207,1

# routes.txt
route_id
1

# trips.txt
trip_id,service_id,route_id
101255-L001I01S1LAB,1,1

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","stop_headsign","pickup_type","drop_off_type","continuous_pickup","continuous_drop_off","shape_dist_traveled","timepoint"
"101255-L001I01S1LAB","06:20:00","06:20:00","101255-6","0","","0","0","","","","1"
"101255-L001I01S1LAB","","","101255-48","1","","0","0","","","",""
"101255-L001I01S1LAB","06:49:00","07:00:00","101255-23","18","","0","0","","","","1"
)";

  auto const date = date::sys_days{2025_y / December / 7};
  auto tt = loader::load({{.tag_ = "test",
                           .path_ = kTimetable,
                           .loader_config_ = {.default_tz_ = "Europe/Berlin"}}},
                         {}, {date, date::sys_days{2025_y / December / 8}});

  auto const fr = resolve(tt, nullptr, "101255-L001I01S1LAB", "20251207");
  ASSERT_EQ(3U, fr.size());
  EXPECT_EQ(date + 5h + 20min, fr[0].time(event_type::kDep));
  EXPECT_EQ(date + 5h + 35min, fr[1].time(event_type::kArr));
  EXPECT_EQ(date + 5h + 35min, fr[1].time(event_type::kDep));
  EXPECT_EQ(date + 5h + 49min, fr[2].time(event_type::kArr));
}

TEST(gtfs, unquoted_interpolate) {
  constexpr auto const kTimetable = R"(
# agency.txt

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
6,Stop 1,52.0,13.0
48,Stop 2,52.01,13.01
23,Stop 3,52.02,13.02

# calendar_dates.txt
service_id,date,exception_type
1,20251207,1

# routes.txt
route_id
1

# trips.txt
trip_id,service_id,route_id
L001I01S1FES,1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
L001I01S1FES,08:00:00,,6,1,,0,0,
L001I01S1FES, , ,48,2,,0,0,0.265
L001I01S1FES,08:31:00,08:37:00,23,19,,0,0,7.473
)";

  auto const date = date::sys_days{2025_y / December / 7};
  auto tt = loader::load({{.tag_ = "test",
                           .path_ = kTimetable,
                           .loader_config_ = {.default_tz_ = "Europe/Berlin"}}},
                         {}, {date, date::sys_days{2025_y / December / 8}});

  auto const fr = resolve(tt, nullptr, "L001I01S1FES", "20251207");
  ASSERT_EQ(3U, fr.size());
  EXPECT_EQ(date + 7h, fr[0].time(event_type::kDep));
  EXPECT_EQ(date + 7h + 16min, fr[1].time(event_type::kArr));
  EXPECT_EQ(date + 7h + 16min, fr[1].time(event_type::kDep));
  EXPECT_EQ(date + 7h + 31min, fr[2].time(event_type::kArr));
}

TEST(gtfs, start_end_interpolate) {
  constexpr auto const kStopTimes =
      R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
L001I01S1FES,,08:00:00,6,1,,0,0,
L001I01S1FES, , ,48,2,,0,0,0.265
L001I01S1FES,08:31:00,,23,19,,0,0,7.473
)";

  auto trips = trip_data{};
  trips.trips_.emplace("L001I01S1FES", gtfs_trip_idx_t{0U});
  trips.data_
      .emplace_back(route_id_idx_t::invalid(), nullptr, nullptr, "L001I01S1FES",
                    kEmptyTranslation, kEmptyTranslation,
                    direction_id_t::invalid(), shape_idx_t::invalid(), false,
                    false)
      .trip_idx_ = {};
  auto tt = timetable{};
  tt.trip_debug_.emplace_back().emplace_back(trip_debug{});
  auto i18n = translator{.tt_ = tt};
  auto stops = stops_map_t{};
  stops.emplace("6", location_idx_t{0});
  stops.emplace("48", location_idx_t{1});
  stops.emplace("23", location_idx_t{2});
  read_stop_times(trips, stops, {}, {}, {}, i18n, kStopTimes, true);

  EXPECT_TRUE(trips.data_[gtfs_trip_idx_t{0}].requires_interpolation_);
  EXPECT_EQ(interpolate_result::kOk,
            interpolate(trips.data_[gtfs_trip_idx_t{0}].event_times_));
  auto const& ev = trips.data_[gtfs_trip_idx_t{0}].event_times_;
  ASSERT_EQ(3U, ev.size());
  EXPECT_EQ(0h, ev[0].arr_);
  EXPECT_EQ(8h, ev[0].dep_);
  EXPECT_EQ(8h + 16min, ev[1].arr_);
  EXPECT_EQ(8h + 16min, ev[1].dep_);
  EXPECT_EQ(8h + 31min, ev[2].arr_);
  EXPECT_EQ(8h + 31min, ev[2].dep_);
}

TEST(gtfs, failed_first_interpolate) {
  constexpr auto const kStopTimes =
      R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
L001I01S1FES,,,6,1,,0,0,
L001I01S1FES, , ,48,2,,0,0,0.265
L001I01S1FES,,08:31:00,23,19,,0,0,7.473
)";

  auto trips = trip_data{};
  trips.trips_.emplace("L001I01S1FES", gtfs_trip_idx_t{0U});
  trips.data_
      .emplace_back(route_id_idx_t::invalid(), nullptr, nullptr, "L001I01S1FES",
                    kEmptyTranslation, kEmptyTranslation,
                    direction_id_t::invalid(), shape_idx_t::invalid(), false,
                    false)
      .trip_idx_ = {};
  auto tt = timetable{};
  tt.trip_debug_.emplace_back().emplace_back(trip_debug{});
  auto i18n = translator{.tt_ = tt};
  auto stops = stops_map_t{};
  stops.emplace("6", location_idx_t{0});
  stops.emplace("48", location_idx_t{1});
  stops.emplace("23", location_idx_t{2});
  read_stop_times(trips, stops, {}, {}, {}, i18n, kStopTimes, true);

  EXPECT_EQ(interpolate_result::kErrorFirstMissing,
            interpolate(trips.data_[gtfs_trip_idx_t{0}].event_times_));
}

TEST(gtfs, failed_last_interpolate) {
  constexpr auto const kStopTimes =
      R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
L001I01S1FES,,,6,1,,0,0,
L001I01S1FES, , ,48,2,,0,0,0.265
L001I01S1FES,,,23,19,,0,0,7.473
)";

  auto trips = trip_data{};
  trips.trips_.emplace("L001I01S1FES", gtfs_trip_idx_t{0U});
  trips.data_
      .emplace_back(route_id_idx_t::invalid(), nullptr, nullptr, "L001I01S1FES",
                    kEmptyTranslation, kEmptyTranslation,
                    direction_id_t::invalid(), shape_idx_t::invalid(), false,
                    false)
      .trip_idx_ = {};
  auto tt = timetable{};
  tt.trip_debug_.emplace_back().emplace_back(trip_debug{});
  auto i18n = translator{.tt_ = tt};
  auto stops = stops_map_t{};
  stops.emplace("6", location_idx_t{0});
  stops.emplace("48", location_idx_t{1});
  stops.emplace("23", location_idx_t{2});
  read_stop_times(trips, stops, {}, {}, {}, i18n, kStopTimes, true);

  EXPECT_EQ(interpolate_result::kErrorLastMissing,
            interpolate(trips.data_[gtfs_trip_idx_t{0}].event_times_));
}

TEST(gtfs, read_stop_times_example_data) {
  auto const files = example_files();

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
  auto trip_data =
      read_trips(source_idx_t{}, source_file_idx_t{}, tt, i18n, routes,
                 services, {}, files.get_file(kTripsFile).data(),
                 config.bikes_allowed_default_, config.cars_allowed_default_);
  auto const [stops, _] = read_stops(source_idx_t{0}, tt, i18n, timezones,
                                     files.get_file(kStopFile).data(),
                                     files.get_file(kTransfersFile).data(), 0U);

  read_stop_times(trip_data, stops, {}, {}, {}, i18n,
                  files.get_file(kStopTimesFile).data(), true);

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
    interpolate(t.event_times_);
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

}  // namespace nigiri::loader::gtfs
