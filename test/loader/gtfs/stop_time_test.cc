#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/common/sort_by.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

namespace nigiri::loader::gtfs {

TEST(gtfs, read_stop_times_example_data) {
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
  auto trip_data =
      read_trips(tt, routes, services, files.get_file(kTripsFile).data());
  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  read_stop_times(tt, trip_data, stops, files.get_file(kStopTimesFile).data());

  for (auto& t : trip_data.data_) {
    if (t.requires_sorting_) {
      t.stop_headsigns_.resize(t.seq_numbers_.size());
      std::tie(t.seq_numbers_, t.stop_seq_, t.event_times_, t.stop_headsigns_) =
          sort_by(t.seq_numbers_, t.stop_seq_, t.event_times_,
                  t.stop_headsigns_);
    }
  }

  for (auto& t : trip_data.data_) {
    t.interpolate();
  }

  auto awe1_it = trip_data.trips_.find("AWE1");
  ASSERT_NE(end(trip_data.trips_), awe1_it);

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
  EXPECT_FALSE(stp.out_allowed());
  EXPECT_TRUE(stp.in_allowed());

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

  read_frequencies(trip_data, files.get_file(kFrequenciesFile).data());
}

}  // namespace nigiri::loader::gtfs
