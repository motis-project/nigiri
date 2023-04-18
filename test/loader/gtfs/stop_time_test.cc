#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/common/sort_by.h"
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

  for (auto& [_, t] : trips) {
    if (t->requires_sorting_) {
      t->stop_headsigns_.resize(t->seq_numbers_.size());
      std::tie(t->seq_numbers_, t->stop_seq_, t->event_times_,
               t->stop_headsigns_) =
          sort_by(t->seq_numbers_, t->stop_seq_, t->event_times_,
                  t->stop_headsigns_);
    }
  }

  for (auto& [_, trip] : trips) {
    trip->interpolate();
  }

  auto awe1_it = trips.find("AWE1");
  ASSERT_NE(end(trips), awe1_it);

  auto& awe1_stop_times = awe1_it->second->event_times_[0];
  auto stop = timetable::stop{awe1_it->second->stop_seq_[0]};
  EXPECT_EQ("S1", tt.locations_.ids_[stop.location_idx()].view());
  EXPECT_EQ(6_hours + 10_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 10_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stop.out_allowed());
  EXPECT_TRUE(stop.in_allowed());

  awe1_stop_times = awe1_it->second->event_times_[1];
  stop = timetable::stop{awe1_it->second->stop_seq_[1]};
  EXPECT_EQ("S2", tt.locations_.ids_[stop.location_idx()].view());
  EXPECT_EQ(6_hours + 15_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 15_minutes, awe1_stop_times.dep_);
  EXPECT_FALSE(stop.out_allowed());
  EXPECT_TRUE(stop.in_allowed());

  awe1_stop_times = awe1_it->second->event_times_[2];
  stop = timetable::stop{awe1_it->second->stop_seq_[2]};
  EXPECT_EQ("S3", tt.locations_.ids_[stop.location_idx()].view());
  EXPECT_EQ(6_hours + 20_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 30_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stop.out_allowed());
  EXPECT_TRUE(stop.in_allowed());

  awe1_stop_times = awe1_it->second->event_times_[3];
  stop = timetable::stop{awe1_it->second->stop_seq_[3]};
  EXPECT_EQ("S5", tt.locations_.ids_[stop.location_idx()].view());
  EXPECT_EQ(6_hours + 38_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 38_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stop.out_allowed());
  EXPECT_TRUE(stop.in_allowed());

  awe1_stop_times = awe1_it->second->event_times_[4];
  stop = timetable::stop{awe1_it->second->stop_seq_[4]};
  EXPECT_EQ("S6", tt.locations_.ids_[stop.location_idx()].view());
  EXPECT_EQ(6_hours + 45_minutes, awe1_stop_times.arr_);
  EXPECT_EQ(6_hours + 45_minutes, awe1_stop_times.dep_);
  EXPECT_TRUE(stop.out_allowed());
  EXPECT_TRUE(stop.in_allowed());

  read_frequencies(trips, files.get_file(kFrequenciesFile).data());
}

}  // namespace nigiri::loader::gtfs