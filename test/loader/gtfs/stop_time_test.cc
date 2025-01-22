#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/loader_interface.h"

#include "nigiri/common/sort_by.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace date;

namespace nigiri::loader::gtfs {

TEST(gtfs, quoted_interpolate) {
  constexpr auto const kStopTimes =
      R"("trip_id","arrival_time","departure_time","stop_id","stop_sequence","stop_headsign","pickup_type","drop_off_type","continuous_pickup","continuous_drop_off","shape_dist_traveled","timepoint"
"101255-L001I01S1LAB","06:20:00","06:20:00","101255-6","0","","0","0","","","","1"
"101255-L001I01S1LAB","","","101255-48","1","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-219","2","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-1318","3","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-19","4","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-20","5","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-569","6","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-8","7","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-9","8","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-1986","9","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-10","10","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-11","11","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-12","12","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-925","13","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-14","14","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-15","15","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-16","16","","0","0","","","",""
"101255-L001I01S1LAB","","","101255-17","17","","0","0","","","",""
"101255-L001I01S1LAB","06:49:00","07:00:00","101255-23","18","","0","0","","","","1"
)";

  auto trips = trip_data{};
  trips.trips_.emplace("101255-L001I01S1LAB", gtfs_trip_idx_t{0U});
  auto& t0 =
      trips.data_.emplace_back(nullptr, nullptr, nullptr, "101255-L001I01S1LAB",
                               "", "", shape_idx_t::invalid(), false);
  read_stop_times(tt, trip_data, stops, files.get_file(kStopTimesFile).data(),
                  true);

  for (auto const& s : t0.event_times_) {
    std::cout << s.arr_ << ", " << s.dep_ << "\n";
  }
}

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

}  // namespace nigiri::loader::gtfs
