#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/tb/preprocess.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

nigiri::timetable load(auto const& files) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2021_y / March / 1},
                    date::sys_days{2021_y / March / 8}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, files(), tt);
  finalize(tt);
  return tt;
}

mem_dir no_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, no_transfer) {
  auto const tt = load(no_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  for (auto const transfers : tbd.segment_transfers_) {
    EXPECT_TRUE(transfers.empty());
  }
}

mem_dir same_day_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,06:00:00,06:00:00,S1,1,0,0
R1_MON,12:00:00,12:00:00,S1,0,0,0
R1_MON,13:00:00,13:00:00,S2,1,0,0
)");
}

TEST(tb_preprocess, same_day_transfer) {
  auto const tt = load(same_day_transfer_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto const s = tbd.transport_first_segment_[transport_idx_t{0U}];
  ASSERT_TRUE(tbd.segment_transfers_[s].size() == 1U);
  auto const& t = tbd.segment_transfers_[s][0];
  EXPECT_EQ(tbd.transport_first_segment_[transport_idx_t{1U}], t.to_segment_);
  EXPECT_EQ(transport_idx_t{1U}, t.to_transport_);
  EXPECT_EQ(bitfield{"100000"}, tbd.bitfields_[t.traffic_days_]);
  EXPECT_EQ(stop_idx_t{0U}, t.to_stop_idx_);
}

mem_dir next_day_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,TUE,R1_TUE,R1_TUE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_TUE,06:00:00,06:00:00,S1,0,0,0
R1_TUE,08:00:00,08:00:00,S2,1,0,0
)");
}

mem_dir long_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,76:00:00,76:00:00,S1,1,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}

mem_dir weekday_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,WD,R0_WD,R0_WD,1
R1,WD,R1_WD,R1_WD,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_WD,00:00:00,00:00:00,S0,0,0,0
R0_WD,02:00:00,02:00:00,S1,1,0,0
R1_WD,01:00:00,01:00:00,S1,0,0,0
R1_WD,03:00:00,03:00:00,S2,1,0,0
)");
}

mem_dir daily_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,DLY,R0_DLY,R0_DLY,1
R1,DLY,R1_DLY,R1_DLY,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_DLY,00:00:00,00:00:00,S0,0,0,0
R0_DLY,02:00:00,02:00:00,S1,1,0,0
R1_DLY,03:00:00,03:00:00,S1,0,0,0
R1_DLY,04:00:00,04:00:00,S2,1,0,0
)");
}

mem_dir earlier_stop_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON0,R0_MON0,0
R0,MON,R0_MON1,R0_MON1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON0,00:00:00,00:00:00,S0,0,0,0
R0_MON0,01:00:00,01:00:00,S1,1,0,0
R0_MON0,02:00:00,02:00:00,S2,2,0,0
R0_MON0,03:00:00,03:00:00,S3,3,0,0
R0_MON0,04:00:00,04:00:00,S1,4,0,0
R0_MON0,05:00:00,05:00:00,S4,5,0,0
R0_MON1,04:00:00,04:00:00,S0,0,0,0
R0_MON1,05:00:00,05:00:00,S1,1,0,0
R0_MON1,06:00:00,06:00:00,S2,2,0,0
R0_MON1,07:00:00,07:00:00,S3,3,0,0
R0_MON1,08:00:00,08:00:00,S1,4,0,0
R0_MON1,09:00:00,09:00:00,S4,5,0,0
)");
}

mem_dir earlier_transport_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON0,R0_MON0,0
R0,MON,R0_MON1,R0_MON1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON0,00:00:00,00:00:00,S0,0,0,0
R0_MON0,01:00:00,01:00:00,S1,1,0,0
R0_MON0,02:00:00,02:00:00,S2,2,0,0
R0_MON0,03:00:00,03:00:00,S3,3,0,0
R0_MON0,04:00:00,04:00:00,S1,4,0,0
R0_MON0,05:00:00,05:00:00,S4,5,0,0
R0_MON1,02:00:00,02:00:00,S0,0,0,0
R0_MON1,03:00:00,03:00:00,S1,1,0,0
R0_MON1,04:00:00,04:00:00,S2,2,0,0
R0_MON1,05:00:00,05:00:00,S3,3,0,0
R0_MON1,05:00:00,05:00:00,S1,4,0,0
R0_MON1,06:00:00,06:00:00,S4,5,0,0
)");
}

mem_dir uturn_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1 -> S2",2
R1,DTA,R1,R1,"S2 -> S1 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,03:00:00,03:00:00,S2,0,0,0
R1_MON,04:00:00,04:00:00,S1,1,0,0
R1_MON,05:00:00,05:00:00,S3,2,0,0
)");
}

mem_dir unnecessary0_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1 -> S2",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,01:00:00,01:10:00,S1,0,0,0
R1_MON,04:00:00,04:00:00,S2,1,0,0
)");
}

mem_dir unnecessary1_transfer_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,
S5,S5,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2 -> S3 -> S4",2
R1,DTA,R1,R1,"S1 -> S2 -> S3 -> S5",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,0
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S2,1,0,0
R0_MON,02:00:00,02:00:00,S3,2,0,0
R0_MON,03:00:00,03:00:00,S4,3,0,0
R1_MON,00:10:00,00:10:00,S1,0,0,0
R1_MON,01:10:00,01:10:00,S2,1,0,0
R1_MON,02:10:00,02:10:00,S3,2,0,0
R1_MON,03:10:00,03:10:00,S5,3,0,0
)");
}

mem_dir enqueue_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,
S5,S5,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,R0,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,DLY,R0_0,R0_0,0
R0,DLY,R0_1,R0_1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_0,00:00:00,00:00:00,S0,0,0,0
R0_0,01:00:00,01:00:00,S1,1,0,0
R0_0,02:00:00,02:00:00,S2,2,0,0
R0_0,03:00:00,03:00:00,S3,3,0,0
R0_0,04:00:00,04:00:00,S4,4,0,0
R0_0,05:00:00,05:00:00,S5,5,0,0
R0_1,01:00:00,01:00:00,S0,0,0,0
R0_1,02:00:00,02:00:00,S1,1,0,0
R0_1,03:00:00,03:00:00,S2,2,0,0
R0_1,04:00:00,04:00:00,S3,3,0,0
R0_1,05:00:00,05:00:00,S4,4,0,0
R0_1,06:00:00,06:00:00,S5,5,0,0
)");
}

mem_dir footpath_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,49.931555,8.650017,,,
S1,S1,footpath_start,49.87296,8.65152,,,
S2,S2,footpath_end,49.87269, 8.65078,,,
S3,S3,,49.816721,8.644180,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S2 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,06:00:00,06:00:00,S1,1,0,0
R1_MON,12:00:00,12:00:00,S2,0,0,0
R1_MON,13:00:00,13:00:00,S3,1,0,0
)");
}

mem_dir early_train_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
DLY,1,1,1,1,1,1,1,20210301,20210307
WE,0,0,0,0,0,1,1,20210301,20210307
WD,1,1,1,1,1,0,0,20210301,20210307
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307
WED,0,0,1,0,0,0,0,20210301,20210307
THU,0,0,0,1,0,0,0,20210301,20210307
FRI,0,0,0,0,1,0,0,20210301,20210307
SAT,0,0,0,0,0,1,0,20210301,20210307
SUN,0,0,0,0,0,0,1,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2 -> S3",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,THU,R1_THU,R1_THU,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,76:00:00,80:00:00,S2,1,0,0
R0_MON,81:00:00,81:00:00,S3,2,0,0
R1_THU,06:00:00,06:00:00,S1,0,0,0
R1_THU,07:00:00,07:00:00,S2,1,0,0
)");
}