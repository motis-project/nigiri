#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using nigiri::test::raptor_search;

constexpr auto const calendar = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
T1_DAYS,0,1,1,1,1,0,0,20060701,20060731
T2_DAYS,0,1,0,1,0,1,0,20060701,20060731
T3_DAYS,0,0,1,0,0,0,0,20060701,20060731
T4_DAYS,0,0,0,1,0,0,0,20060701,20060731
T5_DAYS,0,0,0,0,1,1,0,20060701,20060731)"};

constexpr auto const calendar_dates =
    R"(service_id,date,exception_type
T1_DAYS,20060702,1
T3_DAYS,20060702,1
T6_DAYS,20060702,1
)";

constexpr auto const stops = std::string_view{
    R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S1,STOP 1,First Stop,49.880015,8.664131,,,
S2,STOP 2,Second Stop,49.878166,8.661501,,,
S3,STOP 3,Third Stop,49.875208,8.658878,,,
S4,STOP 4,Forth Stop,49.872985,8.655666,,,
S5,STOP 5,Fifth Stop,49.872714,8.650961,,,
S6,STOP 6,Sixth Stop,49.871787,8.643587,,,
S7,STOP 7,Seventh Stop,49.870769,8.636022,,,
S8,STOP 8,Eighth Stop,49.872501,8.628313,,,
)"};

constexpr auto const agency = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone,agency_phone,agency_lang
FunBus,The Fun Bus,http://www.thefunbus.org,Europe/Berlin,(310) 555-0222,en
)"};

auto const routes = std::string_view{
    R"(route_id,route_short_name,route_long_name,route_desc,route_type,agency_id
A,17,A,A Route,1,FunBus
)"};

constexpr auto const trips =
    R"(route_id,service_id,trip_id,trip_headsign,trip_short_name,block_id
A,T1_DAYS,T1,Trip 1,T1,1
A,T2_DAYS,T2,Trip 2,T2,1
A,T3_DAYS,T3,Trip 3,T3,1
A,T4_DAYS,T4,Trip 4,T4,1
A,T5_DAYS,T5,Trip 5,T5,1
A,T6_DAYS,T6,Trip 6,T6,1
)";

constexpr auto const stop_times =
    R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,23:00:00,23:00:00,S1,11,0,0
T1,23:55:00,24:00:00,S2,12,0,0
T2,24:05:00,24:00:00,S2,22,0,0
T2,25:00:00,25:00:00,S3,23,0,0
T3,24:00:00,24:00:00,S2,32,0,0
T3,25:00:00,25:00:00,S3,33,0,0
T3,26:00:00,26:00:00,S4,34,0,0
T4,25:00:00,25:00:00,S3,43,0,0
T4,26:00:00,26:00:00,S4,44,0,0
T4,27:00:00,27:00:00,S5,45,0,0
T5,24:00:00,24:00:00,S2,52,0,0
T5,25:00:00,25:00:00,S3,53,0,0
T5,26:00:00,26:00:00,S4,54,0,0
T5,27:00:00,27:00:00,S5,55,0,0
T5,28:00:00,28:00:00,S6,56,0,0
T5,29:00:00,29:00:00,S7,57,0,0
T6,26:00:00,26:00:00,S4,64,0,0
T6,27:00:00,27:00:00,S5,65,0,0
T6,28:00:00,28:00:00,S6,66,0,0
T6,29:00:00,29:00:00,S7,67,0,0
T6,30:00:00,30:00:00,S8,68,0,0)";

constexpr auto const result =
    std::string_view{R"([2006-07-02 21:00, 2006-07-03 04:00]
TRANSFERS: 0
     FROM: (STOP 1, S1) [2006-07-02 21:00]
       TO: (STOP 8, S8) [2006-07-03 04:00]
leg 0: (STOP 1, S1) [2006-07-02 21:00] -> (STOP 8, S8) [2006-07-03 04:00]
   0: S1      STOP 1..........................................                               d: 02.07 21:00 [02.07 23:00]  [{name=17 T1, day=2006-07-02, id=T1, src=0}]
   1: S2      STOP 2.......................................... a: 02.07 21:55 [02.07 23:55]  d: 02.07 22:00 [03.07 00:00]  [{name=17 T3, day=2006-07-02, id=T3, src=0}]
   2: S3      STOP 3.......................................... a: 02.07 23:00 [03.07 01:00]  d: 02.07 23:00 [03.07 01:00]  [{name=17 T3, day=2006-07-02, id=T3, src=0}]
   3: S4      STOP 4.......................................... a: 03.07 00:00 [03.07 02:00]  d: 03.07 00:00 [03.07 02:00]  [{name=17 T6, day=2006-07-02, id=T6, src=0}]
   4: S5      STOP 5.......................................... a: 03.07 01:00 [03.07 03:00]  d: 03.07 01:00 [03.07 03:00]  [{name=17 T6, day=2006-07-02, id=T6, src=0}]
   5: S6      STOP 6.......................................... a: 03.07 02:00 [03.07 04:00]  d: 03.07 02:00 [03.07 04:00]  [{name=17 T6, day=2006-07-02, id=T6, src=0}]
   6: S7      STOP 7.......................................... a: 03.07 03:00 [03.07 05:00]  d: 03.07 03:00 [03.07 05:00]  [{name=17 T6, day=2006-07-02, id=T6, src=0}]
   7: S8      STOP 8.......................................... a: 03.07 04:00 [03.07 06:00]
leg 1: (STOP 8, S8) [2006-07-03 04:00] -> (STOP 8, S8) [2006-07-03 04:00]
  FOOTPATH (duration=0)
)"};

TEST(gtfs, block_id) {
  using std::filesystem::path;
  auto const files =
      mem_dir{{{path{kAgencyFile}, std::string{agency}},
               {path{kStopFile}, std::string{stops}},
               {path{kCalenderFile}, std::string{calendar}},
               {path{kCalendarDatesFile}, std::string{calendar_dates}},
               {path{kRoutesFile}, std::string{routes}},
               {path{kTripsFile}, std::string{trips}},
               {path{kStopTimesFile}, std::string{stop_times}}}};
  ASSERT_TRUE(applicable(files));

  auto tt = timetable{};
  tt.date_range_ = {2006_y / 7 / 1, 2006_y / 8 / 1};
  load_timetable({}, source_idx_t{0U}, files, tt);
  finalize(tt);

  auto const expect_no_transfers = [](routing::journey const& j) {
    return j.legs_.size() == 1U;
  };

  {
    auto const res = raptor_search(tt, nullptr, "S1", "S8",
                                   "2006-07-02 23:00 Europe/Berlin");
    ASSERT_EQ(1, res.size());
    expect_no_transfers(*res.begin());

    std::stringstream ss;
    res.begin()->print(ss, tt, nullptr, false);
    EXPECT_EQ(result, ss.str());
  }

  {
    auto const res = raptor_search(tt, nullptr, "S2", "S1",
                                   "2006-07-02 23:00 Europe/Berlin");
    ASSERT_EQ(0, res.size());
  }

  {
    auto const res = raptor_search(tt, nullptr, "S2", "S3",
                                   "2006-07-09 00:00 Europe/Berlin");
    ASSERT_EQ(1, res.size());
    expect_no_transfers(*res.begin());
  }

  {
    auto const res = raptor_search(tt, nullptr, "S2", "S7",
                                   "2006-07-09 00:00 Europe/Berlin");
    ASSERT_EQ(1, res.size());
    expect_no_transfers(*res.begin());
  }

  {
    auto const res = raptor_search(tt, nullptr, "S1", "S4",
                                   "2006-07-05 23:00 Europe/Berlin");
    ASSERT_EQ(1, res.size());
    expect_no_transfers(*res.begin());
  }

  {
    auto const res = raptor_search(tt, nullptr, "S1", "S5",
                                   "2006-07-06 23:00 Europe/Berlin");
    ASSERT_EQ(1, res.size());
    expect_no_transfers(*res.begin());
  }

  {
    auto const res = raptor_search(tt, nullptr, "S1", "S7",
                                   "2006-07-07 23:00 Europe/Berlin");
    ASSERT_EQ(1, res.size());
    expect_no_transfers(*res.begin());
  }
}