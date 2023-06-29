#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "../loader/hrd/hrd_timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

mem_dir test_files() {
  using std::filesystem::path;
  return {
      {{path{kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
)"}},
       {path{kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
)"}},
       {path{kCalendarDatesFile}, std::string{R"(service_id,date,exception_type
S1,20190501,1
)"}},
       {path{kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3
R2,DB,RE 2,,,3
)"}},
       {path{kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,1
R2,S1,T2,RE 2,1
)"}},
       {path{kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:30:00,00:30:00,A,1,0,0
T1,10:00:00,10:00:00,B,2,0,0
T2,26:10:00,26:10:00,B,1,0,0
T2,27:00:00,27:00:00,C,2,0,0
T2,28:00:00,28:00:00,D,3,0,0
)"}}}};
}

}  // namespace

constexpr auto const fwd_journeys = R"()";

TEST(rt, rt_block_id_test) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  auto const rtt = rt::create_rt_timetable(tt, May / 1 / 2019);
}