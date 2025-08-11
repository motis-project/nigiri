#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "../../../include/nigiri/routing/direct.h"

#include "../hrd/hrd_timetable.h"

#include "../../raptor_search.h"
#include "../../routing/results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Paris

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# calendar_dates.txt
service_id,date,exception_type
S_RE1,20190501,1
S_RE2,20190503,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,1,,,3
R_RE2,DB,2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,RE 1,
R_RE2,S_RE2,T_RE2,RE 2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,49:00:00,49:00:00,A,1,0,0
T_RE1,50:00:00,50:00:00,B,2,0,0
T_RE2,00:30:00,00:30:00,B,1,0,0
T_RE2,00:45:00,00:45:00,C,2,0,0
T_RE2,01:00:00,01:00:00,D,3,0,0
)");
}

}  // namespace

TEST(gtfs, lua_test) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({.lua_user_script_ = R"(
function update_route(route)
  if route:get_route_type() == 3:
    route:set_clasz(7)
    route:set_route_type(101)
  elseif route:get_route_type() == 1:
    route:set_clasz(8)
    route:set_route_type(400)
  end

  if route:get_agency():get_name() == 'Deutsche Bahn' and route:get_route_type() == 101:
    route:set_route_short_name('RE ' .. route:get_route_short_name())
  end
end


function update_agency(agency)
  if agency == 'Deutsche Bahn':
    agency:set_timezone('Europe/Berlin')
    agency:set_url('https://bahn.de')
  end
end


function update_trip(trip)
  if trip:get_route():get_route_type() == 101:
    -- Prepend category and eliminate leading zeros (e.g. '00123' -> 'ICE 123')
    trip:set_trip_short_name('ICE ' .. string.format("%u", trip:get_short_name()))
  end
end
)"},
                 source_idx_t{0}, test_files(), tt);
  finalize(tt);
}