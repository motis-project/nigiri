#include "gtest/gtest.h"

#include <string>
#include <string_view>

#include "fmt/format.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace date;
using nigiri::test::raptor_search;

namespace {

// Stops A, B (a few km apart) on 2019-05-01, the trips (route R) with their
// stop times and the stay-seated rows (transfer_type 4).
timetable load(std::string_view const trips,
               std::string_view const stop_times,
               std::string_view const transfers) {
  auto const files = fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.0,6.0
B,B,50.0,6.5

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R,AG,R,,3

# trips.txt
route_id,service_id,trip_id
{}
# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
{}
# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_trip_id,to_trip_id
{}
)",
                                 trips, stop_times, transfers);
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2019_y / May / 1}, sys_days{2019_y / May / 2}};
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0}, mem_dir::read(files), tt);
  finalize(tt);
  return tt;
}

unixtime_t at(std::string_view const hhmm) {
  return parse_time_tz(fmt::format("2019-05-01 {} Europe/Berlin", hhmm),
                       "%Y-%m-%d %H:%M %Z");
}

std::optional<unixtime_t> arrival(timetable const& tt,
                                  std::string_view const from,
                                  std::string_view const to,
                                  std::string_view const hhmm) {
  auto const res = raptor_search(
      tt, nullptr, from, to, fmt::format("2019-05-01 {} Europe/Berlin", hhmm));
  return res.size() == 0U ? std::nullopt
                          : std::optional{begin(res)->dest_time_};
}

}  // namespace

// A trip that stays seated in itself is just a trip.
TEST(gtfs, seated_self_loop_keeps_trip) {
  auto const tt = load("R,S1,T\n",
                       "T,10:00:00,10:00:00,A,1\n"
                       "T,10:30:00,10:30:00,B,2\n",
                       "B,A,4,,T,T\n");
  EXPECT_EQ(at("10:30"), arrival(tt, "A", "B", "10:00"));
}

// T1 -> T2 -> T1: T2 -> T1 is no valid dwell (T1 leaves 23 h after T2
// arrives), so the chain starts with T1 - neither trip may get lost.
TEST(gtfs, seated_back_edge_keeps_trips) {
  auto const tt = load("R,S1,T1\nR,S1,T2\n",
                       "T1,10:00:00,10:00:00,A,1\n"
                       "T1,10:30:00,10:30:00,B,2\n"
                       "T2,10:40:00,10:40:00,B,1\n"
                       "T2,11:00:00,11:00:00,A,2\n",
                       "B,B,4,,T1,T2\nA,A,4,,T2,T1\n");
  EXPECT_EQ(at("10:30"), arrival(tt, "A", "B", "10:00"));
  EXPECT_EQ(at("11:00"), arrival(tt, "B", "A", "10:35"));
}
