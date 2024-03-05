#include "gtest/gtest.h"

#include "utl/enumerate.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

// ROUTING CONNECTIONS:
// 10:00 - 11:00 A-C    airplane direct
// 10:00 - 12:00 A-B-C  train, one transfer
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type

# calendar_dates.txt
service_id,date,exception_type

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
A,B,2,180
B,C,2,180
)"sv;

}  // namespace

TEST(loader, build_footpaths) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files), tt);
  loader::finalize(tt);

  auto ss = std::stringstream{};
  for (auto const [i, x] : utl::enumerate(tt.locations_.footpaths_out_[0])) {
    if (!x.empty()) {
      ss << location{tt, location_idx_t{i}} << "\n";
      for (auto const y : x) {
        ss << "  " << y.duration() << "->" << location{tt, y.target()} << "\n";
      }
    }
  }

  EXPECT_EQ(R"((A, A)
  00:03.0->(B, B)
  00:06.0->(C, C)
(B, B)
  00:03.0->(A, A)
  00:03.0->(C, C)
(C, C)
  00:06.0->(A, A)
  00:03.0->(B, B)
)"sv,
            ss.str());
}

namespace {

constexpr auto const timetable_fr = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type

# calendar_dates.txt
service_id,date,exception_type

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lon,stop_lat,zone_id,stop_url,location_type,parent_station,stop_timezone,level_id,wheelchair_boarding,platform_code
IDFM:2921,,Gare de Breuillet Village,,2.171831602352254,48.56476273942137,4,,0,IDFM:59940,Europe/Paris,,1,
IDFM:18740,,Gare de Breuillet Village,,2.1718863797167716,48.56472713695579,4,,0,IDFM:59940,Europe/Paris,,1,
IDFM:3646,,Gare de Breuillet Village,,2.1723438872522034,48.5649372275467,4,,0,IDFM:59940,Europe/Paris,,1,
IDFM:monomodalStopPlace:43099,,Breuillet Village,,2.170354257290333,48.56433413404574,5,,0,IDFM:59940,,,0,
IDFM:StopPlaceEntrance:50170822,,r. de la Gare (bâtiment voyageur),,2.171354288391762,48.56494833051335,,,2,IDFM:59940,,,0,
IDFM:StopPlaceEntrance:50170823,,r. de la Gare (bus),,2.1717500047925014,48.56478016324463,,,2,IDFM:59940,,,0,
IDFM:StopPlaceEntrance:50170827,,r. de la Gare (parking),,2.171571124479157,48.56494984068485,,,2,IDFM:59940,,,0,
IDFM:StopPlaceEntrance:50170828,,r. de la Gare (passage à niveau),,2.170238281207967,48.56438280290591,,,2,IDFM:59940,,,0,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
IDFM:18740,IDFM:2921,2,60
IDFM:18740,IDFM:3646,2,60
IDFM:18740,IDFM:monomodalStopPlace:43099,2,192
IDFM:2921,IDFM:18740,2,60
IDFM:2921,IDFM:3646,2,60
IDFM:2921,IDFM:monomodalStopPlace:43099,2,187
IDFM:3646,IDFM:18740,2,60
IDFM:3646,IDFM:2921,2,60
IDFM:3646,IDFM:monomodalStopPlace:43099,2,229
IDFM:monomodalStopPlace:43099,IDFM:18740,2,192
IDFM:monomodalStopPlace:43099,IDFM:2921,2,187
IDFM:monomodalStopPlace:43099,IDFM:3646,2,229

)";

}  // namespace

TEST(loader, build_footpaths_fr) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(timetable_fr), tt);
  loader::finalize(tt);

  auto ss = std::stringstream{};
  for (auto const [i, x] : utl::enumerate(tt.locations_.footpaths_out_[0])) {
    if (!x.empty()) {
      ss << location{tt, location_idx_t{i}} << "\n";
      for (auto const y : x) {
        ss << "  " << y.duration() << "->" << location{tt, y.target()} << "\n";
      }
    }
  }

  std::cout << ss.view() << "\n";
}