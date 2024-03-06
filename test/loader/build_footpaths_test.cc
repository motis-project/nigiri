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

namespace {

constexpr auto const timetable_fr1 = R"(
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
IDFM:19713,,Square de Camargue,,2.0002342111935607,48.822929128447626,5,,0,IDFM:64127,Europe/Paris,,2,
IDFM:5902,,Docteur Vaillant,,1.9961640981383888,48.826330811496575,5,,0,IDFM:64161,Europe/Paris,,2,
IDFM:5676,,Docteur Vaillant,,1.995917245354705,48.8263764801907,5,,0,IDFM:64161,Europe/Paris,,2,
IDFM:14077,,Crozatier,,2.0007860710164294,48.82967037692149,5,,0,IDFM:480612,Europe/Paris,,2,
IDFM:5921,,Saintonge,,1.9982696635974295,48.830179980351666,5,,0,IDFM:64191,Europe/Paris,,2,
IDFM:5919,,Pyrénées,,1.999958021659294,48.82454577404645,5,,0,IDFM:64142,Europe/Paris,,1,
IDFM:5912,,Mairie-Gymnase Mimoun,,2.00355266723001,48.830323049842235,5,,0,IDFM:73729,Europe/Paris,,2,
IDFM:5910,,Le Square,,2.001367926461794,48.82703095511505,5,,0,IDFM:64166,Europe/Paris,,2,
IDFM:412344,,Square de Camargue,,2.000371090478694,48.82289429599083,5,,0,IDFM:64127,Europe/Paris,,2,
IDFM:5908,,Haie Bergerie,,1.9986680517130775,48.82717027393197,5,,0,IDFM:64169,Europe/Paris,,2,
IDFM:19714,,Collège Léon Blum,,1.9987567083963635,48.83106548473595,5,,0,IDFM:64191,Europe/Paris,,1,
IDFM:14078,,Crozatier,,2.00060107015779,48.82937202420584,5,,0,IDFM:480612,Europe/Paris,,2,
IDFM:5922,,Saintonge,,1.9983938063113618,48.83010007279966,5,,0,IDFM:64191,Europe/Paris,,2,
IDFM:5920,,Pyrénées,,1.9996368506861923,48.82424627942685,5,,0,IDFM:64142,Europe/Paris,,1,
IDFM:19716,,Jean de la Fontaine,,2.003281818560587,48.82738870023968,5,,0,IDFM:64168,Europe/Paris,,2,
IDFM:5911,,Mairie-Gymnase Mimoun,,2.003020592266038,48.83036358617962,5,,0,IDFM:73729,Europe/Paris,,2,
IDFM:5909,,Le Square,,2.001663261992572,48.827258275128976,5,,0,IDFM:64166,Europe/Paris,,2,
IDFM:5907,,Haie Bergerie,,1.9993233520042182,48.827805350718045,5,,0,IDFM:64169,Europe/Paris,,2,
IDFM:480612,,Crozatier,,2.000693570322423,48.82952120060935,,,1,,,,0,
IDFM:64142,,Pyrénées,,1.9997974357115234,48.82439602685445,,,1,,,,0,
IDFM:64169,,Haie Bergerie,,1.9989956998630334,48.82748781281877,,,1,,,,0,
IDFM:73729,,Mairie-Gymnase Mimoun,,2.003286629851455,48.830343318306994,,,1,,,,0,
IDFM:64127,,Square de Camargue,,2.000302650858988,48.82291171223894,,,1,,,,0,
IDFM:64168,,Jean de la Fontaine,,2.003281818560587,48.82738870023968,,,1,,,,0,
IDFM:64161,,Docteur Vaillant,,1.9960410866355103,48.82635226428657,,,1,,,,0,
IDFM:64166,,Le Square,,2.0015155939052542,48.82714461521893,,,1,,,,0,
IDFM:64191,,Collège Léon Blum / Saintonge,,1.9985139613705365,48.83058226575039,,,1,,,,0,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
IDFM:14077,IDFM:14078,2,60
IDFM:14077,IDFM:19714,2,273
IDFM:14077,IDFM:5907,2,297
IDFM:14077,IDFM:5909,2,351
IDFM:14077,IDFM:5910,2,377
IDFM:14077,IDFM:5911,2,230
IDFM:14077,IDFM:5912,2,274
IDFM:14077,IDFM:5921,2,245
IDFM:14077,IDFM:5922,2,231
IDFM:14078,IDFM:14077,2,60
IDFM:14078,IDFM:19714,2,295
IDFM:14078,IDFM:19716,2,376
IDFM:14078,IDFM:5907,2,251
IDFM:14078,IDFM:5908,2,360
IDFM:14078,IDFM:5909,2,315
IDFM:14078,IDFM:5910,2,339
IDFM:14078,IDFM:5911,2,265
IDFM:14078,IDFM:5912,2,306
IDFM:14078,IDFM:5921,2,245
IDFM:14078,IDFM:5922,2,230
IDFM:19713,IDFM:412344,2,60
IDFM:19713,IDFM:5919,2,230
IDFM:19713,IDFM:5920,2,194
IDFM:19714,IDFM:14077,2,273
IDFM:19714,IDFM:14078,2,295
IDFM:19714,IDFM:5921,2,109
IDFM:19714,IDFM:5922,2,115
IDFM:19716,IDFM:14078,2,376
IDFM:19716,IDFM:5907,2,373
IDFM:19716,IDFM:5909,2,152
IDFM:19716,IDFM:5910,2,185
IDFM:412344,IDFM:19713,2,60
IDFM:412344,IDFM:5919,2,237
IDFM:412344,IDFM:5920,2,203
IDFM:5676,IDFM:5902,2,60
IDFM:5676,IDFM:5907,2,376
IDFM:5676,IDFM:5908,2,280
IDFM:5902,IDFM:5676,2,60
IDFM:5902,IDFM:5907,2,361
IDFM:5902,IDFM:5908,2,262
IDFM:5907,IDFM:14077,2,297
IDFM:5907,IDFM:14078,2,251
IDFM:5907,IDFM:19716,2,373
IDFM:5907,IDFM:5676,2,376
IDFM:5907,IDFM:5902,2,361
IDFM:5907,IDFM:5908,2,89
IDFM:5907,IDFM:5909,2,231
IDFM:5907,IDFM:5910,2,219
IDFM:5907,IDFM:5921,2,350
IDFM:5907,IDFM:5922,2,336
IDFM:5908,IDFM:14078,2,360
IDFM:5908,IDFM:5676,2,280
IDFM:5908,IDFM:5902,2,262
IDFM:5908,IDFM:5907,2,89
IDFM:5908,IDFM:5909,2,279
IDFM:5908,IDFM:5910,2,252
IDFM:5909,IDFM:14077,2,351
IDFM:5909,IDFM:14078,2,315
IDFM:5909,IDFM:19716,2,152
IDFM:5909,IDFM:5907,2,231
IDFM:5909,IDFM:5908,2,279
IDFM:5909,IDFM:5910,2,60
IDFM:5910,IDFM:14077,2,377
IDFM:5910,IDFM:14078,2,339
IDFM:5910,IDFM:19716,2,185
IDFM:5910,IDFM:5907,2,219
IDFM:5910,IDFM:5908,2,252
IDFM:5910,IDFM:5909,2,60
IDFM:5910,IDFM:5919,2,375
IDFM:5911,IDFM:14077,2,230
IDFM:5911,IDFM:14078,2,265
IDFM:5911,IDFM:5912,2,60
IDFM:5912,IDFM:14077,2,274
IDFM:5912,IDFM:14078,2,306
IDFM:5912,IDFM:5911,2,60
IDFM:5919,IDFM:19713,2,230
IDFM:5919,IDFM:412344,2,237
IDFM:5919,IDFM:5910,2,375
IDFM:5919,IDFM:5920,2,60
IDFM:5920,IDFM:19713,2,194
IDFM:5920,IDFM:412344,2,203
IDFM:5920,IDFM:5919,2,60
IDFM:5921,IDFM:14077,2,245
IDFM:5921,IDFM:14078,2,245
IDFM:5921,IDFM:19714,2,109
IDFM:5921,IDFM:5907,2,350
IDFM:5921,IDFM:5922,2,60
IDFM:5922,IDFM:14077,2,231
IDFM:5922,IDFM:14078,2,230
IDFM:5922,IDFM:19714,2,115
IDFM:5922,IDFM:5907,2,336
IDFM:5922,IDFM:5921,2,60

)";

}  // namespace

TEST(loader, build_footpaths_fr1) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(timetable_fr1), tt);
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