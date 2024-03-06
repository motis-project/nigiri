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

constexpr auto const timetable_fr2 = R"(
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
stop_name,parent_station,zone_id,stop_id,stop_lat,stop_lon,location_type,stop_timezone,wheelchair_boarding
Chemin de Refectoire,IDFM:69671,,IDFM:StopPlaceEntrance:50170260,48.7449,2.438695,2,,
Chemin de Refectoire,IDFM:69671,,IDFM:StopPlaceEntrance:50170261,48.74471,2.438792,2,,
Mairie Annexe,,,IDFM:69687,48.748676,2.4352267,1,,
Mairie Annexe,IDFM:69687,4,IDFM:17252,48.748886,2.4351294,,Europe/Paris,1
Mairie Annexe,IDFM:69687,4,IDFM:17253,48.74847,2.4353242,,Europe/Paris,1
Parc en Seine,,,IDFM:74101,48.742958,2.4356422,1,,
Parc en Seine,IDFM:74101,4,IDFM:35844,48.742977,2.435465,,Europe/Paris,2
Parc en Seine,IDFM:74101,4,IDFM:35845,48.742943,2.4358191,,Europe/Paris,2
Villeneuve - Triage RER,IDFM:69671,4,IDFM:23365,48.74714,2.437093,,Europe/Paris,1
Villeneuve Triage,,,IDFM:69671,48.745823,2.4379368,1,,
Villeneuve Triage,IDFM:69671,4,IDFM:17234,48.745544,2.4378583,,Europe/Paris,
Villeneuve Triage,IDFM:69671,4,IDFM:472431,48.745068,2.4380267,,Europe/Paris,
Villeneuve Triage,IDFM:69671,4,IDFM:monomodalStopPlace:46304,48.74461,2.4386978,,,
av. de Choisy (accès principal),IDFM:69671,,IDFM:StopPlaceEntrance:50170077,48.745247,2.4380655,2,,
av. de Choisy (accès secondaire),IDFM:69671,,IDFM:StopPlaceEntrance:50170078,48.744896,2.4383004,2,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
IDFM:17234,IDFM:23365,2,194
IDFM:17234,IDFM:472431,2,60
IDFM:17234,IDFM:monomodalStopPlace:46304,2,218
IDFM:17252,IDFM:17253,2,60
IDFM:17252,IDFM:23365,2,307
IDFM:17253,IDFM:17252,2,60
IDFM:17253,IDFM:23365,2,250
IDFM:23365,IDFM:17234,2,194
IDFM:23365,IDFM:17252,2,307
IDFM:23365,IDFM:17253,2,250
IDFM:23365,IDFM:472431,2,251
IDFM:23365,IDFM:monomodalStopPlace:46304,2,412
IDFM:35844,IDFM:35845,2,60
IDFM:35844,IDFM:472431,2,5376
IDFM:35844,IDFM:monomodalStopPlace:46304,2,3060
IDFM:35845,IDFM:35844,2,60
IDFM:35845,IDFM:472431,2,5376
IDFM:35845,IDFM:monomodalStopPlace:46304,2,3060
IDFM:472431,IDFM:17234,2,60
IDFM:472431,IDFM:23365,2,251
IDFM:472431,IDFM:35844,2,5387
IDFM:472431,IDFM:35845,2,5387
IDFM:472431,IDFM:monomodalStopPlace:46304,2,201
IDFM:monomodalStopPlace:46304,IDFM:17234,2,218
IDFM:monomodalStopPlace:46304,IDFM:23365,2,412
IDFM:monomodalStopPlace:46304,IDFM:35844,2,3595
IDFM:monomodalStopPlace:46304,IDFM:35845,2,3595
IDFM:monomodalStopPlace:46304,IDFM:472431,2,201


)";

}  // namespace

TEST(loader, build_footpaths_fr2) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(timetable_fr2), tt);
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

constexpr auto const timetable_fr3 = R"(
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
stop_name,parent_station,stop_code,zone_id,stop_id,stop_lat,stop_lon,location_type,stop_timezone,wheelchair_boarding
Carrefour de la Résistance,,,,IDFM:70609,48.820324,2.467492,1,,
Carrefour de la Résistance,IDFM:70609,,3,IDFM:39400,48.820053,2.4679987,,Europe/Paris,1
Carrefour de la Résistance,IDFM:70609,,3,IDFM:39407,48.820595,2.4669855,,Europe/Paris,1
Carrefour de la Résistance,IDFM:70609,,3,IDFM:7763,48.82022,2.4677927,,Europe/Paris,1
Jean Jaurès - Chapsal,,,,IDFM:421277,48.823063,2.4655845,1,,
Jean Jaurès - Chapsal,IDFM:421277,,3,IDFM:39547,48.823063,2.4655845,,Europe/Paris,1
Joinville-le-Pont,,,,IDFM:70640,48.820507,2.463847,1,,
Joinville-le-Pont,IDFM:70640,,3,IDFM:monomodalStopPlace:43135,48.821198,2.4638739,,,1
Joinville-le-Pont RER,IDFM:70640,,,IDFM:39406,48.820168,2.4639935,,Europe/Paris,1
Joinville-le-Pont RER,IDFM:70640,,3,IDFM:21252,48.821484,2.4642386,,Europe/Paris,1
Joinville-le-Pont RER,IDFM:70640,,3,IDFM:22452,48.820583,2.4639757,,Europe/Paris,1
Joinville-le-Pont RER,IDFM:70640,,3,IDFM:28032,48.820347,2.4639235,,Europe/Paris,1
Joinville-le-Pont RER,IDFM:70640,,3,IDFM:28033,48.821304,2.4642134,,Europe/Paris,1
Joinville-le-Pont RER,IDFM:70640,,3,IDFM:28065,48.81953,2.4643133,,Europe/Paris,1
Joinville-le-Pont RER,IDFM:70640,,3,IDFM:39402,48.820156,2.4633944,,Europe/Paris,1
Joinville-le-Pont-RER,IDFM:70640,,3,IDFM:27560,48.821125,2.464147,,Europe/Paris,1
av. Jean Jaurès,IDFM:70640,3,,IDFM:StopPlaceEntrance:50148574,48.82097,2.4640534,2,,
av. des Canadiens,IDFM:70640,2,,IDFM:StopPlaceEntrance:50148575,48.82038,2.4634602,2,,
r. J. Mermoz,IDFM:70640,1,,IDFM:StopPlaceEntrance:50148576,48.81999,2.4639273,2,,


# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
IDFM:21252,IDFM:22452,2,106
IDFM:21252,IDFM:27560,2,60
IDFM:21252,IDFM:28032,2,134
IDFM:21252,IDFM:28033,2,60
IDFM:21252,IDFM:28065,2,226
IDFM:21252,IDFM:39402,2,167
IDFM:21252,IDFM:39406,2,154
IDFM:21252,IDFM:39407,2,285
IDFM:21252,IDFM:39547,2,256
IDFM:21252,IDFM:7763,2,376
IDFM:21252,IDFM:monomodalStopPlace:43135,2,321
IDFM:22452,IDFM:21252,2,106
IDFM:22452,IDFM:27560,2,64
IDFM:22452,IDFM:28032,2,60
IDFM:22452,IDFM:28033,2,86
IDFM:22452,IDFM:28065,2,125
IDFM:22452,IDFM:39402,2,66
IDFM:22452,IDFM:39406,2,60
IDFM:22452,IDFM:39407,2,280
IDFM:22452,IDFM:39547,2,382
IDFM:22452,IDFM:7763,2,359
IDFM:22452,IDFM:monomodalStopPlace:43135,2,226
IDFM:27560,IDFM:21252,2,60
IDFM:27560,IDFM:22452,2,64
IDFM:27560,IDFM:28032,2,91
IDFM:27560,IDFM:28033,2,60
IDFM:27560,IDFM:28065,2,185
IDFM:27560,IDFM:39402,2,126
IDFM:27560,IDFM:39406,2,111
IDFM:27560,IDFM:39407,2,275
IDFM:27560,IDFM:39547,2,305
IDFM:27560,IDFM:7763,2,363
IDFM:27560,IDFM:monomodalStopPlace:43135,2,281
IDFM:28032,IDFM:21252,2,134
IDFM:28032,IDFM:22452,2,60
IDFM:28032,IDFM:27560,2,91
IDFM:28032,IDFM:28033,2,113
IDFM:28032,IDFM:28065,2,99
IDFM:28032,IDFM:39402,2,60
IDFM:28032,IDFM:39406,2,60
IDFM:28032,IDFM:39407,2,287
IDFM:28032,IDFM:7763,2,361
IDFM:28032,IDFM:monomodalStopPlace:43135,2,216
IDFM:28033,IDFM:21252,2,60
IDFM:28033,IDFM:22452,2,86
IDFM:28033,IDFM:27560,2,60
IDFM:28033,IDFM:28032,2,113
IDFM:28033,IDFM:28065,2,206
IDFM:28033,IDFM:39402,2,147
IDFM:28033,IDFM:39406,2,133
IDFM:28033,IDFM:39407,2,277
IDFM:28033,IDFM:39547,2,280
IDFM:28033,IDFM:7763,2,367
IDFM:28033,IDFM:monomodalStopPlace:43135,2,301
IDFM:28065,IDFM:21252,2,226
IDFM:28065,IDFM:22452,2,125
IDFM:28065,IDFM:27560,2,185
IDFM:28065,IDFM:28032,2,99
IDFM:28065,IDFM:28033,2,206
IDFM:28065,IDFM:39400,2,351
IDFM:28065,IDFM:39402,2,101
IDFM:28065,IDFM:39406,2,78
IDFM:28065,IDFM:39407,2,291
IDFM:28065,IDFM:7763,2,338
IDFM:28065,IDFM:monomodalStopPlace:43135,2,298
IDFM:39400,IDFM:28065,2,351
IDFM:39400,IDFM:39406,2,373
IDFM:39400,IDFM:39407,2,100
IDFM:39400,IDFM:7763,2,60
IDFM:39402,IDFM:21252,2,167
IDFM:39402,IDFM:22452,2,66
IDFM:39402,IDFM:27560,2,126
IDFM:39402,IDFM:28032,2,60
IDFM:39402,IDFM:28033,2,147
IDFM:39402,IDFM:28065,2,101
IDFM:39402,IDFM:39406,2,60
IDFM:39402,IDFM:39407,2,340
IDFM:39402,IDFM:monomodalStopPlace:43135,2,207
IDFM:39406,IDFM:21252,2,154
IDFM:39406,IDFM:22452,2,60
IDFM:39406,IDFM:27560,2,111
IDFM:39406,IDFM:28032,2,60
IDFM:39406,IDFM:28033,2,133
IDFM:39406,IDFM:28065,2,78
IDFM:39406,IDFM:39400,2,373
IDFM:39406,IDFM:39402,2,60
IDFM:39406,IDFM:39407,2,285
IDFM:39406,IDFM:7763,2,354
IDFM:39406,IDFM:monomodalStopPlace:43135,2,228
IDFM:39407,IDFM:21252,2,285
IDFM:39407,IDFM:22452,2,280
IDFM:39407,IDFM:27560,2,275
IDFM:39407,IDFM:28032,2,287
IDFM:39407,IDFM:28033,2,277
IDFM:39407,IDFM:28065,2,291
IDFM:39407,IDFM:39400,2,100
IDFM:39407,IDFM:39402,2,340
IDFM:39407,IDFM:39406,2,285
IDFM:39407,IDFM:39547,2,373
IDFM:39407,IDFM:7763,2,76
IDFM:39407,IDFM:monomodalStopPlace:43135,2,302
IDFM:39547,IDFM:21252,2,256
IDFM:39547,IDFM:22452,2,382
IDFM:39547,IDFM:27560,2,305
IDFM:39547,IDFM:28033,2,280
IDFM:39547,IDFM:39407,2,373
IDFM:39547,IDFM:monomodalStopPlace:43135,2,308
IDFM:7763,IDFM:21252,2,376
IDFM:7763,IDFM:22452,2,359
IDFM:7763,IDFM:27560,2,363
IDFM:7763,IDFM:28032,2,361
IDFM:7763,IDFM:28033,2,367
IDFM:7763,IDFM:28065,2,338
IDFM:7763,IDFM:39400,2,60
IDFM:7763,IDFM:39406,2,354
IDFM:7763,IDFM:39407,2,76
IDFM:monomodalStopPlace:43135,IDFM:21252,2,241
IDFM:monomodalStopPlace:43135,IDFM:22452,2,225
IDFM:monomodalStopPlace:43135,IDFM:27560,2,199
IDFM:monomodalStopPlace:43135,IDFM:28032,2,216
IDFM:monomodalStopPlace:43135,IDFM:28033,2,221
IDFM:monomodalStopPlace:43135,IDFM:28065,2,241
IDFM:monomodalStopPlace:43135,IDFM:39402,2,207
IDFM:monomodalStopPlace:43135,IDFM:39406,2,202
IDFM:monomodalStopPlace:43135,IDFM:39407,2,302
IDFM:monomodalStopPlace:43135,IDFM:39547,2,308



)";

}  // namespace

TEST(loader, build_footpaths_fr3) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(timetable_fr3), tt);
  loader::finalize(tt);

  constexpr auto const from = "IDFM:28033";
  constexpr auto const to = "IDFM:monomodalStopPlace:43135";
  auto ss = std::stringstream{};
  for (auto const [i, x] : utl::enumerate(tt.locations_.footpaths_out_[0])) {
    if (!x.empty()) {
      for (auto const y : x) {
        if (location{tt, location_idx_t{i}}.id_ == from &&
            location{tt, y.target()}.id_ == to) {
          ss << location{tt, location_idx_t{i}} << " --" << y.duration()
             << "--> " << location{tt, y.target()} << "\n";
        }
      }
    }
  }

  std::cout << ss.view() << "\n";
}