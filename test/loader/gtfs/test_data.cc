#include "./test_data.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs {

constexpr auto const example_calendar_file_content = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WE,0,0,0,0,0,1,1,20060701,20060731
WD,1,1,1,1,1,0,0,20060701,20060731
)"};

constexpr auto const example_calendar_dates_file_content =
    R"(service_id,date,exception_type
WD,20060703,2
WE,20060703,1
WD,20060704,2
WE,20060704,1
)";

constexpr auto const example_stops_file_content = std::string_view{
    R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S1,Mission St. & Silver Ave.,The stop is located at the southwest corner of the intersection.,37.728631,-122.431282,,,
S2,Mission St. & Cortland Ave.,The stop is located 20 feet south of Mission St.,37.74103,-122.422482,,,
S3,Mission St. & 24th St.,The stop is located at the southwest corner of the intersection.,37.75223,-122.418581,,,
S4,Mission St. & 21st St.,The stop is located at the northwest corner of the intersection.,37.75713,-122.418982,,,
S5,Mission St. & 18th St.,The stop is located 25 feet west of 18th St.,37.761829,-122.419382,,,
S6,Mission St. & 15th St.,The stop is located 10 feet north of Mission St.,37.766629,-122.419782,,,
S7,24th St. Mission Station,,37.752240,-122.418450,,,S8
S8,24th St. Mission Station,,37.752240,-122.418450,http://www.bart.gov/stations/stationguide/stationoverview_24st.asp,1,
)"};

constexpr auto const example_agency_file_content = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,http://google.com,America/Los_Angeles
"11","Schweizerische Bundesbahnen SBB","http://www.sbb.ch/","Europe/Berlin","DE","0900 300 300 "
)"};

auto const example_routes_file_content = std::string_view{
    R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,DTA,17,Mission,"The ""A"" route travels from lower Mission to Downtown.",3
)"};

constexpr auto const example_transfers_file_content =
    std::string_view{R"(from_stop_id,to_stop_id,transfer_type,min_transfer_time
S6,S7,2,300
S7,S6,3,
S7,S7,2,900
)"};

constexpr auto const example_shapes_file_content =
    R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
123,51.526339,14.003664,0
123,51.520679,13.980126,1
123,51.520679,13.980126,2
123,51.514264,13.985332,3
)";

constexpr auto const example_trips_file_content =
    R"(route_id,service_id,trip_id,trip_headsign,block_id
A,WE,AWE1,Downtown,1
A,WE,AWD1,Downtown,2
)";

constexpr auto const example_frequencies_file_content =
    std::string_view{R"(trip_id,start_time,end_time,headway_secs
AWE1,05:30:00,06:30:00,300
AWE1,06:30:00,20:30:00,180
AWE1,20:30:00,28:00:00,420
)"};

constexpr auto const example_stop_times_content =
    R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,shape_dist_traveled,
AWD1,6:45,6:45,S6,6,0,0,,
AWD1,,,S5,5,0,0,,
AWD1,,,S4,4,0,0,0.0,
AWD1,6:20,6:20,S3,3,0,0,0.0,
AWD1,,,S2,2,0,0,,
AWD1,6:10,6:10,S1,1,0,0,,
AWE1,6:45,6:45,S6,5,0,0,,
AWE1,,,S5,4,0,0,,
AWE1,6:20,6:30,S3,3,0,0,5.0,
AWE1,,,S2,2,1,3,3.14,
AWE1,6:10,6:10,S1,1,0,0,,
)";

loader::mem_dir example_files() {
  using std::filesystem::path;
  return {
      {{path{kAgencyFile}, std::string{example_agency_file_content}},
       {path{kStopFile}, std::string{example_stops_file_content}},
       {path{kCalenderFile}, std::string{example_calendar_file_content}},
       {path{kCalendarDatesFile},
        std::string{example_calendar_dates_file_content}},
       {path{kTransfersFile}, std::string{example_transfers_file_content}},
       {path{kRoutesFile}, std::string{example_routes_file_content}},
       {path{kFrequenciesFile}, std::string{example_frequencies_file_content}},
       {path{kShapesFile}, std::string{example_shapes_file_content}},
       {path{kTripsFile}, std::string{example_trips_file_content}},
       {path{kStopTimesFile}, std::string{example_stop_times_content}}}};
}

constexpr auto const berlin_agencies_file_content = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
ANG---,Günter Anger Güterverkehrs GmbH & Co. Omnibusvermietung KG,http://www.anger-busvermietung.de,Europe/Berlin,de,033208 22010
BMO---,Busverkehr Märkisch-Oderland GmbH,http://www.busmol.de,Europe/Berlin,de,03341 478383
N04---,DB Regio AG,http://www.bahn.de/brandenburg,Europe/Berlin,de,0331 2356881
BON---,Busverkehr Oder-Spree GmbH,http://www.bos-fw.de,Europe/Berlin,de,03361 556133
)"};

constexpr auto const berlin_stops_file_content = std::string_view{
    R"(stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station
5100071,,Zbaszynek,,52.2425040,15.8180870,,,0,
9230005,,S Potsdam Hauptbahnhof Nord,,52.3927320,13.0668480,,,0,
9230006,,"Potsdam, Charlottenhof Bhf",,52.3930040,13.0362980,,,0,
)"};

constexpr auto const berlin_calendar_file_content = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
000001,0,0,0,0,0,0,0,20150409,20151212
000002,0,0,0,0,0,0,0,20150409,20151212
000856,0,0,0,0,0,0,0,20150409,20151212
000861,0,0,0,0,0,0,0,20150409,20151212
)"};

constexpr auto const berlin_calendar_dates_file_content =
    std::string_view{R"(service_id,exception_type,date
7,1,20211217
)"};

constexpr auto const berlin_transfers_file_content =
    R"(from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_transfer_id,to_transfer_id
9003104,9003174,2,180,,
9003104,9003175,2,240,,
9003104,9003176,2,180,,
9003174,9003104,2,180,,
9003174,9003175,2,180,,)";

constexpr auto const berlin_routes_file_content =
    R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,route_url,route_color,route_text_color
1,ANG---,SXF2,,,700,http://www.vbb.de,,
10,BMO---,927,,,700,http://www.vbb.de,,
2,BON---,548,,,700,http://www.vbb.de,,
809,N04---,,"Leisnig -- Leipzig, Hauptbahnhof",,100,http://www.vbb.de,,
81,BON---,2/412,,,700,http://www.vbb.de,,
810,N04---,,"S+U Lichtenberg Bhf (Berlin) -- Senftenberg, Bahnhof",,100,http://www.vbb.de,,
811,N04---,,"S+U Lichtenberg Bhf (Berlin) -- Altdöbern, Bahnhof",,100,http://www.vbb.de,,
812,N04---,RB14,,,100,http://www.vbb.de,B10093,FFFFFF
F11,F04---,,,,1203,,,
)";

constexpr auto const berlin_shapes_file_content =
    R"("shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"
101,51.851349,13.710569,16
101,51.852000,13.708819,25
101,51.854863,13.704366,55
101,51.862833,13.703288,67
101,51.872519,13.698557,84
102,51.917357,13.583086,102
102,51.917640,13.578006,123
102,51.917317,13.578529,190
102,51.892762,13.609018,325
102,51.851055,13.711492,461
102,51.850520,13.712951,464
103,51.917047,13.582706,113
103,51.895977,13.622296,186
103,51.854119,13.704374,283
103,51.852374,13.709008,300
)";

constexpr auto const berlin_trips_file_content =
    R"(route_id,service_id,trip_id,trip_headsign,trip_short_name,direction_id,block_id,shape_id
1,000856,1,Flughafen Schönefeld Terminal (Airport),,,1,101
1,000856,2,S Potsdam Hauptbahnhof,,,2,102
2,000861,3,"Golzow (PM), Schule",,,3,103
)";

loader::mem_dir berlin_files() {
  using std::filesystem::path;
  return {{{path{kAgencyFile}, std::string{berlin_agencies_file_content}},
           {path{kStopFile}, std::string{berlin_stops_file_content}},
           {path{kCalenderFile}, std::string{berlin_calendar_file_content}},
           {path{kCalendarDatesFile},
            std::string{berlin_calendar_dates_file_content}},
           {path{kTransfersFile}, std::string{berlin_transfers_file_content}},
           {path{kRoutesFile}, std::string{berlin_routes_file_content}},
           {path{kShapesFile}, std::string{berlin_shapes_file_content}},
           {path{kTripsFile}, std::string{berlin_trips_file_content}}}};
}

}  // namespace nigiri::loader::gtfs
