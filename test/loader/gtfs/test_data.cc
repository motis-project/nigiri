#include "./test_data.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs {

constexpr auto const example_calendar_file_content = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WE,0,0,0,0,0,1,1,20060701,20060731
WD,1,1,1,1,1,0,0,20060701,20060731)"};

constexpr auto const example_calendar_dates_file_content =
    R"(service_id,date,exception_type
WD,20060703,2
WE,20060703,1
WD,20060704,2
WE,20060704,1)";

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
"11","Schweizerische Bundesbahnen SBB","http://www.sbb.ch/","Europe/Berlin","DE","0900 300 300 ")"};

auto const example_routes_file_content = std::string_view{
    R"(route_id,route_short_name,route_long_name,route_desc,route_type
A,17,Mission,"The ""A"" route travels from lower Mission to Downtown.",3)"};

constexpr auto const example_transfers_file_content =
    std::string_view{R"(from_stop_id,to_stop_id,transfer_type,min_transfer_time
S6,S7,2,300
S7,S6,3,
)"};

loader::mem_dir example_files() {
  using std::filesystem::path;
  return {{{path{kAgencyFile}, std::string{example_agency_file_content}},
           {path{kStopFile}, std::string{example_stops_file_content}},
           {path{kCalenderFile}, std::string{example_calendar_file_content}},
           {path{kCalendarDatesFile},
            std::string{example_calendar_dates_file_content}},
           {path{kTransfersFile}, std::string{example_transfers_file_content}},
           {path{kRoutesFile}, std::string{example_routes_file_content}}}};
}

}  // namespace nigiri::loader::gtfs