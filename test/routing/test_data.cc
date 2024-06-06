#include "./test_data.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::routing {

constexpr auto const shortest_fp_agency_file_content = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin
)"};

constexpr auto const shortest_fp_stops_file_content = std::string_view{
    R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A0,A0,start_offset0,,,,,,
A1,A1,start_offset1,,,,,,
A2,A2,start_offset2,,,,,,
A3,A3,first_transfer_to_B,,,,,,
A4,A4,second_transfer_to_B,,,,,,
A5,A5,final_stop_of_A,,,,,,
B0,B0,first_transfer_from_A,,,,,,
B1,B1,second_transfer_from_A,,,,,,
B2,B2,first_transfer_to_C,,,,,,
B3,B3,second_transfer_to_C,,,,,,
C0,C0,start_no_transfer,,,,,,
C1,C1,first_transfer_from_B,,,,,,
C2,C2,second_transfer_from_B,,,,,,
C3,C3,dest_offset0,,,,,,
C4,C4,dest_offset1,,,,,,
C5,C5,dest_offset2,,,,,,
)"};

constexpr auto const shortest_fp_routes_file_content = std::string_view{
    R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,MTA,A,A,A0 -> A5,0
B,MTA,B,B,B0 -> B3,0
C,MTA,C,C,C0 -> C5,0
)"};

constexpr auto const shortest_fp_trips_file_content = std::string_view{
    R"(route_id,service_id,trip_id,trip_headsign,block_id
A,WE,AWE,AWE,1
B,WE,BWE,BWE,2
C,WE,CWE,CWE,3
)"};

constexpr auto const shortest_fp_stop_times_file_content = std::string_view{
    R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AWE,02:00,02:00,A0,0,0,0
AWE,02:01,02:02,A1,1,0,0
AWE,02:06,02:07,A2,2,0,0
AWE,02:15,02:16,A3,3,0,0
AWE,02:20,02:21,A4,4,0,0
AWE,02:25,02:26,A5,5,0,0
BWE,02:58,03:00,B0,0,0,0
BWE,03:20,03:22,B1,1,0,0
BWE,03:30,03:32,B2,2,0,0
BWE,03:40,03:42,B3,3,0,0
CWE,03:58,04:00,C0,0,0,0
CWE,04:08,04:10,C1,1,0,0
CWE,04:18,04:20,C2,2,0,0
CWE,04:30,04:32,C3,3,0,0
CWE,04:40,04:42,C4,4,0,0
CWE,04:50,04:52,C5,5,0,0
)"};

constexpr auto const shortest_fp_calendar_file_content = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WE,0,0,0,0,0,1,1,20240101,20241231
WD,1,1,1,1,1,0,0,20240101,20241231
)"};

constexpr auto const shortest_fp_calendar_dates_file_content =
    std::string_view{R"(service_id,exception_type,date
)"};

constexpr auto const shortest_fp_transfers_file_content =
    std::string_view{R"(from_stop_id,to_stop_id,transfer_type,min_transfer_time
A3,B0,2,300
A4,B1,2,600
B2,C1,2,600
B3,C2,2,300
)"};

loader::mem_dir shortest_fp_files() {
  using std::filesystem::path;
  using namespace loader::gtfs;
  return {
      {{path{kAgencyFile}, std::string{shortest_fp_agency_file_content}},
       {path{kStopFile}, std::string{shortest_fp_stops_file_content}},
       {path{kRoutesFile}, std::string{shortest_fp_routes_file_content}},
       {path{kTripsFile}, std::string{shortest_fp_trips_file_content}},
       {path{kStopTimesFile}, std::string{shortest_fp_stop_times_file_content}},
       {path{kCalenderFile}, std::string{shortest_fp_calendar_file_content}},
       {path{kCalendarDatesFile},
        std::string{shortest_fp_calendar_dates_file_content}},
       {path{kTransfersFile},
        std::string{shortest_fp_transfers_file_content}}}};
}

}  // namespace nigiri::routing