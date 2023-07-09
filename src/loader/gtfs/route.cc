#include "nigiri/loader/gtfs/route.h"

#include "utl/get_or_create.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

clasz to_clasz(int const route_type) {
  switch (route_type) {
    case 0 /* Tram, Streetcar, Light rail. Any light rail or street level system within a metropolitan area. */ :
      return clasz::kTram;
    case 1 /* Subway, Metro. Any underground rail system within a metropolitan area. */ :
      return clasz::kSubway;
    case 2 /* Rail. Used for intercity or long-distance travel. */:
      return clasz::kRegionalFast;
    case 3 /* Bus. Used for short- and long-distance bus routes. */:
      return clasz::kBus;
    case 4 /* Ferry. Used for short- and long-distance boat service. */:
      return clasz::kShip;
    case 5 /* Cable tram. Used for street-level rail cars where the cable runs beneath the vehicle, e.g., cable car in San Francisco. */ :
      return clasz::kTram;
    case 6 /* Aerial lift, suspended cable car (e.g., gondola lift, aerial tramway). Cable transport where cabins, cars, gondolas or open chairs are suspended by means of one or more cables. */ :
    case 7 /* Funicular. Any rail system designed for steep inclines. */:
      return clasz::kOther;
    case 11 /* Trolleybus. Electric buses that draw power from overhead wires using poles. */ :
      return clasz::kBus;
    case 12 /* Monorail. Railway in which the track consists of a single rail or a beam. */ :
      return clasz::kOther;
    case 100 /* Railway Service */: return clasz::kRegional;
    case 101 /* High Speed Rail Service */: return clasz::kHighSpeed;
    case 102 /* Long Distance Trains */: return clasz::kLongDistance;
    case 103 /* Inter Regional Rail Service */: return clasz::kRegional;
    case 104 /* Car Transport Rail Service */: return clasz::kLongDistance;
    case 105 /* Sleeper Rail Service */: return clasz::kNight;
    case 106 /* Regional Rail Service */:
    case 107 /* Tourist Railway Service */:
    case 108 /* Rail Shuttle (Within Complex) */: return clasz::kRegional;
    case 109 /* Suburban Railway */: return clasz::kMetro;
    case 110 /* Replacement Rail Service */:
    case 111 /* Special Rail Service */:
    case 112 /* Lorry Transport Rail Service */:
    case 113 /* All Rail Services */: return clasz::kRegional;
    case 114 /* Cross-Country Rail Service */: return clasz::kLongDistance;
    case 115 /* Vehicle Transport Rail Service */:
    case 116 /* Rack and Pinion Railway */:
    case 117 /* Additional Rail Service */: return clasz::kRegional;
    case 200 /* Coach Service */:
    case 201 /* International Coach Service */:
    case 202 /* National Coach Service */:
    case 203 /* Shuttle Coach Service */:
    case 204 /* Regional Coach Service */:
    case 205 /* Special Coach Service */:
    case 206 /* Sightseeing Coach Service */:
    case 207 /* Tourist Coach Service */:
    case 208 /* Commuter Coach Service */:
    case 209 /* All Coach Services */: return clasz::kCoach;
    case 400 /* Urban Railway Service */: return clasz::kSubway;
    case 401 /* Metro Service */: return clasz::kMetro;
    case 402 /* Underground Service */: return clasz::kSubway;
    case 403 /* Urban Railway Service */:
    case 404 /* All Urban Railway Services */:
    case 405 /* Monorail */: return clasz::kMetro;
    case 700 /* Bus Service */:
    case 701 /* Regional Bus Service */:
    case 702 /* Express Bus Service */:
    case 703 /* Stopping Bus Service */:
    case 704 /* Local Bus Service */:
    case 705 /* Night Bus Service */:
    case 706 /* Post Bus Service */:
    case 707 /* Special Needs Bus */:
    case 708 /* Mobility Bus Service */:
    case 709 /* Mobility Bus for Registered Disabled */:
    case 710 /* Sightseeing Bus */:
    case 711 /* Shuttle Bus */:
    case 712 /* School Bus */:
    case 713 /* School and Public Service Bus */:
    case 714 /* Rail Replacement Bus Service */:
    case 715 /* Demand and Response Bus Service */:
    case 716 /* All Bus Services */:
    case 800 /* Trolleybus Service */: return clasz::kBus;
    case 900 /* Tram Service */:
    case 901 /* City Tram Service */:
    case 902 /* Local Tram Service */:
    case 903 /* Regional Tram Service */:
    case 904 /* Sightseeing Tram Service */:
    case 905 /* Shuttle Tram Service */:
    case 906 /* All Tram Services */: return clasz::kTram;
    case 1000 /* Water Transport Service */: return clasz::kShip;
    case 1100 /* Air Service */: return clasz::kAir;
    case 1200 /* Ferry Service */: return clasz::kShip;
    case 1300 /* Aerial Lift Service */:
    case 1301 /* Telecabin Service */:
    case 1302 /* Cable Car Service */:
    case 1303 /* Elevator Service */:
    case 1304 /* Chair Lift Service */:
    case 1305 /* Drag Lift Service */:
    case 1306 /* Small Telecabin Service */:
    case 1307 /* All Telecabin Services */:
    case 1400 /* Funicular Service */:
    case 1500 /* Taxi Service */:
    case 1501 /* Communal Taxi Service */:
    case 1502 /* Water Taxi Service */:
    case 1503 /* Rail Taxi Service */:
    case 1504 /* Bike Taxi Service */:
    case 1505 /* Licensed Taxi Service */:
    case 1506 /* Private Hire Service Vehicle */:
    case 1507 /* All Taxi Services */:
    case 1700 /* Miscellaneous Service */:
    case 1702 /* Horse-drawn Carriage */:
    default: return clasz::kOther;
  }
}

route_map_t read_routes(timetable& tt,
                        tz_map& timezones,
                        agency_map_t& agencies,
                        std::string_view file_content,
                        std::string_view default_tz) {
  auto const timer = nigiri::scoped_timer{"read routes"};

  struct csv_route {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_short_name")> route_short_name_;
    utl::csv_col<utl::cstr, UTL_NAME("route_long_name")> route_long_name_;
    utl::csv_col<utl::cstr, UTL_NAME("route_desc")> route_desc_;
    utl::csv_col<int, UTL_NAME("route_type")> route_type_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Routes")
      .out_bounds(30.F, 32.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
         | utl::csv<csv_route>()  //
         | utl::transform([&](csv_route const& r) {
             auto const agency =
                 utl::get_or_create(agencies, r.agency_id_->view(), [&]() {
                   log(log_lvl::error, "gtfs.route",
                       "agency {} not found, using UNKNOWN with local timezone",
                       r.agency_id_->view());

                   auto const id = r.agency_id_->view().empty()
                                       ? "UKN"
                                       : r.agency_id_->view();
                   return tt.register_provider(
                       {id, "UNKNOWN_AGENCY",
                        get_tz_idx(tt, timezones, default_tz)});
                 });
             return std::pair{r.route_id_->to_str(),
                              std::make_unique<route>(route{
                                  .agency_ = agency,
                                  .id_ = r.route_id_->to_str(),
                                  .short_name_ = r.route_short_name_->to_str(),
                                  .long_name_ = r.route_long_name_->to_str(),
                                  .desc_ = r.route_desc_->to_str(),
                                  .clasz_ = to_clasz(*r.route_type_)})};
           })  //
         | utl::to<route_map_t>();
}

}  // namespace nigiri::loader::gtfs
