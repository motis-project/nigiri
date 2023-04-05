#include "nigiri/loader/gtfs/route.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "nigiri/logging.h"

using namespace utl;
using std::get;

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
      return clasz::kOther;
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
    case 106 /* Regional Rail Service */: return clasz::kRegional;
    case 107 /* Tourist Railway Service */: return clasz::kRegional;
    case 108 /* Rail Shuttle (Within Complex) */: return clasz::kRegional;
    case 109 /* Suburban Railway */: return clasz::kMetro;
    case 110 /* Replacement Rail Service */: return clasz::kRegional;
    case 111 /* Special Rail Service */: return clasz::kRegional;
    case 112 /* Lorry Transport Rail Service */: return clasz::kRegional;
    case 113 /* All Rail Services */: return clasz::kRegional;
    case 114 /* Cross-Country Rail Service */: return clasz::kLongDistance;
    case 115 /* Vehicle Transport Rail Service */: return clasz::kRegional;
    case 116 /* Rack and Pinion Railway */: return clasz::kRegional;
    case 117 /* Additional Rail Service */: return clasz::kRegional;
    case 200 /* Coach Service */: return clasz::kCoach;
    case 201 /* International Coach Service */: return clasz::kCoach;
    case 202 /* National Coach Service */: return clasz::kCoach;
    case 203 /* Shuttle Coach Service */: return clasz::kCoach;
    case 204 /* Regional Coach Service */: return clasz::kCoach;
    case 205 /* Special Coach Service */: return clasz::kCoach;
    case 206 /* Sightseeing Coach Service */: return clasz::kCoach;
    case 207 /* Tourist Coach Service */: return clasz::kCoach;
    case 208 /* Commuter Coach Service */: return clasz::kCoach;
    case 209 /* All Coach Services */: return clasz::kCoach;
    case 400 /* Urban Railway Service */: return clasz::kMetro;
    case 401 /* Metro Service */: return clasz::kMetro;
    case 402 /* Underground Service */: return clasz::kSubway;
    case 403 /* Urban Railway Service */: return clasz::kMetro;
    case 404 /* All Urban Railway Services */: return clasz::kMetro;
    case 405 /* Monorail */: return clasz::kMetro;
    case 700 /* Bus Service */: return clasz::kBus;
    case 701 /* Regional Bus Service */: return clasz::kBus;
    case 702 /* Express Bus Service */: return clasz::kBus;
    case 703 /* Stopping Bus Service */: return clasz::kBus;
    case 704 /* Local Bus Service */: return clasz::kBus;
    case 705 /* Night Bus Service */: return clasz::kBus;
    case 706 /* Post Bus Service */: return clasz::kBus;
    case 707 /* Special Needs Bus */: return clasz::kBus;
    case 708 /* Mobility Bus Service */: return clasz::kBus;
    case 709 /* Mobility Bus for Registered Disabled */: return clasz::kBus;
    case 710 /* Sightseeing Bus */: return clasz::kBus;
    case 711 /* Shuttle Bus */: return clasz::kBus;
    case 712 /* School Bus */: return clasz::kBus;
    case 713 /* School and Public Service Bus */: return clasz::kBus;
    case 714 /* Rail Replacement Bus Service */: return clasz::kBus;
    case 715 /* Demand and Response Bus Service */: return clasz::kBus;
    case 716 /* All Bus Services */: return clasz::kBus;
    case 800 /* Trolleybus Service */: return clasz::kBus;
    case 900 /* Tram Service */: return clasz::kTram;
    case 901 /* City Tram Service */: return clasz::kTram;
    case 902 /* Local Tram Service */: return clasz::kTram;
    case 903 /* Regional Tram Service */: return clasz::kTram;
    case 904 /* Sightseeing Tram Service */: return clasz::kTram;
    case 905 /* Shuttle Tram Service */: return clasz::kTram;
    case 906 /* All Tram Services */: return clasz::kTram;
    case 1000 /* Water Transport Service */: return clasz::kShip;
    case 1100 /* Air Service */: return clasz::kAir;
    case 1200 /* Ferry Service */: return clasz::kShip;
    case 1300 /* Aerial Lift Service */: return clasz::kAir;
    case 1301 /* Telecabin Service */: return clasz::kOther;
    case 1302 /* Cable Car Service */: return clasz::kOther;
    case 1303 /* Elevator Service */: return clasz::kOther;
    case 1304 /* Chair Lift Service */: return clasz::kOther;
    case 1305 /* Drag Lift Service */: return clasz::kOther;
    case 1306 /* Small Telecabin Service */: return clasz::kOther;
    case 1307 /* All Telecabin Services */: return clasz::kOther;
    case 1400 /* Funicular Service */: return clasz::kOther;
    case 1500 /* Taxi Service */: return clasz::kOther;
    case 1501 /* Communal Taxi Service */: return clasz::kOther;
    case 1502 /* Water Taxi Service */: return clasz::kOther;
    case 1503 /* Rail Taxi Service */: return clasz::kOther;
    case 1504 /* Bike Taxi Service */: return clasz::kOther;
    case 1505 /* Licensed Taxi Service */: return clasz::kOther;
    case 1506 /* Private Hire Service Vehicle */: return clasz::kOther;
    case 1507 /* All Taxi Services */: return clasz::kOther;
    case 1700 /* Miscellaneous Service */: return clasz::kOther;
    case 1702 /* Horse-drawn Carriage */: return clasz::kOther;
    default: return clasz::kOther;
  }
}

route_map_t read_routes(agency_map_t const& agencies,
                        std::string_view file_content) {
  scoped_timer timer{"read routes"};

  struct csv_route {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_short_name")> route_short_name_;
    utl::csv_col<utl::cstr, UTL_NAME("route_long_name")> route_long_name_;
    utl::csv_col<utl::cstr, UTL_NAME("route_desc")> route_desc_;
    utl::csv_col<int, UTL_NAME("route_type")> route_type_;
  };

  return utl::line_range{utl::buf_reader{file_content}}  //
         | utl::csv<csv_route>()  //
         |
         utl::transform([&](csv_route const& r) {
           auto const agency_it = agencies.find(r.agency_id_->view());
           if (agency_it == end(agencies)) {
             log(log_lvl::error, "gtfs.route", "agency {} not found",
                 r.agency_id_->view());
           }
           return cista::pair{r.route_id_->to_str(),
                              std::make_unique<route>(route{
                                  .agency_ = (agency_it == end(agencies)
                                                  ? provider_idx_t::invalid()
                                                  : agency_it->second),
                                  .id_ = r.route_id_->to_str(),
                                  .short_name_ = r.route_short_name_->to_str(),
                                  .long_name_ = r.route_long_name_->to_str(),
                                  .desc_ = r.route_desc_->to_str(),
                                  .clasz_ = to_clasz(*r.route_type_)})};
         })  //
         | utl::to<route_map_t>();
}

}  // namespace nigiri::loader::gtfs
