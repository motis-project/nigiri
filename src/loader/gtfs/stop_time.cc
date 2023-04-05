#include "nigiri/loader/gtfs/stop_time.h"

#include <algorithm>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/logging.h"
#include "utl/pipes/for_each.h"

namespace nigiri::loader::gtfs {

void read_stop_times(trip_map& trips,
                     stop_map const& stops,
                     std::string_view file_content) {
  struct csv_stop_time {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("arrival_time")> arrival_time_;
    utl::csv_col<utl::cstr, UTL_NAME("departure_time")> departure_time_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<int, UTL_NAME("stop_sequence")> stop_sequence_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_headsign")> stop_headsign_;
    utl::csv_col<int, UTL_NAME("pickup_type")> pickup_type_;
    utl::csv_col<int, UTL_NAME("drop_off_type")> drop_off_type_;
  };

  scoped_timer timer{"read stop times"};
  std::string last_trip_id;
  trip* last_trip = nullptr;
  auto i = 0;
  return utl::line_range{utl::buf_reader{file_content}}  //
         | utl::csv<csv_stop_time>()  //
         | utl::for_each([&](csv_stop_time const& s) {
             ++i;

             trip* t = nullptr;
             auto const t_id = s.trip_id_->view();
             if (last_trip != nullptr && t_id == last_trip_id) {
               t = last_trip;
             } else {
               auto const trip_it = trips.find(t_id);
               if (trip_it == end(trips)) {
                 log(log_lvl::error, "loader.gtfs.stop_time",
                     "stop_times.txt:{} trip \"{}\" not found", i, t_id);
                 return;
               }
               t = trip_it->second.get();
               last_trip_id = t_id;
               last_trip = t;
             }

             try {
               t->stop_times_.emplace(
                   static_cast<std::size_t>(*s.stop_sequence_),
                   stops.at(s.stop_id_->view()).get(),
                   s.stop_headsign_->to_str(),  //
                   hhmm_to_min(*s.arrival_time_), *s.drop_off_type_ != 1,
                   hhmm_to_min(*s.departure_time_), *s.pickup_type_ != 1);
             } catch (...) {
               log(log_lvl::error, "loader.gtfs.stop_time",
                   "stop_times.txt:{}: unknown stop \"{}\"", i,
                   s.stop_id_->view());
             }
           });
}

}  // namespace nigiri::loader::gtfs
