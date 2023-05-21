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
#include "nigiri/common/cached_lookup.h"
#include "nigiri/logging.h"
#include "utl/pipes/for_each.h"

namespace nigiri::loader::gtfs {

void read_stop_times(timetable& tt,
                     trip_data& trips,
                     locations_map const& stops,
                     std::string_view file_content) {
  struct csv_stop_time {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("arrival_time")> arrival_time_;
    utl::csv_col<utl::cstr, UTL_NAME("departure_time")> departure_time_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<std::uint16_t, UTL_NAME("stop_sequence")> stop_sequence_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_headsign")> stop_headsign_;
    utl::csv_col<int, UTL_NAME("pickup_type")> pickup_type_;
    utl::csv_col<int, UTL_NAME("drop_off_type")> drop_off_type_;
  };

  auto const timer = scoped_timer{"read stop times"};
  std::string last_trip_id;
  trip* last_trip = nullptr;
  auto i = 1U;
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Stop Times")
      .out_bounds(45.F, 70.F)
      .in_high(file_content.size());
  auto lookup_direction = cached_lookup(trips.directions_);
  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_stop_time>()  //
      |
      utl::for_each([&](csv_stop_time const& s) {
        ++i;

        trip* t = nullptr;
        auto const t_id = s.trip_id_->view();
        if (last_trip != nullptr && t_id == last_trip_id) {
          t = last_trip;
        } else {
          if (last_trip != nullptr) {
            last_trip->to_line_ = i - 1;
          }

          auto const trip_it = trips.trips_.find(t_id);
          if (trip_it == end(trips.trips_)) {
            log(log_lvl::error, "loader.gtfs.stop_time",
                "stop_times.txt:{} trip \"{}\" not found", i, t_id);
            return;
          }
          t = &trips.data_[trip_it->second];
          last_trip_id = t_id;
          last_trip = t;

          t->from_line_ = i;
        }

        try {
          auto const arrival_time = hhmm_to_min(*s.arrival_time_);
          auto const departure_time = hhmm_to_min(*s.departure_time_);

          t->requires_interpolation_ |= arrival_time == kInterpolate;
          t->requires_interpolation_ |= departure_time == kInterpolate;
          t->requires_sorting_ |= (!t->seq_numbers_.empty() &&
                                   t->seq_numbers_.back() > *s.stop_sequence_);

          t->seq_numbers_.emplace_back(*s.stop_sequence_);
          t->stop_seq_.push_back(stop{stops.at(s.stop_id_->view()),
                                      *s.pickup_type_ != 1,
                                      *s.drop_off_type_ != 1}
                                     .value());
          t->event_times_.emplace_back(
              stop_events{.arr_ = arrival_time, .dep_ = departure_time});

          if (!s.stop_headsign_->empty()) {
            t->stop_headsigns_.resize(t->seq_numbers_.size(),
                                      trip_direction_idx_t::invalid());
            t->stop_headsigns_.back() =
                lookup_direction(s.stop_headsign_->view(), [&]() {
                  return trips.get_or_create_direction(
                      tt, s.stop_headsign_->view());
                });
          }
        } catch (...) {
          log(log_lvl::error, "loader.gtfs.stop_time",
              "stop_times.txt:{}: unknown stop \"{}\"", i, s.stop_id_->view());
        }
      });

  if (last_trip != nullptr) {
    last_trip->to_line_ = i;
  }
}

}  // namespace nigiri::loader::gtfs
