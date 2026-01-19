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

void add_distance(auto& trip_data, double const distance) {
  auto& distances = trip_data.distance_traveled_;
  if (distances.empty()) {
    if (distance != 0.0) {
      distances.resize(trip_data.seq_numbers_.size());
      distances.back() = distance;
    }
  } else {
    distances.emplace_back(distance);
  }
}

void read_stop_times(trip_data& trips,
                     stops_map_t const& stops,
                     flex_areas_t const& flex_areas,
                     booking_rules_t const& booking_rules,
                     location_groups_t const& location_groups,
                     translator& i18n,
                     std::string_view file_content,
                     bool const store_distances) {
  struct csv_stop_time {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("arrival_time")> arrival_time_;
    utl::csv_col<utl::cstr, UTL_NAME("departure_time")> departure_time_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_sequence")> stop_sequence_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_headsign")> stop_headsign_;
    utl::csv_col<int, UTL_NAME("pickup_type")> pickup_type_;
    utl::csv_col<int, UTL_NAME("drop_off_type")> drop_off_type_;
    utl::csv_col<double, UTL_NAME("shape_dist_traveled")> distance_;

    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_id")> location_id_;
    utl::csv_col<utl::cstr, UTL_NAME("start_pickup_drop_off_window")>
        start_pickup_drop_off_window_;
    utl::csv_col<utl::cstr, UTL_NAME("end_pickup_drop_off_window")>
        end_pickup_drop_off_window_;
    utl::csv_col<utl::cstr, UTL_NAME("pickup_booking_rule_id")>
        pickup_booking_rule_id_;
    utl::csv_col<utl::cstr, UTL_NAME("drop_off_booking_rule_id")>
        drop_off_booking_rule_id_;
  };

  auto line_number = 1U;

  // Parse GTFS Flex trip.
  auto const parse_flex_trip = [&](csv_stop_time const& s, trip* t,
                                   location_group_idx_t const l_group,
                                   flex_area_idx_t const flex_area) {
    auto pickup_booking = booking_rule_idx_t::invalid();
    if (!s.pickup_booking_rule_id_->view().empty()) {
      auto const it = booking_rules.find(s.pickup_booking_rule_id_->view());
      if (it == end(booking_rules)) {
        log(log_lvl::error, "loader.gtfs.stop_time",
            "stop_times.txt:{} booking rule {} not found", line_number,
            s.pickup_booking_rule_id_->view());
      } else {
        pickup_booking = it->second;
      }
    }

    auto drop_off_booking = booking_rule_idx_t::invalid();
    if (!s.drop_off_booking_rule_id_->view().empty()) {
      auto const it = booking_rules.find(s.drop_off_booking_rule_id_->view());
      if (it == end(booking_rules)) {
        log(log_lvl::error, "loader.gtfs.stop_time",
            "stop_times.txt:{} booking rule {} not found", line_number,
            s.drop_off_booking_rule_id_->view());
      } else {
        drop_off_booking = it->second;
      }
    }

    t->flex_stops_.push_back(l_group == location_group_idx_t::invalid()
                                 ? flex_stop_t{flex_area}
                                 : flex_stop_t{l_group});
    t->flex_time_windows_.push_back(stop_time_window{
        .pickup_booking_rule_ = pickup_booking,
        .drop_off_booking_rule_ = drop_off_booking,
        .start_ = hhmm_to_min(*s.start_pickup_drop_off_window_),
        .end_ = hhmm_to_min(*s.end_pickup_drop_off_window_)});
  };

  // Parse regular trip.
  auto const parse_regular_trip = [&](csv_stop_time const& s, trip* t,
                                      location_idx_t const l) {
    auto const arrival_time = hhmm_to_min(*s.arrival_time_);
    auto const departure_time = hhmm_to_min(*s.departure_time_);
    auto const in_allowed = *s.pickup_type_ != 1;
    auto const out_allowed = *s.drop_off_type_ != 1;
    t->stop_seq_.push_back(
        stop{l, in_allowed, out_allowed, in_allowed, out_allowed}.value());
    t->requires_interpolation_ |= arrival_time == kInterpolate;
    t->requires_interpolation_ |= departure_time == kInterpolate;
    t->event_times_.push_back({.arr_ = arrival_time, .dep_ = departure_time});
    if (store_distances) {
      add_distance(*t, *s.distance_);
    }
  };

  auto last_trip = static_cast<trip*>(nullptr);
  auto last_trip_id = std::string{};

  auto const timer = scoped_timer{"read stop times"};
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Stop Times")
      .out_bounds(43.F, 68.F)
      .in_high(file_content.size());
  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_stop_time>()  //
      |
      utl::for_each([&](csv_stop_time const& s) {
        ++line_number;

        // Lazy trip lookup: take previous if trip_id matches.
        trip* t = nullptr;
        auto const t_id = s.trip_id_->view();
        if (last_trip != nullptr && t_id == last_trip_id) {
          t = last_trip;
        } else {
          if (last_trip != nullptr) {
            last_trip->to_line_ = line_number - 1;
          }

          auto const trip_it = trips.trips_.find(t_id);
          if (trip_it == end(trips.trips_)) {
            log(log_lvl::error, "loader.gtfs.stop_time",
                "stop_times.txt:{} trip \"{}\" not found", line_number, t_id);
            return;
          }
          t = &trips.data_[trip_it->second];
          last_trip_id = t_id;
          last_trip = t;

          t->from_line_ = line_number;
        }

        // Lookup stop/location_group/location_id and skip if not found.
        auto l = location_idx_t::invalid();
        auto l_group = location_group_idx_t::invalid();
        auto flex_area = flex_area_idx_t::invalid();
        if (!s.stop_id_->view().empty()) {
          auto const it = stops.find(s.stop_id_->view());
          if (it == end(stops)) {
            log(log_lvl::error, "loader.gtfs.stop_time",
                "stop_times.txt:{}: unknown stop \"{}\"", line_number,
                s.stop_id_->view());
            return;
          }
          l = it->second;
        } else if (!s.location_group_id_->view().empty()) {
          auto const it = location_groups.find(s.location_group_id_->view());
          if (it == end(location_groups)) {
            log(log_lvl::error, "loader.gtfs.stop_time",
                "stop_times.txt:{}: unknown location group \"{}\"", line_number,
                s.location_group_id_->view());
            return;
          }
          l_group = it->second;
        } else if (!s.location_id_->view().empty()) {
          auto const it = flex_areas.find(s.location_id_->view());
          if (it == end(flex_areas)) {
            log(log_lvl::error, "loader.gtfs.stop_time",
                "stop_times.txt:{}: unknown flex area with location_id \"{}\"",
                line_number, s.location_group_id_->view());
            return;
          }
          flex_area = it->second;
        } else {
          log(log_lvl::error, "loader.gtfs.stop_time",
              "stop_times.txt:{}: no stop_id, location_group, or location_id",
              line_number);
          return;
        }

        // Store common attributes of regular trips and flex trips.
        auto const seq = utl::parse<std::uint16_t>(*s.stop_sequence_);
        t->requires_sorting_ |=
            (!t->seq_numbers_.empty() && t->seq_numbers_.back() > seq);
        t->seq_numbers_.push_back(seq);
        if (!s.stop_headsign_->empty()) {
          t->stop_headsigns_.resize(t->seq_numbers_.size(), t->headsign_);
          t->stop_headsigns_.back() = i18n.get(
              t::kStopTimes, f::kStopHeadsign, s.stop_headsign_->view(),
              s.trip_id_->view(), s.stop_sequence_->view());
        }

        if (l == location_idx_t::invalid()) {
          parse_flex_trip(s, t, l_group, flex_area);
        } else {
          parse_regular_trip(s, t, l);
        }
      });

  if (last_trip != nullptr) {
    last_trip->to_line_ = line_number;
  }
}

}  // namespace nigiri::loader::gtfs
