#include "nigiri/loader/gtfs/trip.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

trip::trip(route const* route,
           bitfield const* service,
           std::basic_string<gtfs_trip_idx_t>* blk,
           std::string id,
           trip_direction_idx_t const headsign,
           std::string short_name)
    : route_(route),
      service_(service),
      block_{blk},
      id_{std::move(id)},
      headsign_(headsign),
      short_name_(std::move(short_name)) {}

void trip::interpolate() {
  if (!requires_interpolation_) {
    return;
  }

  struct bound {
    explicit bound(minutes_after_midnight_t t) : min_{t}, max_{t} {}
    minutes_after_midnight_t interpolate(int const idx) const {
      auto const p =
          static_cast<double>(idx - min_idx_) / (max_idx_ - min_idx_);
      return min_ + duration_t{static_cast<duration_t::rep>(
                        std::round((max_ - min_).count() * p))};
    }
    minutes_after_midnight_t min_, max_;
    int min_idx_{-1};
    int max_idx_{-1};
  };
  auto bounds = std::vector<bound>{};
  bounds.reserve(stop_seq_.size());
  for (auto const [i, x] : utl::enumerate(event_times_)) {
    bounds.emplace_back(x.arr_);
    bounds.emplace_back(x.dep_);
  }

  auto max = duration_t{0};
  auto max_idx = 0;
  for (auto it = bounds.rbegin(); it != bounds.rend(); ++it) {
    if (it->max_ == kInterpolate) {
      it->max_ = max;
      it->max_idx_ = max_idx;
    } else {
      max = it->max_;
      max_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }
  utl::verify(max != kInterpolate, "last arrival cannot be interpolated");

  auto min = duration_t{0};
  auto min_idx = 0;
  for (auto it = bounds.begin(); it != bounds.end(); ++it) {
    if (it->min_ == kInterpolate) {
      it->min_ = min;
      it->min_idx_ = min_idx;
    } else {
      min = it->max_;
      min_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }
  utl::verify(min != kInterpolate, "first arrival cannot be interpolated");

  for (auto const [idx, entry] : utl::enumerate(event_times_)) {
    auto const& arr = bounds[2 * idx];
    auto const& dep = bounds[2 * idx + 1];

    if (entry.arr_ == kInterpolate) {
      entry.arr_ = arr.interpolate(static_cast<int>(idx));
    }
    if (entry.dep_ == kInterpolate) {
      entry.dep_ = dep.interpolate(static_cast<int>(idx));
    }
  }
}

std::string trip::display_name(timetable const& tt) const {
  auto const is_digit = [](char const x) { return x >= '0' && x <= '9'; };
  if (route_->clasz_ == clasz::kBus) {
    return route_->short_name_.empty() ? "Bus " + short_name_
                                       : "Bus " + route_->short_name_;
  } else if (route_->clasz_ == clasz::kTram) {
    return route_->short_name_.empty() ? "Tram " + short_name_
                                       : "Tram " + route_->short_name_;
  }

  auto const trip_name_is_number = utl::all_of(short_name_, is_digit);
  if (route_->agency_ != provider_idx_t::invalid() &&
      tt.providers_[route_->agency_].long_name_ == "DB Fernverkehr AG") {
    if (route_->clasz_ == clasz::kHighSpeed) {
      return trip_name_is_number
                 ? fmt::format("ICE {}", utl::parse<int>(short_name_))
                 : fmt::format("ICE {}", short_name_);
    } else if (route_->clasz_ == clasz::kLongDistance) {
      return trip_name_is_number
                 ? fmt::format("IC {}", utl::parse<int>(short_name_))
                 : fmt::format("IC {}", short_name_);
    }
  }

  return route_->short_name_ + " " + short_name_;
}

trip_direction_idx_t trip_data::get_or_create_direction(
    timetable& tt, std::string_view headsign) {
  return utl::get_or_create(directions_, headsign, [&]() {
    auto const trip_dir_str = tt.register_trip_direction_string(headsign);
    auto const idx = tt.trip_directions_.size();
    tt.trip_directions_.emplace_back(trip_dir_str);
    return trip_direction_idx_t{idx};
  });
}

trip_data read_trips(timetable& tt,
                     route_map_t const& routes,
                     traffic_days const& services,
                     std::string_view file_content) {
  struct csv_trip {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> service_id_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_headsign")> trip_headsign_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_short_name")> trip_short_name_;
    utl::csv_col<utl::cstr, UTL_NAME("block_id")> block_id_;
  };

  nigiri::scoped_timer const timer{"read trips"};

  trip_data ret;

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Trips")
      .out_bounds(40.F, 44.F)
      .in_high(file_content.size());
  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_trip>()  //
      |
      utl::for_each([&](csv_trip const& t) {
        auto const traffic_days_it =
            services.traffic_days_.find(t.service_id_->view());
        if (traffic_days_it == end(services.traffic_days_)) {
          log(log_lvl::error, "loader.gtfs.trip",
              R"(trip "{}": service_id "{}" not found)", t.trip_id_->view(),
              t.service_id_->view());
          return;
        }

        auto const blk = t.block_id_->trim().empty()
                             ? nullptr
                             : utl::get_or_create(
                                   ret.blocks_, t.block_id_->trim().view(),
                                   []() {
                                     return std::make_unique<
                                         std::basic_string<gtfs_trip_idx_t>>();
                                   })
                                   .get();
        auto const trp_idx = gtfs_trip_idx_t{ret.data_.size()};
        ret.data_.emplace_back(
            routes.at(t.route_id_->view()).get(), traffic_days_it->second.get(),
            blk, t.trip_id_->to_str(),
            ret.get_or_create_direction(tt, t.trip_headsign_->view()),
            t.trip_short_name_->to_str());
        ret.trips_.emplace(t.trip_id_->to_str(), trp_idx);
        if (blk != nullptr) {
          blk->push_back(trp_idx);
        }
      });
  return ret;
}

void read_frequencies(trip_data& trips, std::string_view file_content) {
  if (file_content.empty()) {
    return;
  }

  struct csv_frequency {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("start_time")> start_time_;
    utl::csv_col<utl::cstr, UTL_NAME("end_time")> end_time_;
    utl::csv_col<utl::cstr, UTL_NAME("headway_secs")> headway_secs_;
    utl::csv_col<utl::cstr, UTL_NAME("exact_times")> exact_times_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Frequencies")
      .out_bounds(44.F, 45.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
         | utl::csv<csv_frequency>()  //
         |
         utl::for_each([&](csv_frequency const& freq) {
           auto const t = freq.trip_id_->trim().view();
           auto const trip_it = trips.trips_.find(t);
           if (trip_it == end(trips.trips_)) {
             log(log_lvl::error, "loader.gtfs.frequencies",
                 "frequencies.txt:{}: skipping frequency (trip \"{}\" not "
                 "found)",
                 t);
             return;
           }

           auto const headway_secs_str = *freq.headway_secs_;
           auto const headway_secs = parse<int>(headway_secs_str, -1);
           if (headway_secs == -1) {
             log(log_lvl::error, "loader.gtfs.frequencies",
                 "frequencies.txt:{}: skipping frequency (invalid headway secs "
                 "\"{}\")",
                 headway_secs_str.view());
             return;
           }

           auto const exact = freq.exact_times_->view();
           auto const schedule_relationship =
               exact == "1" ? frequency::schedule_relationship::kScheduled
                            : frequency::schedule_relationship::kUnscheduled;

           auto& frequencies = trips.data_[trip_it->second].frequency_;
           if (!frequencies.has_value()) {
             frequencies = std::vector<frequency>{};
           }
           frequencies->emplace_back(
               frequency{hhmm_to_min(freq.start_time_->view()),
                         hhmm_to_min(freq.end_time_->view()),
                         duration_t{headway_secs / 60}, schedule_relationship});
         });
}

}  // namespace nigiri::loader::gtfs
