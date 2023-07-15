#include "nigiri/loader/gtfs/trip.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <tuple>

#include "geo/box.h"

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

std::vector<std::pair<std::basic_string<gtfs_trip_idx_t>, bitfield>>
block::rule_services(trip_data& trips) {
  utl::verify(!trips_.empty(), "empty block not allowed");

  utl::erase_if(trips_, [&](gtfs_trip_idx_t const& t) {
    auto const is_empty = trips.data_[t].stop_seq_.empty();
    if (is_empty) {
      log(log_lvl::error, "loader.gtfs.trip", "trip \"{}\": no stop times",
          trips.data_[t].id_);
    }
    return is_empty;
  });

  if (trips_.size() == 1) {
    return {{std::pair{std::basic_string<gtfs_trip_idx_t>{trips_.front()},
                       *trips.get(trips_.front()).service_}}};
  }

  std::sort(begin(trips_), end(trips_),
            [&](gtfs_trip_idx_t const a_idx, gtfs_trip_idx_t const b_idx) {
              auto const& a = trips.get(a_idx);
              auto const& b = trips.get(b_idx);
              return a.event_times_.front().dep_ < b.event_times_.front().dep_;
            });

  struct rule_trip {
    gtfs_trip_idx_t trip_;
    bitfield traffic_days_;
  };
  auto rule_trips = utl::to_vec(trips_, [&](auto&& t) {
    return rule_trip{t, *trips.get(t).service_};
  });

  struct queue_entry {
    std::vector<rule_trip>::iterator current_it_;
    std::vector<std::vector<rule_trip>::iterator> collected_trips_;
    bitfield traffic_days_;
  };

  std::vector<std::pair<std::basic_string<gtfs_trip_idx_t>, bitfield>>
      combinations;
  for (auto start_it = begin(rule_trips); start_it != end(rule_trips);
       ++start_it) {
    std::stack<queue_entry> q;
    q.emplace(queue_entry{start_it, {}, start_it->traffic_days_});
    while (!q.empty()) {
      auto next = q.top();
      q.pop();

      auto& [current_it, collected_trips, traffic_days] = next;
      collected_trips.emplace_back(current_it);
      for (auto succ_it = std::next(current_it); succ_it != end(rule_trips);
           ++succ_it) {
        auto const& curr_trip = trips.data_[current_it->trip_];
        auto const& succ_trip = trips.data_[succ_it->trip_];
        if (stop{curr_trip.stop_seq_.back()}.location_ !=
            stop{succ_trip.stop_seq_.front()}.location_) {
          continue;  // prev last stop != next first stop
        }

        auto const new_intersection = traffic_days & succ_it->traffic_days_;
        traffic_days &= ~succ_it->traffic_days_;
        if (new_intersection.any()) {
          q.emplace(queue_entry{succ_it, collected_trips, new_intersection});
        }
      }

      if (traffic_days.any()) {
        for (auto& rt : collected_trips) {
          rt->traffic_days_ &= ~traffic_days;
        }

        combinations.emplace_back(
            utl::transform_to<std::basic_string<gtfs_trip_idx_t>>(
                collected_trips, [](auto&& rt) { return rt->trip_; }),
            traffic_days);
      }
    }
  }

  return combinations;
}

trip::trip(route const* route,
           bitfield const* service,
           block* blk,
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

  auto const starts_with_letter_and_ends_with_number =
      [](std::string_view line_id) {
        return !(line_id.front() >= '0' && line_id.front() <= '9') &&
               (line_id.back() >= '0' && line_id.back() <= '9');
      };

  if (starts_with_letter_and_ends_with_number(route_->short_name_)) {
    return route_->short_name_;
  } else if (trip_name_is_number) {
    return fmt::format("{} {}", route_->short_name_,
                       utl::parse<int>(short_name_));
  } else {
    return fmt::format("{} {}", route_->short_name_, short_name_);
  }
}

clasz trip::get_clasz(timetable const& tt) const {
  if (route_->clasz_ != clasz::kBus) {
    return route_->clasz_;
  } else {
    geo::box box;
    for (auto const& stp : stop_seq_) {
      box.extend(tt.locations_.coordinates_[stop{stp}.location_idx()]);
    }
    return (geo::distance(box.min_, box.max_) / 1000) > 100 ? clasz::kCoach
                                                            : clasz::kBus;
  }
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
      | utl::for_each([&](csv_trip const& t) {
          auto const traffic_days_it =
              services.traffic_days_.find(t.service_id_->view());
          if (traffic_days_it == end(services.traffic_days_)) {
            log(log_lvl::error, "loader.gtfs.trip",
                R"(trip "{}": service_id "{}" not found)", t.trip_id_->view(),
                t.service_id_->view());
            return;
          }

          auto const route_it = routes.find(t.route_id_->view());
          if (route_it == end(routes)) {
            log(log_lvl::error, "loader.gtfs.trip",
                R"(trip "{}": route_id "{}" not found)", t.trip_id_->view(),
                t.route_id_->view());
            return;
          }

          auto const blk = t.block_id_->trim().empty()
                               ? nullptr
                               : utl::get_or_create(
                                     ret.blocks_, t.block_id_->trim().view(),
                                     []() { return std::make_unique<block>(); })
                                     .get();
          auto const trp_idx = gtfs_trip_idx_t{ret.data_.size()};
          ret.data_.emplace_back(
              route_it->second.get(), traffic_days_it->second.get(), blk,
              t.trip_id_->to_str(),
              ret.get_or_create_direction(tt, t.trip_headsign_->view()),
              t.trip_short_name_->to_str());
          ret.trips_.emplace(t.trip_id_->to_str(), trp_idx);
          if (blk != nullptr) {
            blk->trips_.emplace_back(trp_idx);
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
