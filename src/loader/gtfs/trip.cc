#include "nigiri/loader/gtfs/trip.h"

#include <algorithm>
#include <numeric>
#include <stack>

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

std::vector<std::pair<basic_string<gtfs_trip_idx_t>, bitfield>>
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
    return {{std::pair{basic_string<gtfs_trip_idx_t>{trips_.front()},
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
  auto rule_trips = utl::to_vec(
      trips_, [&](auto&& t) { return rule_trip{t, *trips.get(t).service_}; });

  struct queue_entry {
    std::vector<rule_trip>::iterator current_it_;
    std::vector<std::vector<rule_trip>::iterator> collected_trips_;
    bitfield traffic_days_;
  };

  std::vector<std::pair<basic_string<gtfs_trip_idx_t>, bitfield>> combinations;
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
            utl::transform_to<basic_string<gtfs_trip_idx_t>>(
                collected_trips, [](auto&& rt) { return rt->trip_; }),
            traffic_days);
      }
    }
  }

  return combinations;
}

trip::trip(route_id_idx_t route,
           bitfield const* service,
           block* blk,
           std::string id,
           translation_idx_t const headsign,
           translation_idx_t const short_name,
           direction_id_t const direction_id,
           shape_idx_t const shape_idx,
           bool const bikes_allowed,
           bool const cars_allowed)
    : route_{route},
      service_{service},
      block_{blk},
      id_{std::move(id)},
      headsign_{headsign},
      direction_id_{direction_id},
      short_name_{std::move(short_name)},
      shape_idx_{shape_idx},
      bikes_allowed_{bikes_allowed},
      cars_allowed_{cars_allowed} {}

interpolate_result interpolate(std::vector<stop_events>& event_times) {
  struct bound {
    explicit bound(minutes_after_midnight_t t) : min_{t}, max_{t} {}
    minutes_after_midnight_t interpolate(int const idx) const {
      auto const denom = max_idx_ - min_idx_;
      auto const p =
          denom > 0 ? static_cast<double>(idx - min_idx_) / denom : 0;
      return min_ + duration_t{static_cast<duration_t::rep>(
                        std::round((max_ - min_).count() * p))};
    }
    minutes_after_midnight_t min_, max_;
    int min_idx_{-1};
    int max_idx_{-1};
  };

  auto bounds = std::vector<bound>{};
  bounds.reserve(event_times.size() * 2U);
  for (auto const [i, x] : utl::enumerate(event_times)) {
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
  if (bounds.size() <= 1 || bounds[bounds.size() - 2].max_idx_ == 0) {
    return interpolate_result::kErrorLastMissing;
  }

  auto min = duration_t{0};
  auto const last = static_cast<int>(event_times.size() - 1);
  auto min_idx = last;
  for (auto it = bounds.begin(); it != bounds.end(); ++it) {
    if (it->min_ == kInterpolate) {
      it->min_ = min;
      it->min_idx_ = min_idx;
    } else {
      min = it->max_;
      min_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }
  if (bounds[1].min_idx_ == last) {
    return interpolate_result::kErrorFirstMissing;
  }

  for (auto const [idx, entry] : utl::enumerate(event_times)) {
    auto const& arr = bounds[2 * idx];
    auto const& dep = bounds[2 * idx + 1];

    if (entry.arr_ == kInterpolate) {
      entry.arr_ = arr.interpolate(static_cast<int>(idx));
    }
    if (entry.dep_ == kInterpolate) {
      entry.dep_ = dep.interpolate(static_cast<int>(idx));
    }
  }

  return interpolate_result::kOk;
}

bool trip::has_seated_transfers() const {
  return !seated_in_.empty() || !seated_out_.empty();
}

trip_data read_trips(source_idx_t const src,
                     source_file_idx_t const source_file,
                     timetable& tt,
                     translator& i18n,
                     route_map_t const& routes,
                     traffic_days_t const& services,
                     shape_loader_state const& shape_states,
                     std::string_view file_content,
                     std::array<bool, kNumClasses> const& bikes_allowed_default,
                     std::array<bool, kNumClasses> const& cars_allowed_default,
                     script_runner const& user_script) {
  struct csv_trip {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> service_id_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<generic_string, UTL_NAME("trip_headsign")> trip_headsign_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_short_name")> trip_short_name_;
    utl::csv_col<utl::cstr, UTL_NAME("direction_id")> direction_id_;
    utl::csv_col<utl::cstr, UTL_NAME("block_id")> block_id_;
    utl::csv_col<utl::cstr, UTL_NAME("shape_id")> shape_id_;
    utl::csv_col<std::uint8_t, UTL_NAME("bikes_allowed")> bikes_allowed_;
    utl::csv_col<std::uint8_t, UTL_NAME("cars_allowed")> cars_allowed_;
  };
  auto const& shapes = shape_states.id_map_;

  nigiri::scoped_timer const timer{"read trips"};

  trip_data ret;

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Trips")
      .out_bounds(38.F, 42.F)
      .in_high(file_content.size());
  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_trip>()  //
      |
      utl::for_each([&](csv_trip const& t) {
        auto const traffic_days_it = services.find(t.service_id_->view());
        if (traffic_days_it == end(services)) {
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

        auto const shape_it = shapes.find(t.shape_id_->view());
        auto const shape_idx = (shape_it == end(shapes))
                                   ? shape_idx_t::invalid()
                                   : shape_it->second;

        auto const route_id = route_it->second->route_id_idx_;
        auto const clasz = static_cast<std::size_t>(
            to_clasz(tt.route_ids_[src].route_id_type_[route_id]));

        auto bikes_allowed = bikes_allowed_default[clasz];
        if (t.bikes_allowed_.val() == 1) {
          bikes_allowed = true;
        } else if (t.bikes_allowed_.val() == 2) {
          bikes_allowed = false;
        }

        auto cars_allowed = cars_allowed_default[clasz];
        if (t.cars_allowed_.val() == 1) {
          cars_allowed = true;
        } else if (t.cars_allowed_.val() == 2) {
          cars_allowed = false;
        }

        auto const id = t.trip_id_->view();
        auto const trip_short_name = i18n.get(t::kTrips, f::kTripShortName,
                                              t.trip_short_name_->view(), id);
        auto const display_name = [&]() {
          for (auto const str : {
                   tt.route_ids_[src].route_id_short_names_[route_id],
                   trip_short_name,
                   tt.route_ids_[src].route_id_long_names_[route_id],
               }) {
            if (str != kEmptyTranslation) {
              return str;
            }
          }
          return kEmptyTranslation;
        }();

        auto x = loader::trip{
            tt,
            src,
            id,
            i18n.get(t::kTrips, f::kTripHeadsign, t.trip_headsign_->view(), id),
            trip_short_name,
            display_name,
            "",
            "",
            (t.direction_id_->view() == "1") ? direction_id_t{1U}
                                             : direction_id_t{0U},
            route_id,
            trip_debug{.source_file_idx_ = source_file}};

        auto const keep = process_trip(user_script, x);
        if (!keep) {
          log(log_lvl::info, "nigiri.import.gtfs.trip",
              "script removed trip {}", t.trip_id_->view());
          return;
        }

        auto const blk =
            t.block_id_->trim().empty()
                ? nullptr
                : utl::get_or_create(ret.blocks_, t.block_id_->trim().view(),
                                     []() { return std::make_unique<block>(); })
                      .get();

        auto const gtfs_trp_idx = gtfs_trip_idx_t{ret.data_.size()};
        ret.data_.push_back(trip{route_id, traffic_days_it->second.get(), blk,
                                 t.trip_id_->to_str(), x.headsign_,
                                 trip_short_name, x.direction_, shape_idx,
                                 bikes_allowed, cars_allowed});
        ret.data_.back().trip_idx_ = register_trip(tt, x);
        ret.trips_.emplace(t.trip_id_->to_str(), gtfs_trp_idx);
        if (blk != nullptr) {
          blk->trips_.emplace_back(gtfs_trp_idx);
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
      .out_bounds(42.F, 43.F)
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
                 "frequencies.txt: skipping frequency (trip \"{}\" not "
                 "found)",
                 t);
             return;
           }

           auto const headway_secs_str = *freq.headway_secs_;
           auto const headway_secs = parse<int>(headway_secs_str, -1);
           if (headway_secs == -1) {
             log(log_lvl::error, "loader.gtfs.frequencies",
                 R"(frequencies.txt: skipping frequency (invalid headway secs "{}"))",
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

           // If the service operates multiple times per minute, make sure
           // not to end up with zero.
           auto const headway_minutes = duration_t{std::max(
               static_cast<int>(
                   std::round(static_cast<float>(headway_secs) / 60.F)),
               1)};
           frequencies->emplace_back(
               frequency{hhmm_to_min(freq.start_time_->view()),
                         hhmm_to_min(freq.end_time_->view()), headway_minutes,
                         schedule_relationship});
         });
}

}  // namespace nigiri::loader::gtfs
