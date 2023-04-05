#include "nigiri/loader/gtfs/trip.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

std::vector<std::pair<std::vector<trip*>, bitfield>> block::rule_services() {
  utl::verify(!trips_.empty(), "empty block not allowed");

  utl::erase_if(trips_, [](trip const* t) {
    auto const is_empty = t->stop_times_.empty();
    if (is_empty) {
      log(log_lvl::error, "loader.gtfs.trip", "trip \"{}\": no stop times",
          t->id_);
    }
    return is_empty;
  });

  std::sort(begin(trips_), end(trips_), [](trip const* a, trip const* b) {
    return a->stop_times_.front().dep_.time_ <
           b->stop_times_.front().dep_.time_;
  });

  struct rule_trip {
    trip* trip_;
    bitfield traffic_days_;
  };
  auto rule_trips = utl::to_vec(trips_, [](auto&& t) {
    return rule_trip{t, *t->service_};
  });

  struct queue_entry {
    std::vector<rule_trip>::iterator current_it_;
    std::vector<std::vector<rule_trip>::iterator> collected_trips_;
    bitfield traffic_days_;
  };

  std::vector<std::pair<std::vector<trip*>, bitfield>> combinations;
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
        if (current_it->trip_->stop_times_.back().stop_ !=
            succ_it->trip_->stop_times_.front().stop_) {
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
            utl::to_vec(collected_trips, [](auto&& rt) { return rt->trip_; }),
            traffic_days);
      }
    }
  }

  return combinations;
}

stop_time::stop_time() = default;

stop_time::stop_time(stop* s,
                     std::string headsign,
                     int arr_time,
                     bool out_allowed,
                     int dep_time,
                     bool in_allowed)
    : stop_{s},
      headsign_{std::move(headsign)},
      arr_{arr_time, out_allowed},
      dep_{dep_time, in_allowed} {}

trip::trip(route const* route,
           bitfield const* service,
           block* blk,
           std::string id,
           std::string headsign,
           std::string short_name,
           std::size_t line)
    : route_(route),
      service_(service),
      block_{blk},
      id_{std::move(id)},
      headsign_(std::move(headsign)),
      short_name_(std::move(short_name)),
      line_(line) {}

void trip::interpolate() {
  struct bound {
    explicit bound(int t) : min_{t}, max_{t} {}
    int interpolate(int const idx) const {
      auto const p =
          static_cast<double>(idx - min_idx_) / (max_idx_ - min_idx_);
      return static_cast<int>(min_ + std::round((max_ - min_) * p));
    }
    int min_, max_;
    int min_idx_{-1};
    int max_idx_{-1};
  };
  auto bounds = std::vector<bound>{};
  bounds.reserve(stop_times_.size());
  for (auto const [i, x] : utl::enumerate(stop_times_)) {
    bounds.emplace_back(x.second.arr_.time_);
    bounds.emplace_back(x.second.dep_.time_);
  }

  auto max = 0;
  auto max_idx = 0;
  for (auto it = bounds.rbegin(); it != bounds.rend(); ++it) {
    if (it->max_ == stop_time::kInterpolate) {
      it->max_ = max;
      it->max_idx_ = max_idx;
    } else {
      max = it->max_;
      max_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }
  utl::verify(max != stop_time::kInterpolate,
              "last arrival cannot be interpolated");

  auto min = 0;
  auto min_idx = 0;
  for (auto it = bounds.begin(); it != bounds.end(); ++it) {
    if (it->min_ == stop_time::kInterpolate) {
      it->min_ = min;
      it->min_idx_ = min_idx;
    } else {
      min = it->max_;
      min_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }
  utl::verify(min != stop_time::kInterpolate,
              "first arrival cannot be interpolated");

  for (auto const [idx, entry] : utl::enumerate(stop_times_)) {
    auto& [_, stop_time] = entry;
    auto const& arr = bounds[2 * idx];
    auto const& dep = bounds[2 * idx + 1];

    if (stop_time.arr_.time_ == stop_time::kInterpolate) {
      stop_time.arr_.time_ = arr.interpolate(static_cast<int>(idx));
    }
    if (stop_time.dep_.time_ == stop_time::kInterpolate) {
      stop_time.dep_.time_ = dep.interpolate(static_cast<int>(idx));
    }
  }
}

trip::stop_seq trip::stops() const {
  return utl::to_vec(
      stop_times_, [](flat_map<stop_time>::entry_t const& e) -> stop_identity {
        return {e.second.stop_, e.second.arr_.in_out_allowed_,
                e.second.dep_.in_out_allowed_};
      });
}

trip::stop_seq_numbers trip::seq_numbers() const {
  return utl::to_vec(stop_times_,
                     [](flat_map<stop_time>::entry_t const& e) -> unsigned {
                       assert(e.first >= 0);
                       return static_cast<unsigned>(e.first);
                     });
}

int trip::avg_speed() const {
  int travel_time = 0.;  // minutes
  double travel_distance = 0.;  // meters

  for (auto const [dep_entry, arr_entry] : utl::pairwise(stop_times_)) {
    auto const& dep = dep_entry.second;
    auto const& arr = arr_entry.second;
    if (dep.stop_->timezone_ != arr.stop_->timezone_) {
      continue;
    }
    if (arr.arr_.time_ < dep.dep_.time_) {
      continue;
    }

    travel_time += arr.arr_.time_ - dep.dep_.time_;
    travel_distance += geo::distance(dep.stop_->coord_, arr.stop_->coord_);
  }

  return travel_time > 0 ? (travel_distance / 1000.) / (travel_time / 60.) : 0;
}

int trip::distance() const {
  geo::box box;
  for (auto const& [_, stop_time] : stop_times_) {
    box.extend(stop_time.stop_->coord_);
  }
  return geo::distance(box.min_, box.max_) / 1000;
}

void trip::print_stop_times(std::ostream& out, unsigned const indent) const {
  for (auto const& t : stop_times_) {
    for (auto i = 0U; i != indent; ++i) {
      out << "  ";
    }
    out << std::setw(60) << t.second.stop_->name_ << " [" << std::setw(5)
        << t.second.stop_->id_
        << "]: arr: " << (t.second.arr_.time_ * 1_minutes)
        << ", dep: " << (t.second.dep_.time_ * 1_minutes) << "\n";
  }
}

void trip::expand_frequencies(
    std::function<void(trip const&, frequency::schedule_relationship)> const&
        consumer) const {
  utl::verify(frequency_.has_value(), "bad call to trip::expand_frequencies");

  for (auto const& f : frequency_.value()) {
    for (auto start = f.start_time_; start < f.end_time_; start += f.headway_) {
      trip t{*this};

      auto const delta = t.stop_times_.front().dep_.time_ - start;
      for (auto& stop_time : t.stop_times_) {
        stop_time.second.dep_.time_ -= delta;
        stop_time.second.arr_.time_ -= delta;
      }
      consumer(t, f.schedule_relationship_);
    }
  }
}

std::pair<trip_map, block_map> read_trips(route_map_t const& routes,
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

  std::pair<trip_map, block_map> ret;
  auto& trips = ret.first;
  auto& blocks = ret.second;

  auto i = 0U;
  utl::line_range{utl::buf_reader{file_content}}  //
      | utl::csv<csv_trip>()  //
      |
      utl::for_each([&](csv_trip const& t) {
        ++i;

        auto const blk =
            t.block_id_->trim().empty()
                ? nullptr
                : utl::get_or_create(blocks, t.block_id_->trim().view(), []() {
                    return std::make_unique<block>();
                  }).get();
        auto const trp =
            trips
                .emplace(
                    t.trip_id_->to_str(),
                    std::make_unique<trip>(
                        routes.at(t.route_id_->view()).get(),
                        services.traffic_days_.at(t.service_id_->view()).get(),
                        blk, t.trip_id_->to_str(), t.trip_headsign_->to_str(),
                        t.trip_short_name_->to_str(), i))
                .first->second.get();
        if (blk != nullptr) {
          blk->trips_.emplace_back(trp);
        }
      });
  return ret;
}

void read_frequencies(trip_map& trips, std::string_view file_content) {
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

  return utl::line_range{utl::buf_reader{file_content}}  //
         | utl::csv<csv_frequency>()  //
         |
         utl::for_each([&](csv_frequency const& freq) {
           auto const t = freq.trip_id_->trim().view();
           auto const trip_it = trips.find(t);
           if (trip_it == end(trips)) {
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

           auto& frequencies = trip_it->second->frequency_;
           if (!frequencies.has_value()) {
             frequencies = std::vector<frequency>{};
           }
           frequencies->emplace_back(
               frequency{hhmm_to_min(freq.start_time_->view()),
                         hhmm_to_min(freq.end_time_->view()),
                         (headway_secs / 60), schedule_relationship});
         });
}

}  // namespace nigiri::loader::gtfs
