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
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

template <typename K,
          typename Key,
          typename T,
          typename Hash,
          typename KeyEqual,
          typename Allocator,
          unsigned NeighborhoodSize,
          bool StoreHash,
          typename GrowthPolicy,
          typename CreateFun>
auto& get_or_create(tsl::hopscotch_map<Key,
                                       T,
                                       Hash,
                                       KeyEqual,
                                       Allocator,
                                       NeighborhoodSize,
                                       StoreHash,
                                       GrowthPolicy>& m,
                    K&& key,
                    CreateFun&& f) {
  using Map = std::decay_t<decltype(m)>;
  if (auto const it = m.find(key); it == end(m)) {
    return m.emplace_hint(it, typename Map::key_type{std::forward<K>(key)}, f())
        .value();
  } else {
    return it.value();
  }
}

std::vector<std::pair<std::vector<trip*>, bitfield>> block::rule_services() {
  utl::verify(!trips_.empty(), "empty block not allowed");

  utl::erase_if(trips_, [](trip const* t) {
    auto const is_empty = t->stop_seq_.empty();
    if (is_empty) {
      log(log_lvl::error, "loader.gtfs.trip", "trip \"{}\": no stop times",
          t->id_);
    }
    return is_empty;
  });

  std::sort(begin(trips_), end(trips_), [](trip const* a, trip const* b) {
    return a->event_times_.front().dep_ < b->event_times_.front().dep_;
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
        if (timetable::stop{current_it->trip_->stop_seq_.back()}.location_ !=
            timetable::stop{succ_it->trip_->stop_seq_.front()}.location_) {
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

trip::trip(route const* route,
           bitfield const* service,
           block* blk,
           std::string id,
           std::string headsign,
           std::string short_name,
           std::uint32_t line)
    : route_(route),
      service_(service),
      block_{blk},
      id_{std::move(id)},
      headsign_(std::move(headsign)),
      short_name_(std::move(short_name)),
      line_(line) {}

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

void trip::print_stop_times(std::ostream& out,
                            timetable const& tt,
                            unsigned const indent) const {
  for (auto const [stop, ev_times, seq_numbers] :
       utl::zip(stop_seq_, event_times_, seq_numbers_)) {
    auto const s = timetable::stop{stop};
    for (auto i = 0U; i != indent; ++i) {
      out << "  ";
    }
    out << std::setw(60) << location{tt, s.location_idx()}
        << "]: arr: " << ev_times.arr_ << ", dep: " << ev_times.dep_ << "\n";
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
                : get_or_create(blocks, t.block_id_->trim().view(), []() {
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
                         duration_t{headway_secs / 60}, schedule_relationship});
         });
}

}  // namespace nigiri::loader::gtfs
