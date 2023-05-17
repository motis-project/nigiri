#pragma once

#include "utl/equal_ranges_linear.h"
#include "utl/erase_if.h"
#include "utl/timing.h"

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/for_each_meta.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#define NIGIRI_LOWER_BOUND

#define NIGIRI_RAPTOR_COUNTING
#ifdef NIGIRI_RAPTOR_COUNTING
#define NIGIRI_COUNT(s) ++stats_.s
#else
#define NIGIRI_COUNT(s)
#endif

// #define NIGIRI_ROUTING_TRACING
// #define NIGIRI_ROUTING_TRACING_ONLY_UPDATES
// #define NIGIRI_ROUTING_INTERVAL_TRACING

#ifdef NIGIRI_ROUTING_INTERVAL_TRACING
#define trace_interval(...) fmt::print(__VA_ARGS__)
#else
#define trace_interval(...)
#endif

#ifdef NIGIRI_ROUTING_TRACING

#ifdef NIGIRI_ROUTING_TRACING_ONLY_UPDATES
#define trace(...)
#else
#define trace(...) fmt::print(__VA_ARGS__)
#endif

#define trace_always(...) fmt::print(__VA_ARGS__)
#define trace_upd(...) fmt::print(__VA_ARGS__)
#else
#define trace(...)
#define trace_always(...)
#define trace_upd(...)
#endif

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct search_state {
  void reset(timetable const& tt) {
    is_destination_.resize(tt.n_locations());
    utl::fill(is_destination_, false);

    travel_time_lower_bound_.resize(tt.n_locations());
    utl::fill(travel_time_lower_bound_, duration_t{0});

    starts_.clear();
    destinations_.clear();
    results_.clear();
  }

  std::vector<duration_t> travel_time_lower_bound_;
  std::vector<bool> is_destination_;
  std::vector<start> starts_;
  std::vector<std::set<location_idx_t>> destinations_;
  std::vector<pareto_set<journey>> results_;
  interval<unixtime_t> search_interval_;
};

struct search_stats {
  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t lb_time_{0ULL};
  std::uint64_t fastest_direct_{0ULL};
  std::uint64_t search_iterations_{0ULL};
  std::uint64_t interval_extensions_{0ULL};
};

template <direction SearchDir, typename Algo>
struct search {
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);

  search(timetable const& tt, search_state& s, query q)
      : search_state_{s},
        tt_{tt},
        q_{std::move(q)},
        fastest_direct_{get_fastest_direct(tt_, q_, SearchDir)} {}

  constexpr Algo& algo() { return *static_cast<Algo*>(this); }

  bool is_ontrip() const {
    return holds_alternative<unixtime_t>(q_.start_time_);
  }

  bool start_dest_overlap() const {
    if (q_.start_match_mode_ == location_match_mode::kIntermodal ||
        q_.dest_match_mode_ == location_match_mode::kIntermodal) {
      // Handled by fastest_direct
      return false;
    }

    auto const overlaps_start = [&](location_idx_t const x) {
      return utl::any_of(q_.start_, [&](offset const& o) {
        bool overlaps = false;
        for_each_meta(
            tt_, q_.start_match_mode_, o.target_,
            [&](location_idx_t const eq) { overlaps = overlaps || eq == x; });
        return overlaps;
      });
    };

    return utl::any_of(q_.destinations_.front(), [&](offset const& o) {
      bool overlaps = false;
      for_each_meta(tt_, q_.dest_match_mode_, o.target_,
                    [&](location_idx_t const eq) {
                      overlaps = overlaps || overlaps_start(eq);
                    });
      return overlaps;
    });
  }

  void route() {
    algo().reset_state();
    search_state_.reset(tt_);

    collect_destinations(tt_, q_.destinations_, q_.dest_match_mode_,
                         search_state_.destinations_,
                         search_state_.is_destination_);
    search_state_.results_.resize(std::max(search_state_.results_.size(),
                                           search_state_.destinations_.size()));
    if (start_dest_overlap()) {
      trace_always("start/dest overlap - finished");
      return;
    }

    if constexpr (Algo::use_lower_bounds) {
      UTL_START_TIMING(lb);
      dijkstra(tt_, q_,
               kFwd ? tt_.fwd_search_lb_graph_ : tt_.bwd_search_lb_graph_,
               search_state_.travel_time_lower_bound_);
      for (auto l = location_idx_t{0U}; l != tt_.locations_.children_.size();
           ++l) {
        auto const lb = search_state_.travel_time_lower_bound_[to_idx(l)];
        for (auto const c : tt_.locations_.children_[l]) {
          search_state_.travel_time_lower_bound_[to_idx(c)] =
              std::min(lb, search_state_.travel_time_lower_bound_[to_idx(c)]);
        }
      }
      UTL_STOP_TIMING(lb);
      stats_.lb_time_ = static_cast<std::uint64_t>(UTL_TIMING_MS(lb));
    }

#ifdef NIGIRI_ROUTING_TRACING
    for (auto const& o : q_.start_) {
      trace_always("start {}: {}\n", location{tt_, o.target_}, o.duration_);
    }
    for (auto const& o : q_.destinations_.front()) {
      trace_always("dest {}: {}\n", location{tt_, o.target_}, o.duration_);
    }
    for (auto const [l, lb] :
         utl::enumerate(search_state_.travel_time_lower_bound_)) {
      if (lb.count() != std::numeric_limits<duration_t::rep>::max()) {
        trace_always("lb {}: {}\n", location{tt_, location_idx_t{l}},
                     lb.count());
      }
    }
#endif

    stats_.fastest_direct_ =
        static_cast<std::uint64_t>(fastest_direct_.count());
    trace_always("fastest direct {}", stats_.fastest_direct_);

    auto const number_of_results_in_interval = [&]() {
      if (holds_alternative<interval<unixtime_t>>(q_.start_time_)) {
        return static_cast<std::size_t>(utl::count_if(
            search_state_.results_.front(), [&](journey const& j) {
              return search_state_.search_interval_.contains(j.start_time_);
            }));
      } else {
        return search_state_.results_.front().size();
      }
    };

    auto const max_interval_reached = [&]() {
      auto const can_search_earlier =
          q_.extend_interval_earlier_ &&
          search_state_.search_interval_.from_ != tt_.external_interval().from_;
      auto const can_search_later =
          q_.extend_interval_later_ &&
          search_state_.search_interval_.to_ != tt_.external_interval().to_;
      return !can_search_earlier && !can_search_later;
    };

    auto add_start_labels = [&](start_time_t const& start_interval,
                                bool const add_ontrip) {
      get_starts(SearchDir, tt_, start_interval, q_.start_,
                 q_.start_match_mode_, q_.use_start_footpaths_,
                 search_state_.starts_, add_ontrip);
    };

    auto const remove_ontrip_results = [&]() {
      for (auto& r : search_state_.results_) {
        utl::erase_if(r, [&](journey const& j) {
          return !search_state_.search_interval_.contains(j.start_time_);
        });
      }
    };

    search_state_.search_interval_ = q_.start_time_.apply(
        utl::overloaded{[](interval<unixtime_t> const& start_interval) {
                          return start_interval;
                        },
                        [](unixtime_t const start_time) {
                          return interval<unixtime_t>{start_time, start_time};
                        }});
    add_start_labels(q_.start_time_, true);
    trace_interval("start_time={}",
                   q_.start_time_.apply(utl::overloaded{
                       [](interval<unixtime_t> const& start_interval) {
                         return start_interval;
                       },
                       [](unixtime_t const start_time) {
                         return interval<unixtime_t>{start_time, start_time};
                       }}));

    while (true) {
      utl::equal_ranges_linear(
          search_state_.starts_,
          [](start const& a, start const& b) {
            return a.time_at_start_ == b.time_at_start_;
          },
          [&](auto&& from_it, auto&& to_it) {
            algo().next_start_time(from_it->time_at_start_);
            for (auto const& s : it_range{from_it, to_it}) {
              algo().init(s.stop_, routing_time{tt_, s.time_at_stop_});
              trace_always(
                  "init: time_at_start={}, time_at_stop={} at (name={} "
                  "id={})\n",
                  s.time_at_start_, s.time_at_stop_,
                  tt_.locations_.names_.at(s.stop_).view(),
                  tt_.locations_.ids_.at(s.stop_).view());
            }
            algo().set_time_at_destination(
                routing_time{tt_, from_it->time_at_start_} +
                (kFwd ? 1 : -1) *
                    (std::min(fastest_direct_, duration_t{kMaxTravelTime}) +
                     1_minutes));
            trace_always(
                "time_at_destination={} + kMaxTravelTime/fastest_direct={} = "
                "{}\n",
                routing_time{tt_, from_it->time_at_start_}, fastest_direct_,
                time_at_destination_);
            algo().rounds();
            trace_always("reconstruct: time_at_start={}\n",
                         from_it->time_at_start_);
            algo().reconstruct(from_it->time_at_start_);
          });

      if (is_ontrip() || max_interval_reached() ||
          number_of_results_in_interval() >= q_.min_connection_count_) {
        trace_interval(
            "  finished: is_ontrip={}, max_interval_reached={}, "
            "extend_earlier={}, extend_later={}, initial={}, interval={}, "
            "timetable={}, number_of_results_in_interval={}\n",
            is_ontrip(), max_interval_reached(), q_.extend_interval_earlier_,
            q_.extend_interval_later_,
            q_.start_time_.apply(utl::overloaded{
                [](interval<unixtime_t> const& start_interval) {
                  return start_interval;
                },
                [](unixtime_t const start_time) {
                  return interval<unixtime_t>{start_time, start_time};
                }}),
            search_state_.search_interval_, tt_.external_interval(),
            number_of_results_in_interval());
        break;
      } else {
        trace_interval(
            "  continue: max_interval_reached={}, extend_earlier={}, "
            "extend_later={}, initial={}, interval={}, timetable={}, "
            "number_of_results_in_interval={}\n",
            max_interval_reached(), q_.extend_interval_earlier_,
            q_.extend_interval_later_,
            q_.start_time_.apply(utl::overloaded{
                [](interval<unixtime_t> const& start_interval) {
                  return start_interval;
                },
                [](unixtime_t const start_time) {
                  return interval<unixtime_t>{start_time, start_time};
                }}),
            search_state_.search_interval_, tt_.external_interval(),
            number_of_results_in_interval());
      }

      search_state_.starts_.clear();

      auto const new_interval =
          interval{q_.extend_interval_earlier_
                       ? tt_.external_interval().clamp(
                             search_state_.search_interval_.from_ - 60_minutes)
                       : search_state_.search_interval_.from_,
                   q_.extend_interval_later_
                       ? tt_.external_interval().clamp(
                             search_state_.search_interval_.to_ + 60_minutes)
                       : search_state_.search_interval_.to_};
      trace_interval("interval adapted: {} -> {}\n",
                     search_state_.search_interval_, new_interval);

      if (new_interval.from_ != search_state_.search_interval_.from_) {
        trace_interval("extend interval earlier - add_start_labels({}, {})\n",
                       new_interval.from_,
                       search_state_.search_interval_.from_);
        add_start_labels(
            interval{new_interval.from_, search_state_.search_interval_.from_},
            SearchDir == direction::kBackward);
        if constexpr (SearchDir == direction::kBackward) {
          trace_interval(
              "dir=BWD, interval extension earlier -> reset state\n");
          algo().reset_arrivals();
          remove_ontrip_results();
        }
      }

      if (new_interval.to_ != search_state_.search_interval_.to_) {
        trace_interval("extend interval later - add_start_labels({}, {})\n",
                       search_state_.search_interval_.to_, new_interval.to_);
        add_start_labels(
            interval{search_state_.search_interval_.to_, new_interval.to_},
            SearchDir == direction::kForward);
        if constexpr (SearchDir == direction::kForward) {
          trace_interval("dir=BWD, interval extension later -> reset state\n");
          algo().reset_arrivals();
          remove_ontrip_results();
        }
      }

      search_state_.search_interval_ = new_interval;

      ++stats_.search_iterations_;
    }

    if (holds_alternative<interval<unixtime_t>>(q_.start_time_)) {
      remove_ontrip_results();
      for (auto& r : search_state_.results_) {
        utl::erase_if(r, [&](journey const& j) {
          return j.travel_time() > fastest_direct_ ||
                 j.travel_time() > kMaxTravelTime;
        });
      }
      for (auto& r : search_state_.results_) {
        std::sort(begin(r), end(r), [](journey const& a, journey const& b) {
          return a.start_time_ < b.start_time_;
        });
      }
    }
  }

  search_state& search_state_;
  timetable const& tt_;
  search_stats stats_;
  query q_;
  duration_t fastest_direct_;
};

}  // namespace nigiri::routing