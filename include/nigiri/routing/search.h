#pragma once

#include "fmt/format.h"

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/timing.h"
#include "utl/to_vec.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/get_otel_tracer.h"
#include "nigiri/logging.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/interval_estimate.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct search_state {
  search_state() = default;
  search_state(search_state const&) = delete;
  search_state& operator=(search_state const&) = delete;
  search_state(search_state&&) = default;
  search_state& operator=(search_state&&) = default;
  ~search_state() = default;

  std::vector<std::uint16_t> travel_time_lower_bound_;
  bitvec is_destination_;
  std::array<bitvec, kMaxVias> is_via_;
  std::vector<std::uint16_t> dist_to_dest_;
  std::vector<start> starts_;
  pareto_set<journey> results_;
};

struct search_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {
        {"lb_time", lb_time_},
        {"fastest_direct", fastest_direct_},
        {"interval_extensions", interval_extensions_},
        {"execute_time", execute_time_.count()},
    };
  }

  std::uint64_t lb_time_{0ULL};
  std::uint64_t fastest_direct_{0ULL};
  std::uint64_t interval_extensions_{0ULL};
  std::chrono::milliseconds execute_time_{0LL};
};

struct routing_result {
  pareto_set<journey> const* journeys_{nullptr};
  interval<unixtime_t> interval_;
  search_stats search_stats_;
  std::map<std::string, std::uint64_t> algo_stats_;
};

template <direction SearchDir, typename Algo>
struct search {
  using algo_state_t = typename Algo::algo_state_t;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);

  Algo init(clasz_mask_t const allowed_claszes,
            bool const require_bikes_allowed,
            bool const require_cars_allowed,
            transfer_time_settings& tts,
            algo_state_t& algo_state) {
    auto span = get_otel_tracer()->StartSpan("search::init");
    auto scope = opentelemetry::trace::Scope{span};

    stats_.fastest_direct_ =
        static_cast<std::uint64_t>(fastest_direct_.count());

    utl::verify(q_.via_stops_.size() <= kMaxVias,
                "too many via stops: {}, limit: {}", q_.via_stops_.size(),
                kMaxVias);

    tts.factor_ = std::max(tts.factor_, 1.0F);
    tts.min_transfer_time_ = std::max(tts.min_transfer_time_, 0_minutes);
    tts.additional_time_ = std::max(tts.additional_time_, 0_minutes);
    tts.default_ = tts.factor_ == 1.0F  //
                   && tts.min_transfer_time_ == 0_minutes  //
                   && tts.additional_time_ == 0_minutes;

    collect_destinations(tt_, q_.destination_, q_.dest_match_mode_,
                         state_.is_destination_, state_.dist_to_dest_);

    for (auto const [i, via] : utl::enumerate(q_.via_stops_)) {
      collect_via_destinations(tt_, via.location_, state_.is_via_[i]);
    }

    if constexpr (Algo::kUseLowerBounds) {
      auto lb_span = get_otel_tracer()->StartSpan("lower bounds");
      auto lb_scope = opentelemetry::trace::Scope{lb_span};
      UTL_START_TIMING(lb);
      dijkstra(tt_, q_,
               kFwd ? tt_.fwd_search_lb_graph_[q_.prf_idx_]
                    : tt_.bwd_search_lb_graph_[q_.prf_idx_],
               state_.travel_time_lower_bound_);
      UTL_STOP_TIMING(lb);
      stats_.lb_time_ = static_cast<std::uint64_t>(UTL_TIMING_MS(lb));

#if defined(NIGIRI_TRACING)
      for (auto const& o : q_.start_) {
        trace_upd("start {}: {}\n", location{tt_, o.target()}, o.duration());
      }
      for (auto const& o : q_.destination_) {
        trace_upd("dest {}: {}\n", location{tt_, o.target()}, o.duration());
      }
      for (auto const [l, lb] :
           utl::enumerate(state_.travel_time_lower_bound_)) {
        if (lb != std::numeric_limits<std::decay_t<decltype(lb)>>::max()) {
          trace_upd("lb {}: {}\n", location{tt_, location_idx_t{l}}, lb);
        }
      }
#endif
    }

    return Algo{
        tt_,
        rtt_,
        algo_state,
        state_.is_destination_,
        state_.is_via_,
        state_.dist_to_dest_,
        q_.td_dest_,
        state_.travel_time_lower_bound_,
        q_.via_stops_,
        day_idx_t{
            std::chrono::duration_cast<date::days>(
                std::chrono::round<std::chrono::days>(
                    search_interval_.from_ +
                    ((search_interval_.to_ - search_interval_.from_) / 2)) -
                tt_.internal_interval().from_)
                .count()},
        allowed_claszes,
        require_bikes_allowed,
        require_cars_allowed,
        q_.prf_idx_ == 2U,
        tts};
  }

  search(timetable const& tt,
         rt_timetable const* rtt,
         search_state& s,
         algo_state_t& algo_state,
         query q,
         std::optional<std::chrono::seconds> timeout = std::nullopt)
      : tt_{tt},
        rtt_{rtt},
        state_{s},
        q_{std::move(q)},
        search_interval_{std::visit(
            utl::overloaded{[](interval<unixtime_t> const start_interval) {
                              return start_interval;
                            },
                            [](unixtime_t const start_time) {
                              return interval<unixtime_t>{start_time,
                                                          start_time};
                            }},
            q_.start_time_)},
        fastest_direct_{get_fastest_direct(tt_, q_, SearchDir)},
        algo_{init(q_.allowed_claszes_,
                   q_.require_bike_transport_,
                   q_.require_car_transport_,
                   q_.transfer_time_settings_,
                   algo_state)},
        timeout_(timeout) {
    utl::sort(q_.start_);
    utl::sort(q_.destination_);
    q_.sanitize(tt);
  }

  routing_result execute() {
    auto span = get_otel_tracer()->StartSpan("search::execute");
    auto scope = opentelemetry::trace::Scope{span};

    state_.results_.clear();

    if (start_dest_overlap()) {
      return {&state_.results_, search_interval_, stats_,
              algo_.get_stats().to_map()};
    }

    auto const itv_est = interval_estimator<SearchDir>{tt_, q_};
    if (is_pretrip()) {
      search_interval_ = itv_est.initial(search_interval_);
    }

    state_.starts_.clear();
    if (search_interval_.size() != 0_minutes) {
      add_start_labels(search_interval_, true);
    } else {
      add_start_labels(q_.start_time_, true);
    }

    auto const processing_start_time = std::chrono::steady_clock::now();
    auto const is_timeout_reached = [&]() {
      if (timeout_) {
        return (std::chrono::steady_clock::now() - processing_start_time) >=
               *timeout_;
      }

      return false;
    };

    while (true) {
      trace("start_time={}\n", search_interval_);

      search_interval();

      if (state_.results_.empty() || is_ontrip() ||
          n_results_in_interval() >= q_.min_connection_count_ ||
          is_timeout_reached()) {
        trace(
            "  finished: is_ontrip={}, extend_earlier={}, extend_later={}, "
            "initial={}, interval={}, timetable={}, "
            "number_of_results_in_interval={}, results_with_+1_ontrip={}, "
            "timeout_reached={}\n",
            is_ontrip(), q_.extend_interval_earlier_, q_.extend_interval_later_,
            std::visit(
                utl::overloaded{[](interval<unixtime_t> const& start_interval) {
                                  return start_interval;
                                },
                                [](unixtime_t const start_time) {
                                  return interval<unixtime_t>{start_time,
                                                              start_time};
                                }},
                q_.start_time_),
            search_interval_, tt_.external_interval(), n_results_in_interval(),
            state_.results_.size(), is_timeout_reached());
        span->SetAttribute("nigiri.search.timeout_reached",
                           is_timeout_reached());
        break;
      } else {
        trace(
            "  continue: extend_earlier={}, "
            "extend_later={}, initial={}, interval={}, timetable={}, "
            "number_of_results_in_interval={}\n",
            q_.extend_interval_earlier_, q_.extend_interval_later_,
            std::visit(
                utl::overloaded{[](interval<unixtime_t> const& start_interval) {
                                  return start_interval;
                                },
                                [](unixtime_t const start_time) {
                                  return interval<unixtime_t>{start_time,
                                                              start_time};
                                }},
                q_.start_time_),
            search_interval_, tt_.external_interval(), n_results_in_interval());
      }

      state_.starts_.clear();

      auto const new_interval = itv_est.extension(
          search_interval_, q_.min_connection_count_ - n_results_in_interval());
      trace("interval adapted: {} -> {}\n", search_interval_, new_interval);

      if (new_interval == search_interval_) {
        trace("maximum interval searched: {}\n", search_interval_);
        break;
      }

      if (new_interval.from_ != search_interval_.from_) {
        add_start_labels(interval{new_interval.from_, search_interval_.from_},
                         kBwd);
        if constexpr (kBwd) {
          trace("dir=BWD, interval extension earlier -> reset state\n");
          algo_.reset_arrivals();
          remove_ontrip_results();
        }
      }

      if (new_interval.to_ != search_interval_.to_) {
        add_start_labels(interval{search_interval_.to_, new_interval.to_},
                         kFwd);
        if constexpr (kFwd) {
          trace("dir=BWD, interval extension later -> reset state\n");
          algo_.reset_arrivals();
          remove_ontrip_results();
        }
      }

      search_interval_ = new_interval;

      ++stats_.interval_extensions_;
    }

    if (is_pretrip()) {
      utl::erase_if(state_.results_, [&](journey const& j) {
        return !search_interval_.contains(j.start_time_) ||
               j.travel_time() >= fastest_direct_ ||
               j.travel_time() > q_.max_travel_time_;
      });

      if (q_.slow_direct_) {
        auto direct = std::vector<journey>{};
        auto done = hash_set<std::pair<location_idx_t, location_idx_t>>{};
        for (auto const& j : state_.results_) {
          if (j.transfers_ != 0) {
            continue;
          }
          auto const transport_leg_it =
              utl::find_if(j.legs_, [](journey::leg const& l) {
                return holds_alternative<journey::run_enter_exit>(l.uses_);
              });
          if (transport_leg_it == end(j.legs_)) {
            continue;
          }
          auto const& l = *transport_leg_it;
          get_direct(tt_, rtt_, kFwd ? l.from_ : l.to_, kFwd ? l.to_ : l.from_,
                     q_, search_interval_, SearchDir, done, direct);
        }

        utl::concat(state_.results_.els_, direct);
        utl::erase_duplicates(state_.results_);
      }

      utl::sort(state_.results_, [](journey const& a, journey const& b) {
        return std::tuple{a.start_time_, a.transfers_} <
               std::tuple{b.start_time_, b.transfers_};
      });
    }

    utl::erase_if(state_.results_, [&](auto&& j) { return j.legs_.empty(); });

    stats_.execute_time_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            (std::chrono::steady_clock::now() - processing_start_time));
    return {.journeys_ = &state_.results_,
            .interval_ = search_interval_,
            .search_stats_ = stats_,
            .algo_stats_ = algo_.get_stats().to_map()};
  }

private:
  bool is_ontrip() const {
    return holds_alternative<unixtime_t>(q_.start_time_);
  }

  bool is_pretrip() const { return !is_ontrip(); }

  bool start_dest_overlap() const {
    if (q_.start_match_mode_ == location_match_mode::kIntermodal ||
        q_.dest_match_mode_ == location_match_mode::kIntermodal) {
      return false;
    }

    auto const overlaps_start = [&](location_idx_t const x) {
      return utl::any_of(q_.start_, [&](offset const& o) {
        bool overlaps = false;
        for_each_meta(
            tt_, q_.start_match_mode_, o.target(),
            [&](location_idx_t const eq) { overlaps = overlaps || eq == x; });
        return overlaps;
      });
    };

    return utl::any_of(q_.destination_, [&](offset const& o) {
      auto overlaps = false;
      for_each_meta(tt_, q_.dest_match_mode_, o.target(),
                    [&](location_idx_t const eq) {
                      overlaps = overlaps || overlaps_start(eq);
                    });
      return overlaps;
    });
  }

  unsigned n_results_in_interval() const {
    if (holds_alternative<interval<unixtime_t>>(q_.start_time_)) {
      auto count = utl::count_if(state_.results_, [&](journey const& j) {
        return search_interval_.contains(j.start_time_);
      });
      return static_cast<unsigned>(count);
    } else {
      return static_cast<unsigned>(state_.results_.size());
    }
  }

  bool max_interval_reached() const {
    auto const can_search_earlier =
        q_.extend_interval_earlier_ &&
        search_interval_.from_ != tt_.external_interval().from_;
    auto const can_search_later =
        q_.extend_interval_later_ &&
        search_interval_.to_ != tt_.external_interval().to_;
    return !can_search_earlier && !can_search_later;
  }

  void add_start_labels(start_time_t const& start_interval,
                        bool const add_ontrip) {
    state_.starts_.reserve(500'000);
    get_starts(SearchDir, tt_, rtt_, start_interval, q_.start_, q_.td_start_,
               q_.max_start_offset_, q_.start_match_mode_,
               q_.use_start_footpaths_, state_.starts_, add_ontrip, q_.prf_idx_,
               q_.transfer_time_settings_);
    std::sort(
        begin(state_.starts_), end(state_.starts_),
        [&](start const& a, start const& b) { return kFwd ? b < a : a < b; });
  }

  void remove_ontrip_results() {
    utl::erase_if(state_.results_, [&](journey const& j) {
      return !search_interval_.contains(j.start_time_);
    });
  }

  void search_interval() {
    auto span = get_otel_tracer()->StartSpan("search::search_interval");
    auto scope = opentelemetry::trace::Scope{span};

    utl::equal_ranges_linear(
        state_.starts_,
        [](start const& a, start const& b) {
          return a.time_at_start_ == b.time_at_start_;
        },
        [&](auto&& from_it, auto&& to_it) {
          algo_.next_start_time();
          auto const start_time = from_it->time_at_start_;
          for (auto const& s : it_range{from_it, to_it}) {
            trace("init: time_at_start={}, time_at_stop={} at {}\n",
                  s.time_at_start_, s.time_at_stop_, location{tt_, s.stop_});
            algo_.add_start(s.stop_, s.time_at_stop_);
          }
          trace("RUN ALGO\n");

          /*
           * Upper bound: Search journeys faster than 'worst_time_at_dest'
           * It will not find journeys with the same duration
           */
          auto const worst_time_at_dest =
              start_time + (kFwd ? 1 : -1) *
                               (std::min(fastest_direct_, q_.max_travel_time_) +
                                duration_t{1});
          algo_.execute(start_time, q_.max_transfers_, worst_time_at_dest,
                        q_.prf_idx_, state_.results_);

          for (auto& j : state_.results_) {
            if (j.legs_.empty() && !j.error_ &&
                (is_ontrip() || search_interval_.contains(j.start_time_)) &&
                j.travel_time() < fastest_direct_) {
              try {
                algo_.reconstruct(q_, j);
              } catch (std::exception const& e) {
                j.error_ = true;
                log(log_lvl::error, "search", "reconstruct failed: {}",
                    e.what());
                span->SetStatus(opentelemetry::trace::StatusCode::kError,
                                "exception");
                span->AddEvent(
                    "exception",
                    {{"exception.message",
                      fmt::format("reconstruct failed: {}", e.what())}});
              }
            }
          }
        });
  }

  timetable const& tt_;
  rt_timetable const* rtt_;
  search_state& state_;
  query q_;
  interval<unixtime_t> search_interval_;
  search_stats stats_;
  duration_t fastest_direct_;
  Algo algo_;
  std::optional<std::chrono::seconds> timeout_;
};

}  // namespace nigiri::routing
