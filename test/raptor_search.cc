#include "./raptor_search.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "utl/overloaded.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/mc_criteria.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#if defined(NIGIRI_CUDA)
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/raptor.h"
#endif

namespace nigiri::test {

std::string print_results(timetable const& tt,
                          rt_timetable const* rtt,
                          pareto_set<routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt, rtt);
    ss << "\n\n";
  }
  return ss.str();
}

namespace {

using tuple_t = std::tuple<unixtime_t, unixtime_t, std::uint8_t>;

// what every engine must agree on, whichever equal-value legs it reports
std::vector<tuple_t> journey_tuples(pareto_set<routing::journey> const& js) {
  auto v = std::vector<tuple_t>{};
  for (auto const& j : js) {
    v.emplace_back(j.start_time_, j.dest_time_, j.transfers_);
  }
  std::sort(begin(v), end(v));
  return v;
}

#if defined(NIGIRI_CUDA)
using criteria_tuple_t = std::tuple<unixtime_t,
                                    unixtime_t,
                                    std::uint8_t,
                                    std::uint16_t,
                                    bool,
                                    std::uint8_t>;

// ... plus the extra criteria, for engines optimizing the same ones
std::vector<criteria_tuple_t> criteria_tuples(
    pareto_set<routing::journey> const& js) {
  auto v = std::vector<criteria_tuple_t>{};
  for (auto const& j : js) {
    v.emplace_back(j.start_time_, j.dest_time_, j.transfers_, j.criteria_cost_,
                   j.criteria_mode_filter_, j.criteria_mode_switches_);
  }
  std::sort(begin(v), end(v));
  return v;
}
#endif

template <typename State, typename... StateArgs>
pareto_set<routing::journey> run_search(timetable const& tt,
                                        rt_timetable const* rtt,
                                        routing::query const& q,
                                        direction const search_dir,
                                        StateArgs&&... state_args) {
  auto search_state = routing::search_state{};
  auto state = State{std::forward<StateArgs>(state_args)...};
  return *routing::raptor_search(tt, rtt, search_state, state, q, search_dir)
              .journeys_;
}

struct mc_crits {
  bool non_transit_{false};
  bool mode_filter_{false};
  bool mode_switches_{false};
};

// The legs of a McRAPTOR journey reproduce the criteria it was optimized for:
// each walk (offset, footpath between stops - not the same-stop transfer)
// adds its non-transit units, the classes of the rides give the mode filter
// and the mode switches.
void expect_legs_match_criteria(timetable const& tt,
                                rt_timetable const* rtt,
                                pareto_set<routing::journey> const& js,
                                mc_crits const c,
                                std::string_view const config) {
  using namespace routing;
  for (auto const& j : js) {
    auto units = 0U;
    auto avoided = false;
    auto group = kNoModeGroup;
    auto switches = 0U;
    auto rides = 0U;
    for (auto const& l : j.legs_) {
      std::visit(
          utl::overloaded{
              [&](offset const& o) {
                units = non_transit_add(
                    units, static_cast<std::uint32_t>(o.duration().count()));
              },
              [&](footpath const& fp) {
                if (l.from_ != l.to_) {
                  units = non_transit_add(
                      units, static_cast<std::uint32_t>(fp.duration().count()));
                }
              },
              [&](journey::run_enter_exit const& ree) {
                ++rides;
                auto const cl =
                    ree.r_.is_rt()
                        ? rtt->rt_transport_section_clasz_[ree.r_.rt_][0]
                        : tt.route_clasz_
                              [tt.transport_route_[ree.r_.t_.t_idx_]];
                avoided =
                    avoided || is_allowed(mode_filter_dim::avoided_mask(), cl);
                auto const g = mode_group(cl);
                switches = switches_after_ride(group, switches, g);
                group = g;
              }},
          l.uses_);
    }
    auto ss = std::stringstream{};
    ss << config << ": " << j.departure_time() << " -> " << j.arrival_time();
    EXPECT_EQ(rides == 0U ? 0U : rides - 1U, unsigned{j.transfers_})
        << ss.str();
    EXPECT_EQ(c.non_transit_ ? units * kNonTransitBin : 0U,
              unsigned{j.criteria_cost_})
        << ss.str();
    EXPECT_EQ(c.mode_filter_ && avoided, j.criteria_mode_filter_) << ss.str();
    EXPECT_EQ(c.mode_switches_ ? switches : 0U,
              unsigned{j.criteria_mode_switches_})
        << ss.str();
  }
}

// extra criteria keep every raptor journey (or a variant of it)
template <typename State>
pareto_set<routing::journey> expect_superset(timetable const& tt,
                                             rt_timetable const* rtt,
                                             routing::query const& q,
                                             direction const search_dir,
                                             std::vector<tuple_t> const& base,
                                             mc_crits const crits,
                                             std::string_view const config) {
  auto mc = run_search<State>(tt, rtt, q, search_dir);
  expect_legs_match_criteria(tt, rtt, mc, crits, config);
  auto const mc_tuples = journey_tuples(mc);
  for (auto const& t : base) {
    EXPECT_TRUE(std::binary_search(begin(mc_tuples), end(mc_tuples), t))
        << "raptor journey missing in mcraptor (" << config << ")";
  }
  return mc;
}

#if defined(NIGIRI_CUDA)
// Compared on values, not legs: racing device inserts may pick other trips of
// equal value.
template <typename GpuState>
void expect_gpu_equal(timetable const& tt,
                      rt_timetable const* rtt,
                      routing::query const& q,
                      direction const search_dir,
                      routing::gpu::gpu_timetable const& gpu_tt,
                      pareto_set<routing::journey> const& cpu,
                      mc_crits const crits,
                      std::string_view const config) {
  auto const gpu = run_search<GpuState>(tt, rtt, q, search_dir, gpu_tt);
  EXPECT_EQ(criteria_tuples(cpu), criteria_tuples(gpu))
      << "gpu mcraptor (" << config << ")";
  expect_legs_match_criteria(tt, rtt, gpu, crits, config);
}
#endif

}  // namespace

struct range_result {
  pareto_set<routing::journey> journeys_;
  interval<unixtime_t> interval_;
};

static range_result search_range(timetable const& tt,
                                 rt_timetable const* rtt,
                                 routing::query q,
                                 direction const search_dir,
                                 bool const mc_same_legs) {
  auto search_state = routing::search_state{};
  auto algo_state = routing::raptor_state{};
  auto const result =
      routing::raptor_search(tt, rtt, search_state, algo_state, q, search_dir);
  auto results = *result.journeys_;
  auto const delivered = result.interval_;
  auto const base_tuples = journey_tuples(results);

  auto const mc = routing::mcraptor_supported(q, rtt);
  auto mc_non_transit = pareto_set<routing::journey>{};
  auto mc_mode_filter = pareto_set<routing::journey>{};
  auto mc_non_transit_mode_filter = pareto_set<routing::journey>{};
  auto mc_mode_switches = pareto_set<routing::journey>{};
  auto mc_non_transit_mode_switches = pareto_set<routing::journey>{};
  auto mc_mode_filter_mode_switches = pareto_set<routing::journey>{};
  auto mc_all = pareto_set<routing::journey>{};
  if (mc) {
    // without extra criteria, mcraptor is raptor - down to the legs (as
    // found: mcraptor does not run optimize_footpaths)
    auto const mc_plain =
        run_search<routing::mcraptor_state>(tt, rtt, q, search_dir);
    if (mc_same_legs) {
      auto as_found_state = routing::raptor_state{};
      as_found_state.skip_optimize_footpaths_ = true;
      auto as_found_search_state = routing::search_state{};
      EXPECT_EQ(
          print_results(tt, rtt,
                        *routing::raptor_search(tt, rtt, as_found_search_state,
                                                as_found_state, q, search_dir)
                             .journeys_),
          print_results(tt, rtt, mc_plain))
          << "mcraptor";
    } else {
      EXPECT_EQ(base_tuples, journey_tuples(mc_plain)) << "mcraptor";
    }
    expect_legs_match_criteria(tt, rtt, mc_plain, {}, "mcraptor");

    using namespace routing;
    auto const& b = base_tuples;
    mc_non_transit = expect_superset<mcraptor_non_transit_state>(
        tt, rtt, q, search_dir, b, {.non_transit_ = true}, "non_transit");
    mc_mode_filter = expect_superset<mcraptor_mode_filter_state>(
        tt, rtt, q, search_dir, b, {.mode_filter_ = true}, "mode_filter");
    mc_non_transit_mode_filter =
        expect_superset<mcraptor_non_transit_mode_filter_state>(
            tt, rtt, q, search_dir, b,
            {.non_transit_ = true, .mode_filter_ = true},
            "non_transit + mode_filter");
    mc_mode_switches = expect_superset<mcraptor_mode_switches_state>(
        tt, rtt, q, search_dir, b, {.mode_switches_ = true}, "mode_switches");
    mc_non_transit_mode_switches =
        expect_superset<mcraptor_non_transit_mode_switches_state>(
            tt, rtt, q, search_dir, b,
            {.non_transit_ = true, .mode_switches_ = true},
            "non_transit + mode_switches");
    mc_mode_filter_mode_switches =
        expect_superset<mcraptor_mode_filter_mode_switches_state>(
            tt, rtt, q, search_dir, b,
            {.mode_filter_ = true, .mode_switches_ = true},
            "mode_filter + mode_switches");
    mc_all =
        expect_superset<mcraptor_non_transit_mode_filter_mode_switches_state>(
            tt, rtt, q, search_dir, b,
            {.non_transit_ = true,
             .mode_filter_ = true,
             .mode_switches_ = true},
            "non_transit + mode_filter + mode_switches");
  }

#if defined(NIGIRI_CUDA)
  // no gpu_available() gate: a CUDA build without a device should fail
  if (routing::gpu::gpu_supported(q)) {  // rtt: uploaded below
    auto gpu_timetable = routing::gpu::gpu_timetable{tt};
    if (rtt != nullptr) {
      // Re-upload every call: tests mutate rtt between searches.
      const_cast<rt_timetable&>(*rtt).gpu_rtt_.ptr_ =
          routing::gpu::make_gpu_rtt(tt, *rtt);
    }

    EXPECT_EQ(print_results(tt, rtt, results),
              print_results(tt, rtt,
                            run_search<routing::gpu::gpu_raptor_state>(
                                tt, rtt, q, search_dir, gpu_timetable)));

    if (mc) {
      auto const gpu_plain = run_search<routing::gpu::gpu_mcraptor_state>(
          tt, rtt, q, search_dir, gpu_timetable);
      EXPECT_EQ(base_tuples, journey_tuples(gpu_plain)) << "gpu mcraptor";
      expect_legs_match_criteria(tt, rtt, gpu_plain, {}, "gpu mcraptor");
      using namespace routing::gpu;
      auto const& g = gpu_timetable;
      expect_gpu_equal<gpu_mcraptor_non_transit_state>(
          tt, rtt, q, search_dir, g, mc_non_transit, {.non_transit_ = true},
          "non_transit");
      expect_gpu_equal<gpu_mcraptor_mode_filter_state>(
          tt, rtt, q, search_dir, g, mc_mode_filter, {.mode_filter_ = true},
          "mode_filter");
      expect_gpu_equal<gpu_mcraptor_non_transit_mode_filter_state>(
          tt, rtt, q, search_dir, g, mc_non_transit_mode_filter,
          {.non_transit_ = true, .mode_filter_ = true},
          "non_transit + mode_filter");
      expect_gpu_equal<gpu_mcraptor_mode_switches_state>(
          tt, rtt, q, search_dir, g, mc_mode_switches, {.mode_switches_ = true},
          "mode_switches");
      expect_gpu_equal<gpu_mcraptor_non_transit_mode_switches_state>(
          tt, rtt, q, search_dir, g, mc_non_transit_mode_switches,
          {.non_transit_ = true, .mode_switches_ = true},
          "non_transit + mode_switches");
      expect_gpu_equal<gpu_mcraptor_mode_filter_mode_switches_state>(
          tt, rtt, q, search_dir, g, mc_mode_filter_mode_switches,
          {.mode_filter_ = true, .mode_switches_ = true},
          "mode_filter + mode_switches");
      expect_gpu_equal<
          gpu_mcraptor_non_transit_mode_filter_mode_switches_state>(
          tt, rtt, q, search_dir, g, mc_all,
          {.non_transit_ = true, .mode_filter_ = true, .mode_switches_ = true},
          "non_transit + mode_filter + mode_switches");
    }
  }
#else
  [[maybe_unused]] static auto const warn_no_cuda_once = [&] {
    std::cerr << "\n[raptor_search] WARNING: built without NIGIRI_CUDA - the "
                 "GPU comparison checks are SKIPPED for this run.\n\n";
    return true;
  }();
#endif

  return {std::move(results), delivered};
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           routing::start_time_t time,
                                           direction const search_dir,
                                           routing::clasz_mask_t const mask,
                                           bool const require_bikes_allowed,
                                           bool const require_cars_allowed,
                                           bool const no_compulsory_reservation,
                                           profile_idx_t const profile) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                        0_minutes, 0U}},
      .prf_idx_ = profile,
      .allowed_claszes_ = mask,
      .require_bike_transport_ = require_bikes_allowed,
      .require_car_transport_ = require_cars_allowed,
      .no_compulsory_reservation_ = no_compulsory_reservation,
      .via_stops_ = {}};
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

pareto_set<routing::journey> raptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    std::string_view from,
    std::string_view to,
    std::string_view time,
    direction const search_dir,
    routing::clasz_mask_t mask,
    bool const require_bikes_allowed,
    bool const require_cars_allowed,
    bool const no_compulsory_reservation) {
  return raptor_search(tt, rtt, from, to,
                       parse_time_tz(time, "%Y-%m-%d %H:%M %Z"), search_dir,
                       mask, require_bikes_allowed, require_cars_allowed,
                       no_compulsory_reservation, 0U);
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query&& q,
                                           std::string_view from,
                                           std::string_view to,
                                           std::string_view time,
                                           direction const search_dir) {
  auto const src = source_idx_t{0};
  if (!from.empty()) {
    q.start_ = {
        {tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes, 0U}};
  }
  if (!to.empty()) {
    q.destination_ = {
        {tt.locations_.location_id_to_idx_.at({to, src}), 0_minutes, 0U}};
  }
  if (!time.empty()) {
    q.start_time_ = parse_time_tz(time, "%Y-%m-%d %H:%M %Z");
  }
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

pareto_set<routing::journey> raptor_intermodal_search(
    timetable const& tt,
    rt_timetable const* rtt,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    routing::start_time_t const interval,
    direction const search_dir,
    std::uint8_t const min_connection_count,
    bool const extend_interval_earlier,
    bool const extend_interval_later,
    bool const mc_same_legs) {
  auto q = routing::query{
      .start_time_ = interval,
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .start_ = std::move(start),
      .destination_ = std::move(destination),
      .min_connection_count_ = min_connection_count,
      .extend_interval_earlier_ = extend_interval_earlier,
      .extend_interval_later_ = extend_interval_later,
      .prf_idx_ = 0,
      .via_stops_ = {}};
  return raptor_search(tt, rtt, std::move(q), search_dir, mc_same_legs);
}

pareto_set<routing::journey> search_pong(timetable const& tt,
                                         rt_timetable const* rtt,
                                         routing::query q,
                                         direction const search_dir) {
  auto search_state = routing::search_state{};
  auto algo_state = routing::raptor_state{};
  auto results =
      *(routing::pong_search(tt, rtt, search_state, algo_state, q, search_dir)
            .journeys_);

#if defined(NIGIRI_CUDA)
  if (routing::gpu::gpu_supported(q)) {  // rtt: uploaded below
    auto gpu_search_state = routing::search_state{};
    auto gpu_timetable = routing::gpu::gpu_timetable{tt};
    auto gpu_state = routing::gpu::gpu_raptor_state{gpu_timetable};
    if (rtt != nullptr) {
      // Re-upload every call: tests mutate rtt between searches.
      const_cast<rt_timetable&>(*rtt).gpu_rtt_.ptr_ =
          routing::gpu::make_gpu_rtt(tt, *rtt);
    }
    auto gpu_results = *(routing::pong_search(tt, rtt, gpu_search_state,
                                              gpu_state, q, search_dir)
                             .journeys_);

    EXPECT_EQ(print_results(tt, rtt, results),
              print_results(tt, rtt, gpu_results));
  }
#endif

  return results;
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           direction const search_dir,
                                           bool const mc_same_legs) {
  auto [results, delivered] =
      search_range(tt, rtt, q, search_dir, mc_same_legs);
  if (std::holds_alternative<interval<unixtime_t>>(q.start_time_)) {
    auto const pong_results = search_pong(tt, rtt, q, search_dir);
    auto const iv = std::get<interval<unixtime_t>>(q.start_time_);
    auto const in_window = [&](pareto_set<routing::journey> const& set) {
      auto filtered = pareto_set<routing::journey>{};
      for (auto const& j : set) {
        auto const anchor = search_dir == direction::kForward
                                ? j.departure_time()
                                : j.arrival_time();
        if (!q.extend_interval_earlier_) {
          EXPECT_GE(anchor, iv.from_);
        }
        if (!q.extend_interval_later_) {
          EXPECT_LT(anchor, iv.to_);
        }
        if (iv.contains(anchor)) {
          filtered.add(routing::journey{j});
        }
      }
      return filtered;
    };
    EXPECT_EQ(print_results(tt, rtt, in_window(results)),
              print_results(tt, rtt, in_window(pong_results)));
  }
  return results;
}

}  // namespace nigiri::test
