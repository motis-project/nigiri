#include "./raptor_search.h"

#include <sstream>

#include "gtest/gtest.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::test {

namespace {

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

}  // namespace

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           direction const search_dir) {
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};

  auto const results = *(routing::raptor_search(tt, rtt, search_state,
                                                algo_state, q, search_dir)
                             .journeys_);

  if (routing::mcraptor_supported(q, rtt)) {
    auto mc_search_state = routing::search_state{};
    auto mc_state = routing::mcraptor_state{};
    auto const mc_results = *(routing::raptor_search(tt, rtt, mc_search_state,
                                                     mc_state, q, search_dir)
                                  .journeys_);
    EXPECT_EQ(print_results(tt, rtt, results),
              print_results(tt, rtt, mc_results));

    // generalized-cost criteria: the cost configuration keeps additional
    // pareto trade-offs (later/more transfers but cheaper), so its
    // journeys must be a superset of the baseline on
    // (start, dest, transfers)
    auto walk_search_state = routing::search_state{};
    auto walk_state = routing::mcraptor_cost_state{};
    auto const walk_results =
        *(routing::raptor_search(tt, rtt, walk_search_state, walk_state,
                                 std::move(q), search_dir)
              .journeys_);
    auto const tuples = [](pareto_set<routing::journey> const& js) {
      auto v = std::vector<
          std::tuple<unixtime_t, unixtime_t, std::uint8_t>>{};
      for (auto const& j : js) {
        v.emplace_back(j.start_time_, j.dest_time_, j.transfers_);
      }
      std::sort(begin(v), end(v));
      return v;
    };
    auto const base_tuples = tuples(results);
    auto const walk_tuples = tuples(walk_results);
    for (auto const& t : base_tuples) {
      EXPECT_TRUE(std::find(begin(walk_tuples), end(walk_tuples), t) !=
                  end(walk_tuples))
          << "baseline journey missing in walk results";
    }
  }

  return results;
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
      .via_stops_ = {}};
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           std::string_view time,
                                           direction const search_dir,
                                           routing::clasz_mask_t mask,
                                           bool const require_bikes_allowed,
                                           bool const require_cars_allowed) {
  return raptor_search(tt, rtt, from, to,
                       parse_time_tz(time, "%Y-%m-%d %H:%M %Z"), search_dir,
                       mask, require_bikes_allowed, require_cars_allowed, 0U);
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
    bool const extend_interval_later) {
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
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

}  // namespace nigiri::test
