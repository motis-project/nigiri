#include "./raptor_search.h"

#include <memory>

#include "gtest/gtest.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

#include "nigiri/routing/gpu/raptor.h"

namespace nigiri::test {

std::string print_results(timetable const& tt,
                          rt_timetable const* rtt,
                          pareto_set<nigiri::routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt, rtt);
    ss << "\n\n";
  }
  return ss.str();
}

unixtime_t parse_time(std::string_view s, char const* format) {
  std::stringstream in;
  in << s;

  date::local_seconds ls;
  std::string tz;
  in >> date::parse(format, ls, tz);

  return std::chrono::time_point_cast<unixtime_t::duration>(
      date::make_zoned(tz, ls).get_sys_time());
}

struct range_result {
  pareto_set<routing::journey> journeys_;
  interval<unixtime_t> interval_;
};

static range_result search_range(timetable const& tt,
                                 rt_timetable const* rtt,
                                 routing::query q,
                                 direction const search_dir) {
  auto search_state = routing::search_state{};
  auto algo_state = routing::raptor_state{};
  auto const result =
      routing::raptor_search(tt, rtt, search_state, algo_state, q, search_dir);
  auto results = *result.journeys_;
  auto const delivered = result.interval_;

#if defined(NIGIRI_CUDA)
  if (routing::gpu::gpu_supported(q, rtt)) {
    auto gpu_search_state = routing::search_state{};
    auto gpu_timetable = routing::gpu::gpu_timetable{tt};
    auto gpu_state = routing::gpu::gpu_raptor_state{gpu_timetable};
    if (rtt != nullptr) {
      // Re-upload every call: tests mutate rtt between searches.
      const_cast<rt_timetable&>(*rtt).gpu_rtt_.ptr_ =
          routing::gpu::make_gpu_rtt(tt, *rtt);
    }
    auto gpu_results = *(routing::raptor_search(tt, rtt, gpu_search_state,
                                                gpu_state, q, search_dir)
                             .journeys_);

    EXPECT_EQ(print_results(tt, rtt, results),
              print_results(tt, rtt, gpu_results));
  }
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

static pareto_set<routing::journey> search_pong(timetable const& tt,
                                                rt_timetable const* rtt,
                                                routing::query q,
                                                direction const search_dir) {
  auto search_state = routing::search_state{};
  auto algo_state = routing::raptor_state{};
  auto results =
      *(routing::pong_search(tt, rtt, search_state, algo_state, q, search_dir)
            .journeys_);

#if defined(NIGIRI_CUDA)
  if (routing::gpu::gpu_supported(q, rtt)) {
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
                                           direction const search_dir) {
  auto [results, delivered] = search_range(tt, rtt, q, search_dir);
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
  return std::move(results);
}

}  // namespace nigiri::test
