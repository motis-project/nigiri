#include "./raptor_search.h"

#include "gtest/gtest.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

#include "nigiri/routing/gpu/raptor.h"

namespace nigiri::test {

std::string print_results(timetable const& tt,
                          pareto_set<nigiri::routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
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

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           direction const search_dir) {
  auto search_state = routing::search_state{};
  auto algo_state = routing::raptor_state{};
  auto results =
      *(routing::raptor_search(tt, rtt, search_state, algo_state, q, search_dir)
            .journeys_);

  auto gpu_search_state = routing::search_state{};
  auto gpu_timetable = routing::gpu::gpu_timetable{tt};
  auto gpu_state = routing::gpu::gpu_raptor_state{gpu_timetable};
  auto gpu_results = *(routing::raptor_search(tt, rtt, gpu_search_state,
                                              gpu_state, q, search_dir)
                           .journeys_);

  EXPECT_EQ(print_results(tt, results), print_results(tt, gpu_results));

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
                                           bool const require_bikes_allowed) {
  return raptor_search(tt, rtt, from, to, parse_time(time, "%Y-%m-%d %H:%M %Z"),
                       search_dir, mask, require_bikes_allowed, 0U);
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
    q.start_time_ = parse_time(time, "%Y-%m-%d %H:%M %Z");
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
