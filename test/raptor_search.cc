#include "./raptor_search.h"

#include "nigiri/routing/arcflag_filter.h"
#include "nigiri/routing/no_route_filter.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/reach_filter.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::test {

unixtime_t parse_time(std::string_view s, char const* format) {
  std::stringstream in;
  in << s;

  date::local_seconds ls;
  std::string tz;
  in >> date::parse(format, ls, tz);

  return std::chrono::time_point_cast<unixtime_t::duration>(
      date::make_zoned(tz, ls).get_sys_time());
}

template <direction SearchDir, typename Filter>
routing::routing_result<routing::raptor_stats> run_search_rt(
    routing::search_state& search_state,
    routing::raptor_state& raptor_state,
    Filter const& filter,
    timetable const& tt,
    rt_timetable const* rtt,
    routing::query&& q) {
  if (rtt == nullptr) {
    using algo_t = routing::raptor<Filter, SearchDir, false, false>;
    return routing::search<SearchDir, algo_t>{
        tt, nullptr, search_state, raptor_state, filter, std::move(q)}
        .execute();
  } else {
    using algo_t = routing::raptor<Filter, SearchDir, true, false>;
    return routing::search<SearchDir, algo_t>{
        tt, rtt, search_state, raptor_state, filter, std::move(q)}
        .execute();
  }
}

template <direction SearchDir>
routing::routing_result<routing::raptor_stats> run_search(
    timetable const& tt, rt_timetable const* rtt, routing::query&& q) {
  static auto no_route_filter = routing::no_route_filter{};
  static auto arcflag_filter = routing::arcflag_filter{};
  static auto reach_filter = routing::reach_filter{};
  static auto search_state = routing::search_state{};
  static auto raptor_state = routing::raptor_state{};

  if (!tt.route_reachs_.empty()) {
    reach_filter.init(tt, q);
    return run_search_rt<SearchDir>(search_state, raptor_state, reach_filter,
                                    tt, rtt, std::move(q));
  } else if (!tt.arc_flags_.empty()) {
    arcflag_filter.init(tt, q);
    return run_search_rt<SearchDir>(search_state, raptor_state, arcflag_filter,
                                    tt, rtt, std::move(q));
  } else {
    return run_search_rt<SearchDir>(search_state, raptor_state, no_route_filter,
                                    tt, rtt, std::move(q));
  }
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           direction const search_dir) {
  if (search_dir == direction::kForward) {
    return *run_search<direction::kForward>(tt, rtt, std::move(q)).journeys_;
  } else {
    return *run_search<direction::kBackward>(tt, rtt, std::move(q)).journeys_;
  }
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           routing::start_time_t time,
                                           direction const search_dir) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({to, src}), 0_minutes, 0U}}};
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           std::string_view time,
                                           direction const search_dir) {
  return raptor_search(tt, rtt, from, to, parse_time(time, "%Y-%m-%d %H:%M %Z"),
                       search_dir);
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
      .extend_interval_later_ = extend_interval_later};
  return raptor_search(tt, rtt, std::move(q), search_dir);
}

template routing::routing_result<routing::raptor_stats>
run_search<direction::kForward>(timetable const& tt,
                                rt_timetable const* rtt,
                                routing::query&& q);

template routing::routing_result<routing::raptor_stats>
run_search<direction::kBackward>(timetable const& tt,
                                 rt_timetable const* rtt,
                                 routing::query&& q);

}  // namespace nigiri::test