#include "nigiri/routing/raptor_search.h"

#include <string>
#include <string_view>
#include <utility>

#include "date/date.h"

#include "fmt/format.h"
#include "fmt/ranges.h"

#include "utl/overloaded.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/get_otel_tracer.h"
#include "nigiri/routing/query.h"

namespace nigiri::routing {

namespace {

template <direction SearchDir, via_offset_t Vias>
routing_result raptor_search_with_vias(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {
  if (rtt == nullptr) {
    using algo_t = raptor<SearchDir, false, Vias, search_mode::kOneToOne>;
    return search<SearchDir, algo_t>{tt,      rtt,          s_state,
                                     r_state, std::move(q), timeout}
        .execute();
  } else {
    using algo_t = raptor<SearchDir, true, Vias, search_mode::kOneToOne>;
    return search<SearchDir, algo_t>{tt,      rtt,          s_state,
                                     r_state, std::move(q), timeout}
        .execute();
  }
}

template <direction SearchDir>
routing_result raptor_search_with_dir(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {
  q.sanitize(tt);
  utl::verify(q.via_stops_.size() <= kMaxVias,
              "too many via stops: {}, limit: {}", q.via_stops_.size(),
              kMaxVias);

  static_assert(kMaxVias == 2,
                "raptor_search.cc needs to be adjusted for kMaxVias");

  switch (q.via_stops_.size()) {
    case 0:
      return raptor_search_with_vias<SearchDir, 0>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
    case 1:
      return raptor_search_with_vias<SearchDir, 1>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
    case 2:
      return raptor_search_with_vias<SearchDir, 2>(tt, rtt, s_state, r_state,
                                                   std::move(q), timeout);
  }
  std::unreachable();
}

std::string_view location_match_mode_str(location_match_mode const mode) {
  using namespace std::literals;
  switch (mode) {
    case location_match_mode::kExact: return "exact"sv;
    case location_match_mode::kOnlyChildren: return "only_children"sv;
    case location_match_mode::kEquivalent: return "equivalent"sv;
    case location_match_mode::kIntermodal: return "intermodal"sv;
  }
  std::unreachable();
}

}  // namespace

routing_result raptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  auto span = get_otel_tracer()->StartSpan("raptor_search");
  auto scope = opentelemetry::trace::Scope{span};
  if (span->IsRecording()) {
    std::visit(utl::overloaded{
                   [&](interval<unixtime_t> const& interval) {
                     span->SetAttribute("nigiri.query.start_time_interval.from",
                                        date::format("%FT%RZ", interval.from_));
                     span->SetAttribute("nigiri.query.start_time_interval.to",
                                        date::format("%FT%RZ", interval.to_));
                   },
                   [&](unixtime_t const& t) {
                     span->SetAttribute("nigiri.query.start_time",
                                        date::format("%FT%RZ", t));
                   }},
               q.start_time_);
    span->SetAttribute("nigiri.query.start_match_mode",
                       location_match_mode_str(q.start_match_mode_));
    span->SetAttribute("nigiri.query.destination_match_mode",
                       location_match_mode_str(q.dest_match_mode_));
    span->SetAttribute("nigiri.query.use_start_footpaths",
                       q.use_start_footpaths_);
    span->SetAttribute("nigiri.query.start_count", q.start_.size());
    span->SetAttribute("nigiri.query.destination_count", q.destination_.size());
    span->SetAttribute("nigiri.query.td_start_count", q.td_start_.size());
    span->SetAttribute("nigiri.query.td_destination_count", q.td_dest_.size());
    span->SetAttribute("nigiri.query.max_start_offset",
                       q.max_start_offset_.count());
    span->SetAttribute("nigiri.query.max_transfers", q.max_transfers_);
    span->SetAttribute("nigiri.query.min_connection_count",
                       q.min_connection_count_);
    span->SetAttribute("nigiri.query.extend_interval_earlier",
                       q.extend_interval_earlier_);
    span->SetAttribute("nigiri.query.extend_interval_later",
                       q.extend_interval_later_);
    span->SetAttribute("nigiri.query.prf_idx", q.prf_idx_);
    span->SetAttribute("nigiri.query.allowed_classes", q.allowed_claszes_);
    span->SetAttribute("nigiri.query.require_bike_transport",
                       q.require_bike_transport_);
    span->SetAttribute("nigiri.query.transfer_time_settings.default",
                       q.transfer_time_settings_.default_);
    span->SetAttribute("nigiri.query.via_stops_count", q.via_stops_.size());
    span->SetAttribute(
        "nigiri.query.search_direction",
        search_dir == direction::kForward ? "forward" : "backward");
    if (timeout) {
      span->SetAttribute("nigiri.query.timeout", timeout.value().count());
    }
  }

  if (search_dir == direction::kForward) {
    return raptor_search_with_dir<direction::kForward>(
        tt, rtt, s_state, r_state, std::move(q), timeout);
  } else {
    return raptor_search_with_dir<direction::kBackward>(
        tt, rtt, s_state, r_state, std::move(q), timeout);
  }
}

}  // namespace nigiri::routing
