#include "nigiri/rt/rt_timetable.h"
#include "nigiri/types.h"
#include <optional>

namespace nigiri {

rt_transport_idx_t rt_timetable::add_rt_transport(
    source_idx_t const src,
    timetable const& tt,
    transport const t,
    std::span<stop::value_type> const& stop_seq,
    std::span<delta_t> const& time_seq) {
  auto const [t_idx, day] = t;

  auto const rt_t_idx = rt_transport_src_.size();
  auto const rt_t = rt_transport_idx_t{rt_t_idx};
  // ADDED/NEW stop+time, REPLACEMENT stop, DUPL time
  if (time_seq.empty() && t.is_valid()) {  // REPL
    static_trip_lookup_.emplace(t, rt_t_idx);
    rt_transport_static_transport_.emplace_back(t);

    auto const static_bf = bitfields_[transport_traffic_days_[t_idx]];
    bitfields_.emplace_back(static_bf).set(to_idx(day), false);
    transport_traffic_days_[t_idx] = bitfield_idx_t{bitfields_.size() - 1U};
  }

  auto const r =
      t.is_valid() ? std::optional{tt.transport_route_[t_idx]} : std::nullopt;
  auto const location_seq = stop_seq.empty() && r.has_value()
                                ? std::span{tt.route_location_seq_[*r]}
                                : stop_seq;
  rt_transport_location_seq_.emplace_back(location_seq);
  rt_transport_src_.emplace_back(src);
  rt_transport_train_nr_.emplace_back(0U);

  for (auto const s : location_seq) {
    auto rt_transports = location_rt_transports_[stop{s}.location_idx()];
    if (rt_transports.empty() || rt_transports.back() != rt_t) {
      rt_transports.push_back(rt_t);
    }
  }

  if (time_seq.empty() && r.has_value()) {
    auto times =
        rt_transport_stop_times_.add_back_sized(location_seq.size() * 2U - 2U);
    auto i = 0U;
    auto const static_location_seq_len = tt.route_location_seq_[*r].size();
    auto stop_idx = stop_idx_t{0U};
    for (auto const [a, b] : utl::pairwise(location_seq)) {
      CISTA_UNUSED_PARAM(a)
      CISTA_UNUSED_PARAM(b)
      times[i++] = unix_to_delta(tt.event_time(t, stop_idx, event_type::kDep));
      times[i++] =
          unix_to_delta(tt.event_time(t, ++stop_idx, event_type::kArr));
      if (stop_idx + 1 >= static_location_seq_len) {
        break;
      }
    }
  } else {
    rt_transport_stop_times_.emplace_back(time_seq);
  }

  auto const bikes_allowed_default = false;  // TODO
  auto const default_clasz = std::vector<clasz>{clasz::kOther};
  rt_transport_display_names_.add_back_sized(0U);
  if (r.has_value()) {
    rt_transport_section_clasz_.emplace_back(tt.route_section_clasz_[*r]);
  } else {
    rt_transport_section_clasz_.emplace_back(std::vector<clasz>{clasz::kOther});
  }
  rt_transport_line_.add_back_sized(0U);
  rt_transport_is_cancelled_.resize(rt_transport_is_cancelled_.size() + 1U);

  rt_transport_bikes_allowed_.resize(rt_transport_bikes_allowed_.size() + 2U);
  rt_transport_bikes_allowed_.set(
      rt_t_idx * 2, r.has_value() ? tt.route_bikes_allowed_[r->v_ * 2]
                                  : bikes_allowed_default);
  rt_transport_bikes_allowed_.set(
      rt_t_idx * 2 + 1, r.has_value() ? tt.route_bikes_allowed_[r->v_ * 2 + 1]
                                      : bikes_allowed_default);
  if (r.has_value()) {
    rt_bikes_allowed_per_section_.emplace_back(
        tt.route_bikes_allowed_per_section_[*r]);
  } else {
    rt_bikes_allowed_per_section_.emplace_back(
        std::vector<bool>{bikes_allowed_default});
  }

  assert(static_trip_lookup_.contains(t));
  assert(rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] == t);
  assert(rt_transport_static_transport_.size() == rt_t_idx + 1U);
  assert(rt_transport_src_.size() == rt_t_idx + 1U);
  assert(rt_transport_stop_times_.size() == rt_t_idx + 1U);
  assert(rt_transport_location_seq_.size() == rt_t_idx + 1U);
  assert(rt_transport_display_names_.size() == rt_t_idx + 1U);
  assert(rt_transport_section_clasz_.size() == rt_t_idx + 1U);
  assert(rt_transport_line_.size() == rt_t_idx + 1U);
  assert(rt_bikes_allowed_per_section_.size() == rt_t_idx + 1U);

  return rt_transport_idx_t{rt_t_idx};
}

}  // namespace nigiri
