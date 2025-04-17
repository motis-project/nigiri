#include "nigiri/rt/rt_timetable.h"

namespace nigiri {

rt_transport_idx_t rt_timetable::add_rt_transport(
    source_idx_t const src,
    timetable const& tt,
    transport const t,
    std::span<stop::value_type> const stop_seq,
    std::span<delta_t> const time_seq,
    std::string_view const new_trip_id,
    std::string_view const route_id,
    std::string_view const display_name,
    delta_t const offset) {
  auto const [t_idx, day] = t;

  auto const rt_t_idx = rt_transport_src_.size();
  auto const rt_t = rt_transport_idx_t{rt_t_idx};
  // ADDED/NEW stop+time+new_trip_id, REPLACEMENT stop+time, DUPL new_trip_id
  if (new_trip_id.empty() && t.is_valid()) {  // REPL
    static_trip_lookup_.emplace(t, rt_t_idx);
    rt_transport_static_transport_.emplace_back(t);

    auto const static_bf = bitfields_[transport_traffic_days_[t_idx]];
    bitfields_.emplace_back(static_bf).set(to_idx(day), false);
    transport_traffic_days_[t_idx] = bitfield_idx_t{bitfields_.size() - 1U};
  } else {
    auto const rt_add_idx =
        rt_add_trip_id_idx_t{additional_trips_lookup_.size()};
    trip_id_strings_.emplace_back(
        additional_trips_lookup_.emplace(new_trip_id, rt_t_idx).first);
    rt_transport_static_transport_.emplace_back(rt_add_idx);
  }

  auto const search_by_route_id = [&]() {
    if (route_id.empty()) {
      return route_idx_t::invalid();
    }
    auto const lb = std::lower_bound(
        begin(tt.route_id_to_idx_), end(tt.route_id_to_idx_), route_id,
        [&](pair<route_id_idx_t, route_idx_t> const& a, std::string_view b) {
          return std::tuple(tt.route_id_src_[a.first],
                            tt.route_id_strings_[a.first].view()) <
                 std::tuple(src, b);
        });
    if (lb != end(tt.route_id_to_idx_) &&
        route_id == tt.route_id_strings_[lb->first].view()) {
      return lb->second;
    }
    return route_idx_t::invalid();
  };

  auto const r =
      t.is_valid() ? tt.transport_route_[t_idx] : search_by_route_id();
  auto const location_seq = stop_seq.empty() && r != route_idx_t::invalid()
                                ? std::span{tt.route_location_seq_[r]}
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

  if (time_seq.empty() && r != route_idx_t::invalid()) {
    auto times =
        rt_transport_stop_times_.add_back_sized(location_seq.size() * 2U - 2U);
    auto i = 0U;
    auto const static_location_seq_len = tt.route_location_seq_[r].size();
    auto stop_idx = stop_idx_t{0U};
    for (auto const [a, b] : utl::pairwise(location_seq)) {
      CISTA_UNUSED_PARAM(a)
      CISTA_UNUSED_PARAM(b)
      times[i++] =
          unix_to_delta(tt.event_time(t, stop_idx, event_type::kDep)) + offset;
      times[i++] =
          unix_to_delta(tt.event_time(t, ++stop_idx, event_type::kArr)) +
          offset;
      if (stop_idx + 1U >= static_location_seq_len) {  // TODO assert?
        break;
      }
    }
  } else {
    rt_transport_stop_times_.emplace_back(time_seq);
  }

  auto const bikes_allowed_default = false;  // TODO

  rt_transport_line_.add_back_sized(0U);
  rt_transport_is_cancelled_.resize(rt_transport_is_cancelled_.size() + 1U);
  rt_transport_bikes_allowed_.resize(rt_transport_bikes_allowed_.size() + 2U);
  rt_transport_section_directions_.add_back_sized(0U);  // TODO outside
  if (!display_name.empty()) {
    rt_transport_display_names_.emplace_back(display_name);
  } else if (!new_trip_id.empty() && t.is_valid()) {
    rt_transport_display_names_.emplace_back(tt.transport_name(t.t_idx_));
  } else {
    rt_transport_display_names_.add_back_sized(0);
  }
  if (r != route_idx_t::invalid()) {
    rt_transport_section_clasz_.emplace_back(tt.route_section_clasz_[r]);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2,
                                    tt.route_bikes_allowed_[r.v_ * 2]);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2 + 1,
                                    tt.route_bikes_allowed_[r.v_ * 2 + 1]);
  } else {
    rt_transport_section_clasz_.emplace_back(std::vector<clasz>{clasz::kOther});
    rt_transport_bikes_allowed_.set(rt_t_idx * 2, bikes_allowed_default);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2 + 1, false);
  }
  if (r != route_idx_t::invalid() && stop_seq.empty()) {
    rt_bikes_allowed_per_section_.emplace_back(
        tt.route_bikes_allowed_per_section_[r]);
  } else {
    rt_bikes_allowed_per_section_.emplace_back(
        std::vector<bool>{bikes_allowed_default});
  }

  assert(time_seq.empty() || time_seq.size() == location_seq.size() * 2U - 2U);
  assert(static_trip_lookup_.contains(t) ||
         additional_trips_lookup_.contains(new_trip_id));
  assert(rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] == t ||
         rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] ==
             rt_add_trip_id_idx_t{additional_trips_lookup_.size() - 1U});
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
