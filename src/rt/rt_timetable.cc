#include "nigiri/rt/rt_timetable.h"

#include "utl/enumerate.h"
#include "utl/overloaded.h"
#include "utl/timer.h"

#include "nigiri/loader/gtfs/route.h"

namespace nigiri {

rt_transport_idx_t rt_timetable::add_rt_transport(
    source_idx_t const src,
    timetable const& tt,
    transport const t,
    std::span<stop::value_type> stop_seq,
    std::span<delta_t> time_seq,
    std::string_view new_trip_id,
    std::string_view route_id,
    direction_id_t const direction_id,
    std::string_view trip_short_name,
    delta_t const offset) {
  auto const [t_idx, day] = t;

  auto const rt_t_idx = rt_transport_src_.size();
  auto const rt_t = rt_transport_idx_t{rt_t_idx};
  if (new_trip_id.empty() && t.is_valid()) {
    static_trip_lookup_.emplace(t, rt_t_idx);
    rt_transport_static_transport_.emplace_back(t);

    auto const static_bf = bitfields_[transport_traffic_days_[t_idx]];
    bitfields_.emplace_back(static_bf).set(to_idx(day), false);
    transport_traffic_days_[t_idx] = bitfield_idx_t{bitfields_.size() - 1U};
  } else {
    auto const rt_add_idx =
        rt_add_trip_id_idx_t{additional_trips_.at(src).transports_.size()};
    additional_trips_.at(src).ids_.store(new_trip_id);
    additional_trips_.at(src).transports_.emplace_back(rt_t_idx);
    rt_transport_static_transport_.emplace_back(rt_add_idx);
  }

  auto const r =
      t.is_valid() ? tt.transport_route_[t_idx] : route_idx_t::invalid();
  auto const given_r = tt.route_ids_[src].ids_.find(route_id).value_or(
      route_id_idx_t::invalid());
  auto const location_seq = stop_seq.empty() && r != route_idx_t::invalid()
                                ? std::span{tt.route_location_seq_[r]}
                                : stop_seq;
  rt_transport_location_seq_.emplace_back(location_seq);
  rt_transport_src_.emplace_back(src);
  rt_transport_route_id_.emplace_back(given_r);
  alerts_.rt_transport_.emplace_back_empty();

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
      if (stop_idx + 1U >= static_location_seq_len) {
        break;
      }
    }
  } else {
    rt_transport_stop_times_.emplace_back(time_seq);
  }

  auto const bikes_allowed_default = false;  // TODO
  auto const cars_allowed_default = false;  // TODO

  rt_transport_line_.add_back_sized(0U);
  rt_transport_is_cancelled_.resize(rt_transport_is_cancelled_.size() + 1U);
  rt_transport_bikes_allowed_.resize(rt_transport_bikes_allowed_.size() + 2U);
  rt_transport_cars_allowed_.resize(rt_transport_bikes_allowed_.size() + 2U);
  rt_transport_section_directions_.add_back_sized(0U);  // TODO outside
  rt_transport_trip_short_names_.emplace_back(trip_short_name);

  rt_transport_direction_id_.resize(rt_transport_direction_id_.size() + 1U);
  rt_transport_direction_id_.set(rt_t, direction_id != direction_id_t{});

  if (r != route_idx_t::invalid()) {
    rt_transport_section_clasz_.emplace_back(tt.route_section_clasz_[r]);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2,
                                    tt.route_bikes_allowed_[r.v_ * 2]);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2 + 1,
                                    tt.route_bikes_allowed_[r.v_ * 2 + 1]);
    rt_transport_cars_allowed_.set(rt_t_idx * 2,
                                   tt.route_cars_allowed_[r.v_ * 2]);
    rt_transport_cars_allowed_.set(rt_t_idx * 2 + 1,
                                   tt.route_cars_allowed_[r.v_ * 2 + 1]);
  } else if (given_r != route_id_idx_t::invalid()) {
    rt_transport_section_clasz_.emplace_back(
        std::vector<clasz>{loader::gtfs::to_clasz(
            tt.route_ids_[src].route_id_type_.at(given_r).v_)});  // TODO
    rt_transport_bikes_allowed_.set(rt_t_idx * 2, bikes_allowed_default);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2 + 1, false);
    rt_transport_cars_allowed_.set(rt_t_idx * 2, cars_allowed_default);
    rt_transport_cars_allowed_.set(rt_t_idx * 2 + 1, false);
  } else {
    rt_transport_section_clasz_.emplace_back(std::vector<clasz>{clasz::kOther});
    rt_transport_bikes_allowed_.set(rt_t_idx * 2, bikes_allowed_default);
    rt_transport_bikes_allowed_.set(rt_t_idx * 2 + 1, false);
    rt_transport_cars_allowed_.set(rt_t_idx * 2, cars_allowed_default);
    rt_transport_cars_allowed_.set(rt_t_idx * 2 + 1, false);
  }
  if (r != route_idx_t::invalid() && stop_seq.empty()) {
    rt_bikes_allowed_per_section_.emplace_back(
        tt.route_bikes_allowed_per_section_[r]);
    rt_cars_allowed_per_section_.emplace_back(
        tt.route_cars_allowed_per_section_[r]);
  } else {
    rt_bikes_allowed_per_section_.emplace_back(
        std::vector<bool>{bikes_allowed_default});
    rt_cars_allowed_per_section_.emplace_back(
        std::vector<bool>{cars_allowed_default});
  }

  assert(time_seq.empty() || time_seq.size() == location_seq.size() * 2U - 2U);
  assert(static_trip_lookup_.contains(t) ||
         additional_trips_.at(src).ids_.find(new_trip_id).has_value());
  assert(rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] == t ||
         rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] ==
             rt_add_trip_id_idx_t{additional_trips_.at(src).transports_.size() -
                                  1U});
  assert(additional_trips_.at(src).transports_.size() ==
         additional_trips_.at(src).ids_.strings_.size());
  assert(rt_transport_static_transport_.size() == rt_t_idx + 1U);
  assert(rt_transport_src_.size() == rt_t_idx + 1U);
  assert(rt_transport_route_id_.size() == rt_t_idx + 1U);
  assert(rt_transport_stop_times_.size() == rt_t_idx + 1U);
  assert(rt_transport_location_seq_.size() == rt_t_idx + 1U);
  assert(rt_transport_trip_short_names_.size() == rt_t_idx + 1U);
  assert(rt_transport_section_clasz_.size() == rt_t_idx + 1U);
  assert(rt_transport_line_.size() == rt_t_idx + 1U);
  assert(rt_bikes_allowed_per_section_.size() == rt_t_idx + 1U);

  return rt_transport_idx_t{rt_t_idx};
}

std::string_view rt_timetable::transport_name(
    timetable const& tt, rt_transport_idx_t const t) const {
  return std::visit(utl::overloaded{[&](translation_idx_t const idx) {
                                      return tt.get_default_translation(idx);
                                    },
                                    [](std::string_view const s) { return s; }},
                    trip_short_name(tt, t));
}

void rt_timetable::update_lbs(timetable const& tt,
                              rt_transport_idx_t const rt_t,
                              stop_idx_t const stop_idx,
                              event_type const ev_type,
                              std::array<paged_vecvec<location_idx_t, footpath>,
                                         kNProfiles>& fwd_search_lb_graph,
                              std::array<paged_vecvec<location_idx_t, footpath>,
                                         kNProfiles>& bwd_search_lb_graph) {
  auto const from_stop_idx = ev_type == event_type::kDep
                                 ? stop_idx
                                 : static_cast<stop_idx_t>(stop_idx - 1);
  auto const to_stop_idx = ev_type == event_type::kDep
                               ? static_cast<stop_idx_t>(stop_idx + 1U)
                               : stop_idx;

  auto const travel_time =
      duration_t{event_time(rt_t, to_stop_idx, event_type::kArr) -
                 event_time(rt_t, from_stop_idx, event_type::kDep)};

  auto const loc_seq = rt_transport_location_seq_[rt_t];
  if (travel_time < duration_t{0}) {
    log(log_lvl::error, "nigiri.rt.update_time",
        "travel_time < 0: {} -> {}: dep={} - arr={}",
        loc{tt, stop{loc_seq[from_stop_idx]}.location_idx()},
        loc{tt, stop{loc_seq[to_stop_idx]}.location_idx()},
        event_time(rt_t, from_stop_idx, event_type::kDep),
        event_time(rt_t, to_stop_idx, event_type::kArr));
    return;
  }

  auto const from =
      tt.locations_.get_root_idx(stop{loc_seq[from_stop_idx]}.location_idx());
  auto const to =
      tt.locations_.get_root_idx(stop{loc_seq[to_stop_idx]}.location_idx());

  if (from == to) {
    return;  // e.g. from one child to another within the same parent
  }

  auto const update = [&](paged_vecvec<location_idx_t, footpath>& lbs,
                          direction const dir) {
    auto bucket = lbs.at(dir == direction::kForward ? from : to);
    auto const target = dir == direction::kForward ? to : from;
    auto const it = utl::find_if(
        bucket, [&](footpath const& fp) { return fp.target() == target; });
    if (it == end(bucket)) {
      bucket.push_back(footpath{target, travel_time});
    } else if (it->duration() > travel_time) {
      it->duration_ = static_cast<location_idx_t::value_t>(travel_time.count());
    }
  };

  for (auto& lbs : fwd_search_lb_graph) {
    if (!lbs.empty()) {
      update(lbs, direction::kBackward);
    }
  }

  for (auto& lbs : bwd_search_lb_graph) {
    if (!lbs.empty()) {
      update(lbs, direction::kForward);
    }
  }
}

void rt_timetable::update_lbs(timetable const& tt) {
  auto timer = utl::scoped_timer{"update_lbs"};

  auto const copy = [](auto&& to, auto&& from) {
    for (auto const [i, x] : utl::enumerate(from)) {
      to[i].clear();
      for (auto const y : x) {
        to[i].emplace_back(y);
      }
    }
  };

  auto fwd_search_lb_graph =
      std::array<paged_vecvec<location_idx_t, footpath>, kNProfiles>{};
  auto bwd_search_lb_graph =
      std::array<paged_vecvec<location_idx_t, footpath>, kNProfiles>{};

  copy(fwd_search_lb_graph, tt.fwd_search_lb_graph_);
  copy(bwd_search_lb_graph, tt.bwd_search_lb_graph_);

  for (auto rt_t = rt_transport_idx_t{0U}; rt_t != n_rt_transports(); ++rt_t) {
    auto const n_events = rt_transport_stop_times_[rt_t].size();
    auto const n_segments = static_cast<stop_idx_t>(n_events / 2U);
    for (auto i = stop_idx_t{0U}; i != n_segments; ++i) {
      update_lbs(tt, rt_t, i, event_type::kDep, fwd_search_lb_graph,
                 bwd_search_lb_graph);
    }
  }

  copy(fwd_search_lb_graph_, fwd_search_lb_graph);
  copy(bwd_search_lb_graph_, bwd_search_lb_graph);
}

void rt_timetable::cancel_run(rt::run const& r) {
  if (r.is_rt()) {
    rt_transport_is_cancelled_.set(to_idx(r.rt_), true);
  }
  if (r.is_scheduled()) {
    auto const bf = bitfields_[transport_traffic_days_[r.t_.t_idx_]];
    bitfields_.emplace_back(bf).set(to_idx(r.t_.day_), false);
    transport_traffic_days_[r.t_.t_idx_] =
        bitfield_idx_t{bitfields_.size() - 1U};

    for (auto i = r.stop_range_.from_; i != r.stop_range_.to_; ++i) {
      dispatch_stop_change(r, i, event_type::kArr, std::nullopt, false);
      dispatch_stop_change(r, i, event_type::kDep, std::nullopt, false);
    }
  }
}

}  // namespace nigiri
