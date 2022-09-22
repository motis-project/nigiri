#include "utl/enumerate.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/search_state.h"

namespace nigiri::routing {

template <direction SearchDir>
void reconstruct_journey(timetable const& tt,
                         query const& q,
                         search_state const& s,
                         journey& j) {
  (void)q;  // TODO(felix) support intermodal start

  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const is_better_or_eq = [](auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  };

  auto const find_entry_in_prev_round =
      [&](unsigned const k, transport const& t, route_idx_t const r,
          std::size_t const from_stop_idx,
          routing_time const time) -> std::optional<journey::leg> {
    auto const& stop_seq = tt.route_location_seq_[r];

    auto const n_stops =
        kFwd ? from_stop_idx + 1 : stop_seq.size() - from_stop_idx;
    for (auto i = 1U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<unsigned>(kFwd ? from_stop_idx - i : from_stop_idx + i);
      auto const l = timetable::stop{stop_seq[stop_idx]}.location_idx();

      auto const event_time = routing_time{
          t.day_, tt.event_mam(t.t_idx_, stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)};
      if (is_better_or_eq(s.round_times_[k - 1][to_idx(l)], event_time)) {
        return journey::leg{
            SearchDir,
            timetable::stop{stop_seq[stop_idx]}.location_idx(),
            timetable::stop{stop_seq[from_stop_idx]}.location_idx(),
            event_time.to_unixtime(tt),
            time.to_unixtime(tt),
            journey::transport_enter_exit{
                t, stop_idx, static_cast<unsigned>(from_stop_idx)}};
      }

      // special case: first stop with meta stations
      if (k == 1 && q.start_match_mode_ == location_match_mode::kEquivalent) {
        for (auto const& eq : tt.locations_.equivalences_[l]) {
          if (is_better_or_eq(s.round_times_[k - 1][to_idx(eq)], event_time)) {
            return journey::leg{
                SearchDir,
                timetable::stop{stop_seq[stop_idx]}.location_idx(),
                timetable::stop{stop_seq[from_stop_idx]}.location_idx(),
                event_time.to_unixtime(tt),
                time.to_unixtime(tt),
                journey::transport_enter_exit{
                    t, stop_idx, static_cast<unsigned>(from_stop_idx)}};
          }
        }
      }
    }

    return std::nullopt;
  };

  auto const get_route_transport =
      [&](unsigned const k, routing_time const time, route_idx_t const r,
          std::size_t const stop_idx) -> std::optional<journey::leg> {
    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const event_mam =
          tt.event_mam(t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      if (event_mam.count() % 1440 != time.mam().count()) {
        fmt::print(
            std::cerr,
            "  {}: ev_mam mismatch at {}: transport_ev={} vs footpath = {}\n ",
            tt.trip_display_names_
                [tt.merged_trips_[tt.transport_to_trip_section_[t].front()]
                     .front()]
                    .view(),
            tt.locations_
                .names_[timetable::stop{tt.route_location_seq_[r][stop_idx]}
                            .location_idx()]
                .view(),
            duration_t{event_mam.count()}, duration_t{time.mam().count()});
        continue;
      }

      auto const day_offset =
          static_cast<cista::base_t<day_idx_t>>(event_mam.count() / 1440);
      auto const day = time.day() - day_offset;
      if (!tt.bitfields_[tt.transport_traffic_days_[t]].test(to_idx(day))) {
        continue;
      }

      auto const leg =
          find_entry_in_prev_round(k, transport{t, day}, r, stop_idx, time);
      if (leg.has_value()) {
        return leg;
      }
    }
    return std::nullopt;
  };

  auto const get_transport =
      [&](unsigned const k, location_idx_t const l,
          routing_time const time) -> std::optional<journey::leg> {
    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [i, stop] : utl::enumerate(location_seq)) {
        if (timetable::stop{stop}.location_idx() != l ||
            (kFwd && (i == 0U || !timetable::stop{stop}.out_allowed())) ||
            (!kFwd && (i == location_seq.size() - 1 ||
                       !timetable::stop{stop}.in_allowed()))) {
          continue;
        }

        auto const leg = get_route_transport(k, time, r, i);
        if (leg.has_value()) {
          return leg;
        }
      }
    }
    return std::nullopt;
  };

  auto const check_fp = [&](unsigned const k, location_idx_t const l,
                            routing_time const curr_time, footpath const fp)
      -> std::optional<std::pair<journey::leg, journey::leg>> {
    auto const fp_start = curr_time - (kFwd ? fp.duration_ : -fp.duration_);

    fmt::print(std::cerr,
               "round {}: searching for transports at {} with fp_start = {}\n ",
               k, tt.locations_.names_.at(fp.target_).view(), fp_start);

    auto const transport_leg = get_transport(k, fp.target_, fp_start);

    if (transport_leg.has_value()) {
      std::cerr << "found:\n";
      transport_leg->print(std::cerr, tt, 1, false);

      auto const fp_leg = journey::leg{SearchDir,
                                       fp.target_,
                                       l,
                                       fp_start.to_unixtime(tt),
                                       curr_time.to_unixtime(tt),
                                       footpath_idx_t::invalid()};
      return std::pair{fp_leg, *transport_leg};
    } else {
      std::cerr << "nothing found\n";
    }
    return std::nullopt;
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    std::cerr << "CHECKING TRANSFER\n";
    auto const curr_time = s.round_times_[k][to_idx(l)];
    auto const transfer_at_same_stop =
        check_fp(k, l, curr_time, footpath{l, tt.locations_.transfer_time_[l]});
    if (transfer_at_same_stop.has_value()) {
      return std::move(*transfer_at_same_stop);
    }

    std::cerr << "CHECKING FOOTPATHS OF " << tt.locations_.names_.at(l).view()
              << "\n";
    auto const fps =
        kFwd ? tt.locations_.footpaths_in_[l] : tt.locations_.footpaths_out_[l];
    for (auto const& fp : fps) {
      std::cerr << "FP: " << fp.duration_.count() << " --> "
                << tt.locations_.names_.at(fp.target_).view() << "\n";
      auto const adjusted_duration =
          fp.duration_ -
          (kFwd ? 1 : -1) * tt.locations_.transfer_time_[fp.target_];
      auto const fp_legs =
          check_fp(k, l, curr_time, footpath{fp.target_, adjusted_duration});
      if (fp_legs.has_value()) {
        return std::move(*fp_legs);
      }
    }

    throw utl::fail(
        "reconstruction failed at k={}, t={}, stop=(name={}, id={}), time={}",
        k, j.transfers_, tt.locations_.names_[l].view(),
        tt.locations_.ids_[l].view(), curr_time);
  };

  auto l = j.dest_;
  for (auto i = 0U; i <= j.transfers_; ++i) {
    auto const k = j.transfers_ + 1 - i;
    auto [fp_leg, transport_leg] = get_legs(k, l);
    l = kFwd ? transport_leg.from_ : transport_leg.to_;
    j.add(std::move(fp_leg));
    j.add(std::move(transport_leg));
  }

  if constexpr (kFwd) {
    std::reverse(begin(j.legs_), end(j.legs_));
  }
}

template void reconstruct_journey<direction::kForward>(timetable const& tt,
                                                       query const& q,
                                                       search_state const& s,
                                                       journey& j);

template void reconstruct_journey<direction::kBackward>(timetable const& tt,
                                                        query const& q,
                                                        search_state const& s,
                                                        journey& j);

}  // namespace nigiri::routing
