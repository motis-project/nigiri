#include "nigiri/routing/transfers.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/td_footpath.h"

namespace nigiri::routing {

std::optional<duration_t> shortest_transfer(timetable const& tt,
                                            rt_timetable const* rtt,
                                            profile_idx_t const prf_idx,
                                            location_idx_t const from,
                                            location_idx_t const to) {
  auto shortest = std::optional<duration_t>{};
  if (from == to) {
    if (auto const transfer_time = get_transfer_time(tt, rtt, from);
        transfer_time != kNoTransferAllowed) {
      shortest = duration_t{transfer_time.count()};
    }
  }
  for (auto const fp :
       get_transfers<direction::kForward>(tt, rtt, prf_idx, from)) {
    if (fp.target() == to &&
        (!shortest.has_value() || fp.duration() < *shortest)) {
      shortest = fp.duration();
    }
  }
  return shortest;
}

std::optional<std::pair<duration_t, duration_t>> shortest_transfer(
    timetable const& tt,
    rt_timetable const* rtt,
    profile_idx_t const prf_idx,
    transfer_time_settings const& settings,
    location_idx_t const from,
    location_idx_t const to) {
  auto best = std::optional<std::pair<duration_t, duration_t>>{};
  auto const update = [&](duration_t const d, duration_t const raw) {
    if (!best.has_value() || d < best->first) {
      best = std::pair{d, raw};
    }
  };
  if (from == to) {
    if (auto const transfer_time = get_transfer_time(tt, rtt, from);
        transfer_time != kNoTransferAllowed) {
      auto const raw = duration_t{transfer_time.count()};
      update(adjusted_transfer_time(settings, raw), raw);
    }
  }
  for_each_transfer<direction::kForward>(
      tt, rtt, prf_idx, settings, from,
      [&](footpath const fp, duration_t const d) {
        if (fp.target() == to) {
          update(d, fp.duration());
        }
      });
  return best;
}

std::optional<footpath_restriction> get_footpath_restriction(
    timetable const& tt,
    profile_idx_t const prf_idx,
    transfer_time_settings const& settings,
    location_idx_t const from,
    location_idx_t const to,
    rt_timetable const* rtt) {
  auto const& loc = tt.locations_;
  auto const prf = loc.footpath_profile(prf_idx);
  auto const& fps = loc.footpaths_out_[prf];
  auto const& records = loc.footpath_restrictions_out_[prf];
  if (from >= fps.size() || from >= records.size()) {
    return std::nullopt;
  }
  auto const targets = fps[from];
  auto const r = records[from];
  auto const split = loc.restricted_begin(prf, true, from);
  auto const records_start = records.bucket_starts_[to_idx(from)];
  auto best = std::optional<std::pair<int, footpath_restriction>>{};
  for (auto i = split; i != targets.size(); ++i) {
    if (targets[i].target() != to ||
        (rtt != nullptr &&
         rtt->is_blocked(prf, true, records_start + (i - split)))) {
      continue;
    }
    auto const d = adjusted_transfer_time(
        settings, targets[i].duration().count(), r[i - split]);
    if (d >= 0 && (!best.has_value() || d < best->first)) {
      best = std::pair{d, r[i - split]};
    }
  }
  return best.transform([](auto const& x) { return x.second; });
}

std::optional<footpath_duration> get_footpath_duration(
    timetable const& tt,
    rt_timetable const* rtt,
    profile_idx_t const prf_idx,
    transfer_time_settings const& settings,
    direction const dir,
    unixtime_t const t,
    location_idx_t const from,
    location_idx_t const to) {
  auto const td_prf = tt.locations_.footpath_profile(prf_idx);
  if (from != to && is_td_source(rtt, td_prf, from)) {
    auto best = std::optional<footpath_duration>{};
    for_each_footpath(
        dir, rtt->td_footpaths_out_[td_prf][from], t,
        [&](location_idx_t const target, duration_t const d, duration_t const w,
            footpath_restriction const& r) {
          if (target != to || !is_usable(r, settings.restrictions_)) {
            return;
          }
          if (!best.has_value() || d < best->duration_) {
            best = footpath_duration{d, w, w};
          }
        });
    return best;
  }
  auto const shortest = shortest_transfer(tt, rtt, prf_idx, settings, from, to);
  if (!shortest.has_value()) {
    return std::nullopt;
  }
  return footpath_duration{shortest->first, shortest->first, shortest->second};
}

std::optional<journey::leg> get_offset_leg(timetable const& tt,
                                           event_type const ev,
                                           unixtime_t const t,
                                           location_idx_t const l,
                                           special_station const special,
                                           std::vector<offset> const& offsets,
                                           td_offsets_t const& td_offsets) {
  auto const dir =
      ev == event_type::kDep ? direction::kBackward : direction::kForward;

  auto best = std::optional<std::pair<offset, duration_t>>{};
  auto const update = [&](offset const& o, duration_t const d) {
    if (!best.has_value() || d < best->second) {
      best = std::pair{o, d};
    }
  };

  for (auto const target : {l, tt.locations_.parents_[l]}) {
    if (target == location_idx_t::invalid() ||
        !matches(tt, nullptr, location_match_mode::kIntermodal, target, l)) {
      continue;
    }
    for (auto const& o : offsets) {
      if (o.target() == target) {
        update(o, o.duration());
      }
    }
    if (auto const it = td_offsets.find(target); it != end(td_offsets)) {
      if (auto const td = get_td_duration(dir, it->second, t); td.has_value()) {
        update(offset{target, td->second.duration(), td->second.mode()},
               td->first);
      }
    }
  }

  if (!best.has_value()) {
    return std::nullopt;
  }

  auto const& o = best->first;
  auto const boundary = get_special_station(special);
  if (ev == event_type::kDep) {
    // Boarding -> waiting spent after the walk.
    auto const dep = t - best->second;
    return journey::leg{direction::kForward, boundary, o.target(), dep,
                        dep + o.duration(),  o};
  } else {
    // Alighting -> waiting spent before the walk.
    auto const arr = t + best->second;
    return journey::leg{direction::kForward, o.target(), boundary,
                        arr - o.duration(),  arr,        o};
  }
}

}  // namespace nigiri::routing
