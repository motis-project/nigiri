#include "nigiri/loader/gtfs/seated.h"

#include <ranges>

#define trace(...) fmt::println(__VA_ARGS__)
// #define trace(...)

namespace nigiri::loader::gtfs {

std::vector<utc_trip> build_seated_trips(timetable const& tt,
                                         trip_data const& trip_data,
                                         expanded_seated& seated) {
  [[maybe_unused]] auto const base = tt.internal_interval_days().from_;

  auto const is_empty = [](utc_trip const& x) {
    return x.utc_traffic_days_.none();
  };
  auto const is_not_empty = [](utc_trip const& x) {
    return x.utc_traffic_days_.any();
  };
  auto const shift = [](bitfield const& b, int const offset) {
    return offset > 0 ? b << static_cast<std::size_t>(offset)
                      : b >> static_cast<std::size_t>(-offset);
  };

  using remaining_idx_t = unsigned;
  auto& remaining = seated.expanded_.data_;
  auto const get_remaining = [&](gtfs_trip_idx_t const i) {
    return seated.expanded_.at(seated.ref_.at(i)) |
           std::views::transform([&](utc_trip const& x) {
             return static_cast<remaining_idx_t>(remaining.index_of(&x));
           });
  };
  auto const get_trp_idx = [&](remaining_idx_t const i) -> gtfs_trip_idx_t {
    return remaining[i].trips_.front();
  };
  auto const get_trp = [&](remaining_idx_t const i) -> trip const& {
    return trip_data.get(get_trp_idx(i));
  };
  auto const first_dep =
      [&](remaining_idx_t const x) -> minutes_after_midnight_t {
    return remaining[x].utc_times_.front();
  };
  auto const last_arr =
      [&](remaining_idx_t const x) -> minutes_after_midnight_t {
    return remaining[x].utc_times_.back();
  };
  auto const get_day_change_offset = [&](remaining_idx_t const a,
                                         remaining_idx_t const b) {
    auto const day_span =
        last_arr(a) / date::days{1U} - first_dep(a) / date::days{1U};
    auto const day_change = last_arr(a) % 1440 > first_dep(b) ? 1 : 0;
    return day_span + day_change;
  };

  auto q = hash_map<remaining_idx_t,
                    int /* offset relative to its traffic days */>{};
  while (!utl::all_of(remaining, is_empty)) {
    // Find first trip with unprocessed/remaining traffic days.
    auto const non_empty_it = utl::find_if(remaining, is_not_empty);
    assert(non_empty_it != end(remaining));

    // Build a "maximum component":
    // Collect all trips reachable from this trip connected by stay-seated
    // transfers from here (forward+backward, direct + transitive) while
    // building the traffic day intersection of all visited trips. Stop early if
    // the intersection would be empty.
    auto component = hash_map<remaining_idx_t, int>{};
    q.emplace(remaining.index_of(non_empty_it), 0U);
    auto component_traffic_days = non_empty_it->utc_traffic_days_;
    while (!q.empty()) {
      // Extract next queue element.
      auto const curr_it = q.begin();
      auto const [current_idx, offset] = *curr_it;
      auto& current = remaining[current_idx];
      q.erase(curr_it);

      trace("\nEXTRACT {}, offset={}", get_trp(current_idx).display_name(),
            offset);

      // Intersect traffic days.
      auto const next_traffic_days =
          shift(current.utc_traffic_days_, -offset) & component_traffic_days;
      trace(
          "      current: {}\n"
          "      shifted: {}\n"
          "    component: {}\n"
          "         next: {}",
          fmt::streamed(day_list{current.utc_traffic_days_, base}),
          fmt::streamed(
              day_list{shift(current.utc_traffic_days_, -offset), base}),
          fmt::streamed(day_list{component_traffic_days, base}),
          fmt::streamed(day_list{next_traffic_days, base}));
      if (next_traffic_days.none()) {
        trace("-> EMPTY INTERSECTION");
        continue;  // Nothing left, skip.
      }

      // Non-empty intersection!
      // Add trip to component + update component traffic days.
      trace("UPDATE: {}", fmt::streamed(day_list{next_traffic_days, base}));
      component_traffic_days = next_traffic_days;
      component.emplace(current_idx, offset);

      // Expand search to neighbors.
      for (auto const& out_trp : get_trp(current_idx).seated_out_) {
        for (auto const out : get_remaining(out_trp)) {
          if (!component.contains(out)) {
            auto const o = offset + get_day_change_offset(current_idx, out);
            trace(
                "    EXPAND OUT: {}, day_change_offset={} (curr_dep={}, "
                "curr_arr={}, next_dep={})  =>  {}",
                trip_data.get(out_trp).display_name(),
                get_day_change_offset(current_idx, out), first_dep(current_idx),
                last_arr(current_idx), first_dep(out), o);
            q.emplace(out, o);
          }
        }
      }

      for (auto const& in_trp : get_trp(current_idx).seated_in_) {
        for (auto const in : get_remaining(in_trp)) {
          if (!component.contains(in)) {
            auto const o = offset - get_day_change_offset(in, current_idx);
            trace(
                "    EXPAND IN: {}, day_change_offset={} (pred_dep={}, "
                "pred_arr={}, curr_dep={})  =>  {}",
                trip_data.get(in_trp).display_name(),
                get_day_change_offset(in, current_idx), first_dep(in),
                last_arr(current_idx), first_dep(current_idx), o);
            q.emplace(in, o);
          }
        }
      }
    }  // END while (!q.empty())

    // Handle connected component.
    trace("\n=> COMPONENT: {}",
          fmt::streamed(day_list{component_traffic_days, base}));

    auto represented_by = hash_map<gtfs_trip_idx_t, remaining_idx_t>{};
    for (auto const& [remaining_idx, offset] : component) {
      auto const [_, added] =
          represented_by.emplace(get_trp_idx(remaining_idx), remaining_idx);
      assert(added);
    }
    auto const translate_represented = [&](gtfs_trip_idx_t const i) {
      auto const it = represented_by.find(i);
      return it == end(represented_by) ? -1 : static_cast<int>(it->second);
    };

    for (auto const& [remaining_idx, offset] : component) {
      auto& before = remaining.at(remaining_idx).utc_traffic_days_;
      auto const active = before & shift(component_traffic_days, offset);
      trace(
          "  idx={}, {} [offset={}]\n"
          "    ->  before={}\n"
          "    ->  active={}\n"
          "    ->  update={}",  //
          remaining_idx, get_trp(remaining_idx).display_name(), offset,
          fmt::streamed(day_list{before, base}),
          fmt::streamed(day_list{active, base}),
          fmt::streamed(day_list{before & ~active, base}));
      trace("    -> in={} out={}",
            get_trp(remaining_idx).seated_in_ |
                std::views::transform(translate_represented) |
                std::views::filter([](int const i) { return i != -1; }),
            get_trp(remaining_idx).seated_out_ |
                std::views::transform(translate_represented) |
                std::views::filter([](int const i) { return i != -1; }));
      before &= ~active;
    }
    trace("------------\n");
  }  // END while (!utl::all_of(remaining, is_empty))

  return {};
}

}  // namespace nigiri::loader::gtfs