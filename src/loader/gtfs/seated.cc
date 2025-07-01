#include "nigiri/loader/gtfs/seated.h"

#include <ranges>

// #define trace(...) fmt::println(__VA_ARGS__)
#define trace(...)

namespace nigiri::loader::gtfs {

void build_seated_trips(timetable& tt,
                        trip_data& trip_data,
                        expanded_seated& seated,
                        std::function<void(utc_trip&&)> const& consumer) {
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
  auto const get_trp = [&](remaining_idx_t const i) -> trip& {
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

  auto combinations = std::vector<utc_trip>{};
  while (!utl::all_of(remaining, is_empty)) {
    // Find first trip with unprocessed/remaining traffic days.
    auto const non_empty_it = utl::find_if(remaining, is_not_empty);
    assert(non_empty_it != end(remaining));
    trace("ORIGIN {}",
          trip_data.get(non_empty_it->trips_.front()).display_name());

    // ===============================
    // PART 1: Find maximum component.
    // -------------------------------
    auto component = hash_map<remaining_idx_t, int>{};
    auto component_traffic_days = non_empty_it->utc_traffic_days_;
    {
      // Collect all trips reachable from this trip connected by stay-seated
      // transfers from here (forward+backward, direct + transitive) while
      // building the traffic day intersection of all visited trips. Stop early
      // if the intersection would be empty.
      auto q = hash_map<remaining_idx_t,
                        int /* offset relative to its traffic days */>{};
      q.emplace(static_cast<remaining_idx_t>(remaining.index_of(non_empty_it)),
                0U);
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
            day_list{current.utc_traffic_days_, base},
            day_list{shift(current.utc_traffic_days_, -offset), base},
            day_list{component_traffic_days, base},
            day_list{next_traffic_days, base});
        if (next_traffic_days.none()) {
          trace("-> EMPTY INTERSECTION");
          continue;  // Nothing left, skip.
        }

        // Non-empty intersection!
        // Add trip to component + update component traffic days.
        trace("UPDATE: {}", day_list{next_traffic_days, base});
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
                  get_day_change_offset(current_idx, out),
                  first_dep(current_idx), last_arr(current_idx), first_dep(out),
                  o);
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
    }

    // =========================
    // PART 2: Queue all starts.
    // -------------------------
    trace("\n=> COMPONENT: {}", day_list{component_traffic_days, base});

    // Finds the remaining_idx
    // that represents a given gtfs_trip_idx in this specific component.
    auto const get_representative =
        [&](gtfs_trip_idx_t const t) -> std::optional<remaining_idx_t> {
      auto const it = utl::find_if(
          component, [&](std::pair<remaining_idx_t, int> const& x) {
            return remaining[x.first].trips_.front() == t;
          });
      return it == end(component) ? std::nullopt : std::optional{it->first};
    };

    // Initialize queue with all remaining_idx that do not have any incoming
    // seated transfers *in this component*.
    auto q = std::vector<
        std::pair<utc_trip /* concatenated transport */,
                  int /* transport offset relative to component */>>{};
    for (auto const& [remaining_idx, offset] : component) {
      [[maybe_unused]] auto const& trp = get_trp(remaining_idx);
      trace("  -> {}: {} on {}", trp.display_name(), offset,
            day_list{shift(component_traffic_days, offset), base});
      auto const is_entry = std::ranges::empty(
          get_trp(remaining_idx).seated_in_ |
          std::views::transform(
              [&](gtfs_trip_idx_t const t) { return get_representative(t); }) |
          std::views::filter([](std::optional<remaining_idx_t> const& r) {
            return r.has_value();
          }));
      if (is_entry) {
        auto const transport_traffic_days =
            shift(component_traffic_days, offset);
        assert((remaining.at(remaining_idx).utc_traffic_days_ &
                transport_traffic_days) == transport_traffic_days);
        trace("    -> PUSH {}, on={} (transport_offset={})", trp.display_name(),
              day_list{transport_traffic_days, base}, -offset);
        auto start = remaining.at(remaining_idx);
        start.utc_traffic_days_ = transport_traffic_days;
        start.stop_seq_ = get_trp(remaining_idx).stop_seq_;
        q.emplace_back(std::move(start), -offset);
      }
    }

    // ============
    // PART 4: DFS.
    // ------------
    while (!q.empty()) {
      auto [curr, transport_offset] = std::move(q.back());
      q.resize(q.size() - 1U);

      // Expand search.
      auto has_next = false;
      for (auto const& next : trip_data.get(curr.trips_.back()).seated_out_) {
        auto const r = get_representative(next);
        if (r.has_value()) {
          auto copy = curr;
          auto const& next_r = remaining.at(*r);
          auto const next_stop_seq = get_trp(*r).stop_seq_;
          auto next_times = next_r.utc_times_;
          for (auto& t : next_times) {
            t += transport_offset * date::days{1};
          }
          copy.trips_.push_back(get_trp_idx(*r));
          copy.utc_times_.insert(end(copy.utc_times_), begin(next_times),
                                 end(next_times));
          copy.stop_seq_.insert(end(copy.stop_seq_),
                                std::next(begin(next_stop_seq)),
                                end(next_stop_seq));
          q.emplace_back(std::move(copy), transport_offset);
          has_next = true;
        }
      }

      // No outgoing seated-transfer *in this component*.
      if (!has_next) {
        trace("adding trips={}, stops={}, times={}",
              curr.trips_ | std::views::transform([&](gtfs_trip_idx_t const t) {
                return trip_data.get(t).display_name();
              }),
              curr.stop_seq_ |
                  std::views::transform([&](stop::value_type const& s) {
                    return location{tt, stop{s}.location_idx()};
                  }),
              curr.utc_times_ | std::views::transform(std::identity{}));
        consumer(std::move(curr));
      }
    }

    // ===========================
    // PART 5: Update Traffic Days
    // ---------------------------
    for (auto const& [remaining_idx, offset] : component) {
      remaining.at(remaining_idx).utc_traffic_days_ &=
          ~shift(component_traffic_days, offset);
    }

    trace("------------\n");
  }  // END while (!utl::all_of(remaining, is_empty))
}

}  // namespace nigiri::loader::gtfs
