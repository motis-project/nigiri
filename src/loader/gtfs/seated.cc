#include "nigiri/loader/gtfs/seated.h"

#include <ranges>

// #define trace(...) fmt::println(__VA_ARGS__)
#define trace(...)

namespace nigiri::loader::gtfs {

std::vector<utc_trip> build_seated_trips(
    timetable& tt,
    hash_map<bitfield, bitfield_idx_t>& bitfield_indices,
    trip_data& trip_data,
    expanded_seated& seated,
    mutable_fws_multimap<location_idx_t, route_idx_t>& location_routes) {
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

  auto q = hash_map<remaining_idx_t,
                    int /* offset relative to its traffic days */>{};
  while (!utl::all_of(remaining, is_empty)) {
    // ==============================
    // PART 1: find maximum component
    // ------------------------------

    // Find first trip with unprocessed/remaining traffic days.
    auto const non_empty_it = utl::find_if(remaining, is_not_empty);
    assert(non_empty_it != end(remaining));

    // Build a "maximum component":
    // Collect all trips reachable from this trip connected by stay-seated
    // transfers from here (forward+backward, direct + transitive) while
    // building the traffic day intersection of all visited trips. Stop early if
    // the intersection would be empty.
    auto component = hash_map<remaining_idx_t, int>{};
    q.emplace(static_cast<remaining_idx_t>(remaining.index_of(non_empty_it)),
              0U);
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

    // ========================
    // PART 2: Handle component
    // ------------------------
    trace("\n=> COMPONENT: {}",
          fmt::streamed(day_list{component_traffic_days, base}));

    struct represented {
      remaining_idx_t remaining_idx_;
      route_idx_t route_;
    };
    auto represented_by = hash_map<gtfs_trip_idx_t, represented>{};
    for (auto const& [remaining_idx, offset] : component) {
      [[maybe_unused]] auto const [_, added] = represented_by.emplace(
          get_trp_idx(remaining_idx),
          represented{remaining_idx, route_idx_t::invalid()});
      assert(added);
    }
    auto const translate_represented = [&](gtfs_trip_idx_t const i) {
      auto const it = represented_by.find(i);
      return it == end(represented_by) ? std::nullopt
                                       : std::optional{it->second};
    };

    for (auto const& [remaining_idx, offset] : component) {
      auto& trp = get_trp(remaining_idx);
      auto& utc_trp = remaining.at(remaining_idx);
      auto& before = utc_trp.utc_traffic_days_;
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
      trace(
          "    -> in={} out={}",
          get_trp(remaining_idx).seated_in_ |
              std::views::transform(translate_represented) |
              std::views::filter([](auto&& x) { return x.has_value(); }) |
              std::views::transform([](auto&& x) { return x->remaining_idx_; }),
          get_trp(remaining_idx).seated_out_ |
              std::views::transform(translate_represented) |
              std::views::filter([](auto&& x) { return x.has_value(); }) |
              std::views::transform(
                  [](auto&& x) { return x->remaining_idx_; }));

      auto const merged_trip = tt.register_merged_trip({trp.trip_idx_});
      auto const route_idx =
          tt.register_route(trp.stop_seq_, {trp.get_clasz(tt)},
                            trp.bikes_allowed_ ? kSingleTripBikesAllowed
                                               : kSingleTripBikesNotAllowed,
                            trp.cars_allowed_ ? kSingleTripBikesAllowed
                                              : kSingleTripBikesNotAllowed);

      for (auto const& s : trp.stop_seq_) {
        auto s_routes = location_routes[stop{s}.location_idx()];
        if (s_routes.empty() || s_routes.back() != route_idx) {
          s_routes.emplace_back(route_idx);
        }
      }

      trp.transport_ranges_.emplace_back(transport_range_t{
          tt.next_transport_idx(),
          {stop_idx_t{0U}, static_cast<stop_idx_t>(trp.stop_seq_.size())}});

      tt.add_transport(timetable::transport{
          .bitfield_idx_ = utl::get_or_create(
              bitfield_indices, active,
              [&]() { return tt.register_bitfield(active); }),
          .route_idx_ = route_idx,
          .first_dep_offset_ = {utc_trp.first_dep_offset_, utc_trp.tz_offset_},
          .external_trip_ids_ = {merged_trip},
          .section_attributes_ = {},
          .section_providers_ = {trp.route_->agency_},
          .section_directions_ = {trp.headsign_},
          .route_colors_ = {{trp.route_->color_, trp.route_->text_color_}}});
      tt.finish_route();

      auto const stop_times_begin = tt.route_stop_times_.size();
      for (auto const [from, to] :
           utl::pairwise(interval{std::size_t{0U}, trp.stop_seq_.size()})) {
        tt.route_stop_times_.emplace_back(utc_trp.utc_times_[from * 2]);
        tt.route_stop_times_.emplace_back(utc_trp.utc_times_[to * 2 - 1]);
      }
      auto const stop_times_end = tt.route_stop_times_.size();
      tt.route_stop_time_ranges_.emplace_back(
          interval{stop_times_begin, stop_times_end});

      represented_by.at(get_trp_idx(remaining_idx)).route_ = route_idx;

      before &= ~active;
    }

    for (auto const& [remaining_idx, offset] : component) {
      auto const route_idx =
          translate_represented(get_trp_idx(remaining_idx)).value().route_;
      for (auto const& out_trp : get_trp(remaining_idx).seated_out_) {
        auto const out = translate_represented(out_trp);
        if (out.has_value()) {
          auto const diff = component.at(out->remaining_idx_) - offset;
          tt.route_has_seated_out_.set(to_idx(route_idx), true);
          tt.route_has_seated_in_.set(to_idx(out->route_), true);
          tt.route_seated_transfers_out_[route_idx].push_back(
              seated_transfer{out->route_, static_cast<std::int8_t>(diff)}
                  .value());
          tt.route_seated_transfers_in_[out->route_].push_back(
              seated_transfer{route_idx, static_cast<std::int8_t>(diff)}
                  .value());
        }
      }
    }

    trace("------------\n");
  }  // END while (!utl::all_of(remaining, is_empty))

  return {};
}

}  // namespace nigiri::loader::gtfs