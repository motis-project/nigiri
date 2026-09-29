#include "nigiri/loader/hotel_trains.h"

#include <map>
#include <span>
#include <vector>

#include "fmt/core.h"

#include "geo/point_rtree.h"

#include "nigiri/loader/import_context.h"
#include "nigiri/loader/register.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

namespace {

bool is_rail_or_coach(clasz const c) {
  switch (c) {
    case clasz::kHighSpeed:
    case clasz::kLongDistance:
    case clasz::kNight:
    case clasz::kCoach:
    case clasz::kRegional:
    case clasz::kSuburban: return true;
    default: return false;
  }
}

// Before finalize(): the children are in the import context, and neither they
// nor location_routes_ need to cover every location yet.
std::span<route_idx_t const> routes_at(timetable const& tt,
                                       location_idx_t const l) {
  if (to_idx(l) >= tt.location_routes_.size()) {
    return {};
  }
  auto const routes = tt.location_routes_[l];
  return {routes.begin(), routes.end()};
}

std::vector<location_idx_t> children(import_context& ctx,
                                     location_idx_t const l) {
  if (to_idx(l) >= ctx.children_.size()) {
    return {};
  }
  auto const c = ctx.children_[l];
  return {c.begin(), c.end()};
}

bool served_by_rail_or_coach(timetable const& tt,
                             import_context& ctx,
                             location_idx_t const l) {
  auto const served = [&](location_idx_t const x) {
    return utl::any_of(routes_at(tt, x), [&](route_idx_t const r) {
      return is_rail_or_coach(tt.route_clasz_[r]);
    });
  };
  return served(l) || utl::any_of(children(ctx, l), served);
}

// station timezone; like run_stop::get_tz, falls back to the timezone of a
// provider serving it
timezone_idx_t find_tz(timetable const& tt,
                       import_context& ctx,
                       location_idx_t const l) {
  auto const valid = [](timezone_idx_t const tz) {
    return tz != timezone_idx_t::invalid();
  };
  auto const of = [&](location_idx_t const x) {
    if (valid(tt.locations_.location_timezones_[x])) {
      return tt.locations_.location_timezones_[x];
    }
    for (auto const r : routes_at(tt, x)) {
      for (auto const t : tt.route_transport_ranges_[r]) {
        for (auto const p : tt.transport_section_providers_[t]) {
          if (valid(tt.providers_[p].tz_)) {
            return tt.providers_[p].tz_;
          }
        }
      }
    }
    return timezone_idx_t::invalid();
  };
  if (auto const tz = of(l); valid(tz)) {
    return tz;
  }
  for (auto const c : children(ctx, l)) {
    if (auto const tz = of(c); valid(tz)) {
      return tz;
    }
  }
  return timezone_idx_t::invalid();
}

i32_minutes utc_offset(timezone const& tz, unixtime_t const utc) {
  return to_local_time(tz, utc).time_since_epoch() - utc.time_since_epoch();
}

// UTC departure of the 22:00 local departure on `day`. The offset is taken
// at the resulting instant; the one day a year where it changes is
// intentionally not handled.
unixtime_t hotel_departure(timezone const& tz, date::sys_days const day) {
  auto const local = unixtime_t{i32_minutes{day.time_since_epoch()} +
                                i32_minutes{kHotelDeparture}};
  auto const guess = local - utc_offset(tz, local);
  return local - utc_offset(tz, guess);
}

}  // namespace

void add_hotel_night_trains(timetable& tt,
                            import_context& ctx,
                            std::span<geo::latlng const> hotels) {
  if (hotels.empty()) {
    return;
  }

  auto const hotel_rtree = geo::make_point_rtree(hotels);
  auto stations = std::vector<location_idx_t>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.src_[l] != source_idx_t::invalid() &&
        tt.locations_.parents_[l] == location_idx_t::invalid() &&
        find_tz(tt, ctx, l) != timezone_idx_t::invalid() &&
        served_by_rail_or_coach(tt, ctx, l) &&
        !hotel_rtree.in_radius(tt.locations_.coordinates_[l], kHotelRadius)
             .empty()) {
      stations.push_back(l);
    }
  }
  if (stations.empty()) {
    return;
  }

  auto const src = source_idx_t{tt.n_sources_++};
  tt.route_ids_.resize(to_idx(src) + 1U);

  auto const name = tt.register_translation(std::string_view{"Hotel"});
  auto provider_of = hash_map<timezone_idx_t, provider_idx_t>{};
  auto const source_file = tt.register_source_file("hotel");
  auto const route_flags = std::array<bitvec, kNumRouteFlags>{
      bitvec{}, bitvec{}, bitvec{}, bitvec{"1"}};
  auto trip_flags = std::array<bool, kNumRouteFlags>{};

  auto const internal = tt.internal_interval_days();
  auto const n_days = (internal.to_ - internal.from_).count();

  auto station_route = hash_map<location_idx_t, route_idx_t>{};
  auto const merged_trips_of = [&](trip_idx_t const t) {
    return tt.register_merged_trip({t});
  };
  auto providers = basic_string<provider_idx_t>{};
  auto const no_attributes = basic_string<attribute_combination_idx_t>{};
  auto const no_directions = basic_string<translation_idx_t>{};

  for (auto const l : stations) {
    auto const tz_idx = find_tz(tt, ctx, l);
    auto const& tz = tt.timezones_[tz_idx];
    auto const [provider_it, is_new] = provider_of.try_emplace(tz_idx);
    if (is_new) {
      provider_it->second = register_agency(
          tt, agency{tt, src, fmt::format("hotel/{}", to_idx(tz_idx)), name,
                     kEmptyTranslation, kEmptyTranslation, tz_idx});
    }
    providers.clear();
    providers.push_back(provider_it->second);

    // departure minute of day (UTC) -> traffic days; one entry per offset
    // (winter/summer time)
    auto variants = std::map<int, bitfield>{};
    for (auto d = tt.date_range_.from_; d != tt.date_range_.to_;
         d += std::chrono::days{1}) {
      auto const dep = hotel_departure(tz, d);
      auto const dep_day = std::chrono::floor<std::chrono::days>(dep);
      auto const idx = (dep_day - internal.from_).count();
      if (idx >= 0 && idx < n_days) {
        variants[(dep - dep_day).count()].set(static_cast<std::size_t>(idx));
      }
    }

    auto const stop_seq =
        basic_string<stop::value_type>{stop{l, true, true, true, true}.value(),
                                       stop{l, true, true, true, true}.value()};
    auto const route = tt.register_route(
        stop_seq, basic_string<clasz>{clasz::kNight}, route_flags);
    station_route.emplace(l, route);

    auto dep_times = std::vector<delta>{};
    auto arr_times = std::vector<delta>{};
    for (auto const& [dep_mam, traffic_days] : variants) {
      auto const trip_id =
          fmt::format("hotel/{}/{}", tt.locations_.ids_[l].view(), dep_mam);
      auto const trip = register_trip(
          tt, ctx,
          loader::trip{tt, src, trip_id, kEmptyTranslation, name, name, "", "",
                       direction_id_t::invalid(), route_id_idx_t::invalid(),
                       trip_flags, trip_debug{source_file, 0U, 0U}});
      ctx.trip_stop_seq_numbers_.emplace_back(
          ctx.trip_stop_seq_numbers_cache_,
          std::initializer_list<stop_idx_t>{});
      ctx.trip_transport_ranges_[trip].emplace_back(transport_range_t{
          tt.next_transport_idx(), interval<stop_idx_t>{0U, 2U}});

      auto const arr = dep_mam + kHotelDuration.count();
      dep_times.emplace_back(0U, static_cast<std::uint16_t>(dep_mam));
      arr_times.emplace_back(static_cast<std::uint16_t>(arr / 1440),
                             static_cast<std::uint16_t>(arr % 1440));

      auto merged = basic_string<merged_trips_idx_t>{};
      merged.push_back(merged_trips_of(trip));
      add_transport(tt, ctx,
                    timetable::transport{.bitfield_idx_ = register_bitfield(
                                             tt, ctx, traffic_days),
                                         .route_idx_ = route,
                                         .first_dep_offset_ = {0U, 0U},
                                         .external_trip_ids_ = merged,
                                         .section_attributes_ = no_attributes,
                                         .section_providers_ = providers,
                                         .section_directions_ = no_directions});
    }
    tt.finish_route();

    auto const begin = ctx.route_stop_times_.size();
    for (auto const t : dep_times) {
      ctx.route_stop_times_.emplace_back(t);
    }
    for (auto const t : arr_times) {
      ctx.route_stop_times_.emplace_back(t);
    }
    ctx.route_stop_time_ranges_.emplace_back(
        interval{begin, ctx.route_stop_times_.size()});
  }

  auto location_routes = decltype(tt.location_routes_){};
  auto routes = std::vector<route_idx_t>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    routes.clear();
    if (l < tt.location_routes_.size()) {
      for (auto const r : tt.location_routes_[l]) {
        routes.push_back(r);
      }
    }
    if (auto const it = station_route.find(l); it != end(station_route)) {
      routes.push_back(it->second);
    }
    location_routes.emplace_back(routes);
  }
  tt.location_routes_ = std::move(location_routes);

  if (ctx.trip_ticketing_identifier_.size() != 0U) {
    ctx.trip_ticketing_identifier_.resize(tt.n_trips());
  }
  if (tt.trip_ticketing_unavailable_.size() != 0U) {
    tt.trip_ticketing_unavailable_.resize(tt.n_trips());
  }

  log(log_lvl::info, "nigiri.loader.hotel_trains",
      "{} stations with a hotel nearby, {} hotels", stations.size(),
      hotels.size());
}

}  // namespace nigiri::loader
