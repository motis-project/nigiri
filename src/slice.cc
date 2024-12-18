#include "nigiri/slice.h"

#include <ranges>

#include "utl/enumerate.h"
#include "utl/get_or_create.h"

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/gtfs/route_key.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/delta_t.h"
#include "nigiri/timetable.h"

namespace nigiri {

using loader::gtfs::route_key_equals;
using loader::gtfs::route_key_hash;
using loader::gtfs::route_key_ptr_t;
using loader::gtfs::route_key_t;
using loader::gtfs::stop_seq_t;

template <typename T>
T to(auto&& range) {
  auto t = T{};
  for (auto&& x : range) {
    t.push_back(std::forward<decltype(x)>(x));
  }
  return t;
}

timetable slice(nigiri::timetable const& tt,
                interval<date::sys_days> slice_interval) {
  auto s = timetable{};
  loader::register_special_stations(s);
  s.date_range_ = slice_interval;

  // Copy basic information about trips.
  s.trip_id_to_idx_ = tt.trip_id_to_idx_;
  s.trip_ids_ = tt.trip_ids_;
  s.trip_id_strings_ = tt.trip_id_strings_;
  s.trip_id_src_ = tt.trip_id_src_;
  s.trip_train_nr_ = tt.trip_train_nr_;
  s.trip_stop_seq_numbers_ = tt.trip_stop_seq_numbers_;
  s.trip_debug_ = tt.trip_debug_;
  s.source_file_names_ = tt.source_file_names_;
  s.trip_display_names_ = tt.trip_display_names_;
  s.merged_trips_ = tt.merged_trips_;
  s.attributes_ = tt.attributes_;
  s.attribute_combinations_ = tt.attribute_combinations_;
  s.providers_ = tt.providers_;
  s.trip_direction_strings_ = tt.trip_direction_strings_;
  s.trip_directions_ = tt.trip_directions_;
  s.trip_lines_ = tt.trip_lines_;

  // ====================
  // LOCATION TRANSLATION
  // --------------------
  using l_idx_t = location_idx_t;  // l_idx for location_idx in the slice
  auto l_location = vector_map<l_idx_t, location_idx_t>{};
  auto location_l = vector_map<location_idx_t, l_idx_t>{};
  location_l.resize(tt.n_locations(), l_idx_t::invalid());

  auto const get_or_create_l = [&](location_idx_t const x) {
    if (location_l[x] == l_idx_t::invalid()) {
      location_l[x] = s.locations_.register_location(tt.locations_.get(x));
      l_location.push_back(x);
    }
    return location_l[x];
  };

  auto const get_or_create_stop =
      [&](stop::value_type const v) -> stop::value_type {
    auto const orig = stop{v};
    return stop{get_or_create_l(orig.location_idx()), orig.in_allowed(),
                orig.out_allowed(), orig.in_allowed_wheelchair(),
                orig.out_allowed_wheelchair()}
        .value();
  };

  // ============
  // BUILD ROUTES
  // ------------
  auto const get_transport_time = [&](route_idx_t const r, transport const t,
                                      std::size_t const ev_idx) {
    // Example (d = departure, a = arrival)
    //   ev idx: 0 | 1 2 | 3 4 | 5
    //  ev type: d | a d | a d | a
    // stop idx: 0 |  1  |  2  | 3
    auto const ev_type = ev_idx % 2 == 0 ? event_type::kDep : event_type::kArr;
    auto const stop_idx = static_cast<stop_idx_t>((ev_idx + 1U) / 2U);
    auto const ev_time = tt.event_time(r, t, stop_idx, ev_type);
    return unix_to_delta(s.internal_interval_days().from_, ev_time);
  };

  auto const get_n_events = [&](route_idx_t const r) {
    auto const n_stops = tt.route_location_seq_[r].size();
    return static_cast<unsigned>(n_stops * 2U - 2U);
  };

  auto const get_event_times = [&](transport const t) {
    auto const r = tt.transport_route_[t.t_idx_];
    return std::views::iota(0U, get_n_events(r))  //
           | std::views::transform([&, r, t](unsigned const ev_idx) {
               return get_transport_time(r, t, ev_idx);
             });
  };

  auto const stays_ordered = [&](auto&& a, auto&& b) {
    return std::ranges::all_of(std::views::zip(a, b),
                               [](auto&& x) { return get<0>(x) <= get<1>(x); });
  };

  using tmp_route_idx_t = cista::strong<std::uint32_t, struct tmp_route_idx_>;
  auto const get_index = [&](paged_vecvec<tmp_route_idx_t, transport>::bucket r,
                             transport const t) {
    auto const index = static_cast<unsigned>(std::distance(
        begin(r),
        std::lower_bound(begin(r), end(r), t,
                         [&](transport const a, transport const b) {
                           return std::ranges::lexicographical_compare(
                               get_event_times(a), get_event_times(b));
                         })));
    return (stays_ordered(get_event_times(r[index - 1U]), get_event_times(t)) &&
            stays_ordered(get_event_times(t), get_event_times(r[index])))
               ? std::optional{index}
               : std::nullopt;
  };

  auto tmp_routes = paged_vecvec<tmp_route_idx_t, transport>{};
  auto const add_transport = [&](std::vector<tmp_route_idx_t>& r_candidates,
                                 transport const tr) {
    for (auto const& r : r_candidates) {
      auto transports = tmp_routes[r];
      auto const index = get_index(transports, tr);
      if (index.has_value()) {
        transports.insert(begin(transports) + *index, tr);
        return;
      }
    }
    tmp_routes.emplace_back({tr});  // no fitting route found - create new
  };

  auto const get_bikes_allowed_seq = [&](route_idx_t const r) -> bitvec {
    auto const section_bikes_allowed = tt.route_bikes_allowed_per_section_[r];
    auto ret = bitvec{};
    ret.resize(section_bikes_allowed.size());
    for (auto const [i, x] : utl::enumerate(section_bikes_allowed)) {
      ret.set(static_cast<unsigned>(i), x);
    }
    return ret;
  };

  auto stop_seq_tmp_routes = hash_map<route_key_t, std::vector<tmp_route_idx_t>,
                                      route_key_hash, route_key_equals>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const route_key = route_key_t{
        tt.route_clasz_[r],
        to<stop_seq_t>(tt.route_location_seq_[r].view() |
                       std::views::transform([&](stop::value_type const stp) {
                         return get_or_create_stop(stp);
                       })),
        get_bikes_allowed_seq(r)};
    auto& r_candidates = stop_seq_tmp_routes[route_key];
    auto const last_stop_idx =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size() - 1U);
    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const day_extend =
          tt.event_mam(r, t, last_stop_idx, event_type::kArr).days_;
      auto const first_day_interval = interval{
          tt.day_idx(slice_interval.from_ - day_extend * date::days{1}),
          tt.day_idx(slice_interval.to_)};
      for (auto const day : first_day_interval) {
        if (tt.bitfields_[tt.transport_traffic_days_[t]].test(to_idx(day))) {
          add_transport(r_candidates, transport{t, day});
        }
      }
    }
  }

  // ===========
  // COPY ROUTES
  // -----------
  auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
  for (auto const& [key, routes] : stop_seq_tmp_routes) {
    for (auto const tmp_r : routes) {
      auto const route_idx =
          s.register_route(key.stop_seq_, {key.clasz_}, key.bikes_allowed_);

      for (auto const stp : key.stop_seq_) {
        auto s_routes = location_routes[stop{stp}.location_idx()];
        if (s_routes.empty() || s_routes.back() != route_idx) {
          s_routes.emplace_back(route_idx);
        }
      }

      for (auto const t : tmp_routes[tmp_r]) {
        auto const transport_idx = s.add_transport(
            {.bitfield_idx_ = bitfield_idx_t{0U},
             .route_idx_ = route_idx,
             .first_dep_offset_ = tt.transport_first_dep_offset_[t.t_idx_],
             .external_trip_ids_ =
                 tt.transport_to_trip_section_[t.t_idx_].view(),
             .section_attributes_ =
                 tt.transport_section_attributes_[t.t_idx_].view(),
             .section_providers_ =
                 tt.transport_section_providers_[t.t_idx_].view(),
             .section_directions_ =
                 tt.transport_section_directions_[t.t_idx_].view(),
             .section_lines_ = tt.transport_section_lines_[t.t_idx_].view(),
             .route_colors_ =
                 tt.transport_section_route_colors_[t.t_idx_].view()});

        using it_t =
            vecvec<transport_idx_t, merged_trips_idx_t>::const_bucket::iterator;
        auto const section_trips = tt.transport_to_trip_section_[t.t_idx_];
        if (section_trips.size() == 1U) {
          auto const trip = tt.merged_trips_[section_trips.front()].front();
          auto const n_stops =
              static_cast<stop_idx_t>(key.stop_seq_.size() - 1U);
          s.trip_transport_ranges_[trip].push_back(
              {transport_idx, interval{stop_idx_t{0U}, n_stops}});
        } else {
          auto const start = begin(tt.transport_to_trip_section_[t.t_idx_]);
          utl::equal_ranges_linear(
              tt.transport_to_trip_section_[t.t_idx_],
              [](merged_trips_idx_t const a, merged_trips_idx_t const b) {
                return a == b;
              },
              [&](it_t const from, it_t const to) {
                // example: sections 0, 1, 2    = [0, 3[
                //        = stops    0, 1, 2, 3 = [0, 4[
                auto const from_stop_idx =
                    static_cast<stop_idx_t>(from - start);
                auto const to_stop_idx =
                    static_cast<stop_idx_t>(to - start + 1U);
                for (auto const trip : tt.merged_trips_[*from]) {
                  s.trip_transport_ranges_[trip].push_back(
                      {transport_idx, {from_stop_idx, to_stop_idx}});
                }
              });
        }
      }

      s.finish_route();

      auto const stop_times_begin = s.route_stop_times_.size();
      for (auto const [from, to] :
           utl::pairwise(interval{std::size_t{0U}, key.stop_seq_.size()})) {
        // Write departure times of all route services at stop i.
        for (auto const t : tmp_routes[tmp_r]) {
          s.route_stop_times_.emplace_back(
              get_transport_time(tt.transport_route_[t.t_idx_], t, from * 2));
        }

        // Write arrival times of all route services at stop i+1.
        for (auto const t : tmp_routes[tmp_r]) {
          s.route_stop_times_.emplace_back(
              get_transport_time(tt.transport_route_[t.t_idx_], t, to * 2 - 1));
        }
      }
      auto const stop_times_end = s.route_stop_times_.size();
      s.route_stop_time_ranges_.emplace_back(
          interval{stop_times_begin, stop_times_end});
    }
  }

  // Build location_routes map.
  for (auto l = tt.location_routes_.size(); l != tt.n_locations(); ++l) {
    s.location_routes_.emplace_back(location_routes[location_idx_t{l}]);
    assert(tt.location_routes_.size() == l + 1U);
  }

  // =========
  // FOOTPATHS
  // ---------
  auto const filter_and_translate_footpaths =
      [&](vecvec<location_idx_t, footpath> const& fps) {
        auto ret = vecvec<l_idx_t, footpath>{};
        for (auto l = l_idx_t{0U}; l != l_location.size(); ++l) {
          using namespace std::views;
          ret.emplace_back(fps[l_location[l]]  //
                           | drop_while([&](footpath const x) {
                               return location_l[x.target()] !=
                                      l_idx_t::invalid();
                             })  //
                           | transform([&](footpath const x) -> footpath {
                               return {location_l[x.target()], x.duration()};
                             }));
        }
        return ret;
      };
  for (auto [full, filtered] :
       utl::zip(tt.locations_.footpaths_out_, s.locations_.footpaths_out_)) {
    filtered = filter_and_translate_footpaths(full);
  }
  for (auto [full, filtered] :
       utl::zip(tt.locations_.footpaths_in_, s.locations_.footpaths_in_)) {
    filtered = filter_and_translate_footpaths(full);
  }

  // ========
  // FINALIZE
  // --------
  loader::build_lb_graph<direction::kForward>(s);
  loader::build_lb_graph<direction::kBackward>(s);
  s.location_routes_.resize(s.n_locations());

  return s;
}

}  // namespace nigiri