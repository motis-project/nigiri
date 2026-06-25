#include "nigiri/loader/permutate_timetable.h"

#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

#include <numbers>
#include <cmath>

namespace nigiri {

template <typename T>
concept Collection = requires(T& x) {
  x.begin();
  x.end();
};

template <typename T>
using permutation_t = vector_map<T, T>;

using location_permutation_t = permutation_t<location_idx_t>;
using route_permutation_t = permutation_t<route_idx_t>;
using transport_permutation_t = permutation_t<transport_idx_t>;
using cartesian_t = std::tuple<double, double, double>;

template <typename T>
void update_refs(permutation_t<T> const& p, T& l) {
  l = l == T::invalid() ? l : p[l];
}

void update_refs(location_permutation_t const& p,
                 vecvec<route_idx_t, stop::value_type>::bucket& seq) {
  for (auto& s : seq) {
    auto old = stop{s};
    s = stop{p[old.location_idx()], old.in_allowed(), old.out_allowed(),
             old.in_allowed_wheelchair(), old.out_allowed_wheelchair()}
            .value();
  }
}

void update_refs(location_permutation_t const& p, footpath& fp) {
  fp = footpath{p[fp.target()], fp.duration()};
}

template <typename T, Collection C>
void update_refs(permutation_t<T> const& p, C& c) {
  if (c.empty()) {
    return;
  }

  for (auto it = begin(c); it != end(c); ++it) {
    if constexpr (std::is_reference_v<decltype(*it)>) {
      update_refs(p, *it);
    } else {
      update_refs(p, it);
    }
  }
}

void update_refs(location_permutation_t const& p,
                 hash_map<owning_location_id,
                          location_idx_t,
                          location_id_hash,
                          location_id_equals>& map) {
  for (auto& it : map) {
    update_refs(p, it.second);
  }
}

template <typename T, typename Vec>
void permutate(permutation_t<T> const& p, Vec& vec) {
  auto copy = Vec{};
  for (auto i = T{0U}; i != vec.size(); ++i) {
    copy.emplace_back(vec[p[i]]);
  }
  vec = copy;
}

template <typename T>
void permutate_bits(permutation_t<T> const& p, bitvec& bv, unsigned int const n_bits_per_object) {
  utl::verify(bv.size() % n_bits_per_object == 0U, "unexpected bitvec dimensions" );
  const auto n_elements = bv.size() / n_bits_per_object;

  bitvec permutated(bv.size());
  for (auto i = T{0U}; i < n_elements; ++i) {
    auto const original_idx = p[i];

    auto src_base = original_idx * n_bits_per_object;
    auto dst_base = i * n_bits_per_object;

    for (auto j=0U; j < n_bits_per_object; ++j) {
      permutated.set(to_idx(dst_base) + j, bv.test(to_idx(src_base) + j));
    }
  }

  bv = std::move(permutated);
}

void permutate_route_times(route_permutation_t const& p,
                           vecvec<route_idx_t, stop::value_type> const& permutated_route_stops,
                           vector_map<route_idx_t, interval<transport_idx_t>> const& permutated_route_transport_ranges,
                           vector_map<route_idx_t, interval<std::uint32_t>>& ranges,
                           vector<delta>& times) {
  const auto n_routes = ranges.size();

  vector_map<route_idx_t, interval<uint32_t>> new_ranges;
  new_ranges.resize(n_routes);
  uint32_t offset = 0;
  for (auto new_idx = route_idx_t{0}; new_idx < n_routes; ++new_idx) {
    const auto n_stops = permutated_route_stops[new_idx].size();
    const auto n_transports = permutated_route_transport_ranges[new_idx].size();
    const auto n_events_per_route = (2 * n_stops - 2) * n_transports;
    new_ranges[new_idx] = {offset, offset + n_events_per_route};
    offset += n_events_per_route;
  }

  vector<delta> new_times;
  new_times.reserve(times.size());

  for (auto new_route_idx = route_idx_t{0U}; new_route_idx < n_routes; ++new_route_idx) {
    auto const original_route_idx = p[new_route_idx];
    auto const old_range = ranges[original_route_idx];

    auto const times_start_it = std::next(times.begin(), *old_range.begin());
    auto const times_end_it = std::next(times.begin(), *old_range.end());

    new_times.insert(new_times.end(), times_start_it, times_end_it);
  }

  utl::verify(new_times.size() == times.size(), "Unexpected times dimensions");
  ranges = std::move(new_ranges);
  times = std::move(new_times);
}



cartesian_t get_cartesian(geo::latlng const& ll) {
  static constexpr auto PI = std::numbers::pi;
  const auto lat_rad = ll.lat() * PI / 180;
  const auto lng_rad = ll.lng() * PI / 180;

  return {
    std::cos(lat_rad) * std::cos(lng_rad),
    std::cos(lat_rad) * std::sin(lng_rad),
    std::sin(lat_rad)
  };
}

vector_map<route_idx_t, geo::latlng> get_route_centroids(timetable const& tt) {
  vector_map<route_idx_t, geo::latlng> ret{};
  ret.resize(tt.n_routes());
  for (auto r = route_idx_t{0U}; r < tt.n_routes(); ++r) {
    const auto stop_seq = tt.route_location_seq_[r];
    const auto n = stop_seq.size();

    cartesian_t sum = {0.0, 0.0, 0.0};
    auto& [sum_x, sum_y, sum_z] = sum;
    for (auto i = 0U; i < n; ++i) {
      const auto stp = stop{stop_seq[i]};
      const auto loc = stp.location_idx();
      const auto p = tt.locations_.coordinates_[loc];
      const auto [x, y, z] = get_cartesian(p);
      sum_x += x;
      sum_y += y;
      sum_z += z;
    }

    const cartesian_t avg = {
      sum_x / n,
      sum_y / n,
      sum_z / n,
    };
    const auto& [centroid_x, centroid_y, centroid_z] = avg;

    static constexpr auto PI = std::numbers::pi;
    auto const centroid_lat_rad = std::atan2(
        centroid_z,
        std::sqrt(centroid_x * centroid_x + centroid_y * centroid_y));
    auto const centroid_lng_rad = std::atan2(centroid_y, centroid_x);
    ret[r] = {centroid_lat_rad * 180.0 / PI, centroid_lng_rad * 180.0 / PI};
  }

  return ret;
}

std::pair<route_permutation_t, route_permutation_t> get_route_permutation(timetable const& tt) {
  if (tt.n_routes() == 0U) {
    return {};
  }

  auto ret = std::pair<route_permutation_t, route_permutation_t>{};
  auto& [p, r] = ret;

  p.resize(tt.n_routes());
  std::generate(begin(p), end(p),
                [route = route_idx_t{0U}]() mutable { return route++; });

  const auto route_centroids = get_route_centroids(tt);


  std::stable_sort(begin(p), end(p),
                   [&](route_idx_t const a, route_idx_t const b) {
                     return morton_encode(route_centroids[a]) <
                            morton_encode(route_centroids[b]);
                   });

  r.resize(tt.n_routes());
  for (auto i = route_idx_t{0U}; i != tt.n_routes(); ++i) {
    r[p[i]] = i;
  }

  return ret;
}

std::pair<transport_permutation_t, transport_permutation_t> get_transport_permutation(vector_map<route_idx_t, interval<transport_idx_t>>& route_transport_ranges) {
  if (route_transport_ranges.empty()) {
    return {};
  }
  const auto n_routes = route_transport_ranges.size();

  auto ret = std::pair<transport_permutation_t, transport_permutation_t>{};
  auto& [p, r] = ret;

  for (auto route = route_idx_t{0U}; route < n_routes; ++route) {
    auto& range = route_transport_ranges[route];

    auto const new_from = transport_idx_t{p.size()};
    for (transport_idx_t const old_idx : range) {
      p.emplace_back(old_idx);
    }
    auto const new_to = transport_idx_t{p.size()};
    range = {new_from, new_to};
  }

  r.resize(p.size());
  for (auto i = transport_idx_t{0U}; i != p.size(); ++i) {
    r[p[i]] = i;
  }

  return ret;
}

std::pair<location_permutation_t, location_permutation_t> get_location_permutation(timetable const& tt) {
  if (tt.n_locations() == 0U) {
    return {};
  }

  auto ret = std::pair<location_permutation_t, location_permutation_t>{};
  auto& [p, r] = ret;

  p.resize(tt.n_locations());
  std::generate(begin(p), end(p),
                [l = location_idx_t{0U}]() mutable { return l++; });
  auto const first_idx = tt.n_locations() >= kNSpecialStations &&
                                 tt.get_default_name(location_idx_t{0U}) ==
                                     special_stations_names[0]
                             ? kNSpecialStations
                             : 0U;
  std::stable_sort(begin(p) + first_idx, end(p),
                   [&](location_idx_t const a, location_idx_t const b) {
                     return morton_encode(tt.locations_.coordinates_[a]) <
                            morton_encode(tt.locations_.coordinates_[b]);
                   });

  r.resize(tt.n_locations());
  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    r[p[i]] = i;
  }

  return ret;
}

void permutate_locations(timetable& tt) {
  auto timer = scoped_timer{"permutate locations"};
  for (auto i = 0U; i != kNProfiles; ++i) {
    assert(tt.locations_.footpaths_out_[i].empty());
    assert(tt.locations_.footpaths_in_[i].empty());
    assert(tt.fwd_search_lb_graph_[i].empty());
    assert(tt.bwd_search_lb_graph_[i].empty());
  }
  assert(tt.locations_.rtree_.nodes_.empty());

  auto const [p, r] = get_location_permutation(tt);

  auto& locations = tt.locations_;
  permutate(p, tt.location_areas_);
  permutate(p, tt.location_routes_);
  permutate(p, tt.location_location_groups_);
  permutate(p, locations.names_);
  permutate(p, locations.descriptions_);
  permutate(p, locations.ids_);
  permutate(p, locations.coordinates_);
  permutate(p, locations.src_);
  permutate(p, locations.transfer_time_);
  permutate(p, locations.types_);
  permutate(p, locations.parents_);
  permutate(p, locations.location_timezones_);
  permutate(p, locations.equivalences_);
  permutate(p, locations.children_);
  permutate(p, locations.preprocessing_footpaths_out_);
  permutate(p, locations.preprocessing_footpaths_in_);
  permutate(p, locations.location_importance_);

  update_refs(r, tt.route_location_seq_);
  update_refs(r, tt.location_group_locations_);
  update_refs(r, tt.flex_area_locations_);
  update_refs(r, locations.preprocessing_footpaths_out_);
  update_refs(r, locations.preprocessing_footpaths_in_);
  update_refs(r, locations.equivalences_);
  update_refs(r, locations.parents_);
  update_refs(r, locations.children_);
  update_refs(r, locations.location_id_to_idx_);
}

void permutate_routes_and_transports(timetable& tt) {
  auto timer = scoped_timer{"permutate routes and transports"};
  // ===========================
  // Permutate Routes First
  // ---------------------------
  auto const [p, r] = get_route_permutation(tt);

  permutate(p, tt.route_transport_ranges_);
  permutate(p, tt.route_location_seq_);
  permutate(p, tt.route_clasz_);
  permutate(p, tt.route_section_clasz_);
  permutate(p, tt.route_traffic_days_);

  permutate_bits(p, tt.route_bikes_allowed_, 2);
  permutate_bits(p, tt.route_cars_allowed_, 2);
  permutate_bits(p, tt.route_wheelchair_accessible_, 2);

  permutate(p, tt.route_bikes_allowed_per_section_);
  permutate(p, tt.route_cars_allowed_per_section_);
  permutate(p, tt.route_wheelchair_accessibility_per_section_);

  /*
   * tt.route_transport_ranges_
   * and tt.route_stop_time_ranges_
   * are expected to be permutated!!!
   */
  permutate_route_times(p,
                        tt.route_location_seq_,
                        tt.route_transport_ranges_,
                        tt.route_stop_time_ranges_,
                        tt.route_stop_times_);




  update_refs(r, tt.location_routes_);
  update_refs(r, tt.transport_route_);

  // ===========================
  // Given the route permutation,
  // permutate the transports
  // ---------------------------
  auto const [pt, _] = get_transport_permutation(tt.route_transport_ranges_);

  permutate(pt, tt.transport_first_dep_offset_);
  permutate(pt, tt.initial_day_offset_);
  permutate(pt, tt.transport_traffic_days_);
  permutate(pt, tt.transport_route_);
  permutate(pt, tt.transport_to_trip_section_);
  permutate(pt, tt.transport_section_attributes_);
  permutate(pt, tt.transport_section_providers_);
  permutate(pt, tt.transport_section_directions_);
}

void permutate_timetable(timetable& tt) {
  permutate_locations(tt);
  permutate_routes_and_transports(tt);
}

}  // namespace nigiri