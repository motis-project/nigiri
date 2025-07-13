#include "nigiri/loader/permutate_locations.h"

#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri {

template <typename T>
concept Collection = requires(T& x) {
  x.begin();
  x.end();
};

using permutation_t = vector_map<location_idx_t, location_idx_t>;

void update_refs(permutation_t const& p, location_idx_t& l) {
  l = l == location_idx_t::invalid() ? l : p[l];
}

void update_refs(permutation_t const& p,
                 vecvec<route_idx_t, stop::value_type>::bucket& seq) {
  for (auto& s : seq) {
    auto old = stop{s};
    s = stop{p[old.location_idx()], old.in_allowed(), old.out_allowed(),
             old.in_allowed_wheelchair(), old.out_allowed_wheelchair()}
            .value();
  }
}

void update_refs(permutation_t const& p, footpath& fp) {
  fp = footpath{p[fp.target()], fp.duration()};
}

template <Collection T>
void update_refs(permutation_t const& p, T& c) {
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

void update_refs(permutation_t const& p,
                 hash_map<location_id, location_idx_t>& map) {
  for (auto& it : map) {
    update_refs(p, it.second);
  }
}

template <typename Vec>
void permutate(permutation_t const& p, Vec& vec) {
  auto copy = Vec{};
  for (auto i = location_idx_t{0U}; i != vec.size(); ++i) {
    copy.emplace_back(vec[p[i]]);
  }
  vec = copy;
}

std::pair<permutation_t, permutation_t> get_permutation(timetable const& tt) {
  if (tt.n_locations() == 0U) {
    return {};
  }

  auto ret = std::pair<permutation_t, permutation_t>{};
  auto& [p, r] = ret;

  p.resize(tt.n_locations());
  std::generate(begin(p), end(p),
                [l = location_idx_t{0U}]() mutable { return l++; });
  auto const first_idx =
      tt.n_locations() >= kNSpecialStations &&
              tt.locations_.names_.at(location_idx_t{0U}).view() ==
                  special_stations_names[0]
          ? kNSpecialStations
          : 0U;
  std::stable_sort(begin(p) + first_idx, end(p),
                   [&](location_idx_t const a, location_idx_t const b) {
                     return !tt.location_routes_[a].empty() &&
                            tt.location_routes_[b].empty();
                   });

  r.resize(tt.n_locations());
  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    r[p[i]] = i;
  }

  return ret;
}

void permutate_locations(timetable& tt) {
  for (auto i = 0U; i != kNProfiles; ++i) {
    assert(tt.locations_.footpaths_out_[i].empty());
    assert(tt.locations_.footpaths_in_[i].empty());
    assert(tt.fwd_search_lb_graph_[i].empty());
    assert(tt.bwd_search_lb_graph_[i].empty());
  }
  assert(tt.locations_.rtree_.nodes_.empty());

  auto const [p, r] = get_permutation(tt);

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

}  // namespace nigiri