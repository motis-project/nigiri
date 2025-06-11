#include "nigiri/loader/permutate_locations.h"
#include "nigiri/special_stations.h"

namespace nigiri {

vector<location_idx_t> create_mapping_vec(
    vector<location_idx_t> const& permutation) {
  auto n = permutation.size();
  auto result = std::vector<location_idx_t>{};
  result.resize(n);
  for (auto i = 0U; i < n; ++i) {
    result[permutation[i].v_] = location_idx_t{i};
  }
  auto mapping_vec = vector<location_idx_t>{};
  for (auto val : result) {
    mapping_vec.push_back(val);
  }
  return mapping_vec;
}

vector<location_idx_t> build_permutation_vec(
    vecvec<location_idx_t, route_idx_t> const& order,
    std::size_t const first_idx) {
  auto permutation = vector<location_idx_t>{};
  permutation.resize(order.size());
  for (auto i = 0U; i != permutation.size(); ++i) {
    permutation[i] = location_idx_t{i};
  }
  std::stable_sort(begin(permutation) + first_idx, end(permutation),
                   [&](auto&& a, auto&& b) {
                     return order.at(location_idx_t{a}).size() > 0 &&
                            order.at(location_idx_t{b}).size() == 0;
                   });
  return permutation;
}

template <typename T>
inline T apply_permutation_vec(T const& input, auto& permutation) {
  auto sorted = T{};
  for (auto i = 0U; i < input.size(); ++i) {
    auto temp = input[permutation[i]];
    sorted.emplace_back(temp);
  }
  return sorted;
}

vecvec<route_idx_t, stop::value_type> apply_permutation_to_route_loc_seq(
    vecvec<route_idx_t, stop::value_type> const& input,
    vector<location_idx_t> const& mapping_vec) {
  auto sorted = vecvec<route_idx_t, stop::value_type>{};
  for (auto j = 0U; j < input.size(); ++j) {
    auto new_stop_v_types = vector<stop::value_type>{};
    auto loc_seq = input.at(route_idx_t{j});
    for (auto s : loc_seq) {
      auto const old_stop = stop{s};
      auto stop_loc_idx = old_stop.location_idx();
      auto new_idx = mapping_vec.at(stop_loc_idx.v_);
      stop new_stop = {new_idx, old_stop.in_allowed(), old_stop.out_allowed(),
                       old_stop.in_allowed_wheelchair(),
                       old_stop.out_allowed_wheelchair()};
      new_stop_v_types.emplace_back(new_stop.value());
    }
    sorted.emplace_back(new_stop_v_types);
  }
  return sorted;
}

vector_map<location_idx_t, location_idx_t> apply_permutation_and_mapping_vec(
    vector_map<location_idx_t, location_idx_t> const& input,
    vector<location_idx_t> const& permutation,
    vector<location_idx_t> const& mapping_vec) {
  auto sorted = vector_map<location_idx_t, location_idx_t>{};
  for (auto i = 0U; i < input.size(); ++i) {
    auto temp = input[permutation[i]];
    if (temp == std::numeric_limits<std::uint32_t>::max()) {
      sorted.emplace_back(temp);
    } else {
      sorted.emplace_back(mapping_vec.at(temp.v_));
    }
  }
  return sorted;
}

mutable_fws_multimap<location_idx_t, location_idx_t> apply_permutation_multimap(
    mutable_fws_multimap<location_idx_t, location_idx_t> const& input,
    vector<location_idx_t> const& permutation,
    vector<location_idx_t> const& mapping_vec) {
  auto sorted = mutable_fws_multimap<location_idx_t, location_idx_t>{};
  for (auto i = 0U; i < input.size(); ++i) {
    sorted.emplace_back();
  }
  for (auto i = 0U; i < input.size(); ++i) {
    auto temp = input[permutation[i]];
    if (temp.size() > 0) {
      for (auto j = 0U; j < temp.size(); ++j) {
        sorted.emplace_back_entry(i, mapping_vec.at(temp.at(j).v_));
      }
    }
  }
  return sorted;
}

mutable_fws_multimap<location_idx_t, footpath> apply_permutation_multimap_foot(
    mutable_fws_multimap<location_idx_t, footpath> const& input,
    vector<location_idx_t> const& permutation,
    vector<location_idx_t> const& mapping_vec) {
  auto sorted = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto i = 0U; i < input.size(); ++i) {
    sorted.emplace_back();
  }
  for (auto i = 0U; i < input.size(); ++i) {
    auto temp = input[permutation[i]];
    if (temp.size() > 0) {
      for (auto j = 0U; j < temp.size(); ++j) {
        auto old_footpath = temp.at(j);
        footpath new_footpath = {mapping_vec.at(old_footpath.target().v_),
                                 old_footpath.duration()};
        sorted.emplace_back_entry(i, new_footpath);
      }
    }
  }
  return sorted;
}

void permutate_locations(timetable& tt) {
  auto first_idx = 0U;
  auto unsorted = vector<std::pair<location_id, location_idx_t>>{};
  for (auto& [key, value] : tt.locations_.location_id_to_idx_) {
    if (is_special(value)) {
      first_idx++;
    }
    unsorted.emplace_back(key, value);
  }

  auto const location_permutation =
      build_permutation_vec(tt.location_routes_, first_idx);
  auto const location_mapping = create_mapping_vec(location_permutation);

  auto sorted_loc_id_to_idx = hash_map<location_id, location_idx_t>{};
  sorted_loc_id_to_idx.reserve(unsorted.size());
  for (auto i = 0U; i < location_permutation.size(); ++i) {
    auto temp = unsorted.at(location_permutation.at(i).v_);
    sorted_loc_id_to_idx.insert({temp.first, location_idx_t{i}});
  }
  tt.locations_.location_id_to_idx_ = std::move(sorted_loc_id_to_idx);

  tt.locations_.names_ =
      apply_permutation_vec(tt.locations_.names_, location_permutation);
  tt.locations_.ids_ =
      apply_permutation_vec(tt.locations_.ids_, location_permutation);
  tt.locations_.coordinates_ =
      apply_permutation_vec(tt.locations_.coordinates_, location_permutation);
  tt.locations_.src_ =
      apply_permutation_vec(tt.locations_.src_, location_permutation);
  tt.locations_.transfer_time_ =
      apply_permutation_vec(tt.locations_.transfer_time_, location_permutation);
  tt.locations_.types_ =
      apply_permutation_vec(tt.locations_.types_, location_permutation);
  tt.locations_.location_timezones_ = apply_permutation_vec(
      tt.locations_.location_timezones_, location_permutation);
  tt.locations_.parents_ = apply_permutation_and_mapping_vec(
      tt.locations_.parents_, location_permutation, location_mapping);
  tt.locations_.equivalences_ = apply_permutation_multimap(
      tt.locations_.equivalences_, location_permutation, location_mapping);
  tt.locations_.children_ = apply_permutation_multimap(
      tt.locations_.children_, location_permutation, location_mapping);
  tt.locations_.preprocessing_footpaths_out_ = apply_permutation_multimap_foot(
      tt.locations_.preprocessing_footpaths_out_, location_permutation,
      location_mapping);
  tt.locations_.preprocessing_footpaths_in_ =
      apply_permutation_multimap_foot(tt.locations_.preprocessing_footpaths_in_,
                                      location_permutation, location_mapping);
  tt.location_areas_ =
      apply_permutation_vec(tt.location_areas_, location_permutation);
  tt.route_location_seq_ = apply_permutation_to_route_loc_seq(
      tt.route_location_seq_, location_mapping);
  tt.location_routes_ =
      apply_permutation_vec(tt.location_routes_, location_permutation);
}

}  // namespace nigiri