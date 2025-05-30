#pragma once

#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

#include <algorithm>
#include <ranges>
#include <vector>

namespace nigiri {

extern vector<location_idx_t> permutation_;

vector<location_idx_t> build_permutation_vec(
    vecvec<location_idx_t, route_idx_t> const& order, uint32_t const first_idx);

template <typename T>
inline T apply_permutation_vec(T const& input) {
  T sorted;
  for (auto i = 0U; i < input.size(); ++i) {
    auto temp = input.at(permutation_[i]);
    sorted.emplace_back(temp);
  }
  return sorted;
}

vecvec<route_idx_t, stop::value_type> apply_permutation_to_route_loc_seq(
    vecvec<route_idx_t, stop::value_type> const& input);

vector_map<location_idx_t, location_idx_t> apply_permutation_and_mapping_vec(
    vector_map<location_idx_t, location_idx_t> const& input);

mutable_fws_multimap<location_idx_t, location_idx_t> apply_permutation_multimap(
    mutable_fws_multimap<location_idx_t, location_idx_t> const& input);

mutable_fws_multimap<location_idx_t, footpath> apply_permutation_multimap_foot(
    mutable_fws_multimap<location_idx_t, footpath> const& input);

}  // namespace nigiri