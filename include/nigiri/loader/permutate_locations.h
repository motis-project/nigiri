#pragma once

#include "nigiri/types.h"
#include "nigiri/stop.h"
#include "nigiri/footpath.h"

#include <ranges>
#include <vector>
#include <algorithm>

namespace nigiri {

static vector<location_idx_t> permutation_;
static vector<location_idx_t> mapping_vec;

static vector<location_idx_t> create_mapping_vec(vector<location_idx_t> vec) 
{
  auto n = vec.size();
  std::vector<location_idx_t> result(n);
  for (auto i = 0U; i < n; ++i) {
      result[vec[i].v_] = location_idx_t{i};
  }
  vector<location_idx_t> result_vec;
  for (auto val : result) {
    result_vec.push_back(val);
  }
  return result_vec;
}

static void build_permutation_vec(vecvec<location_idx_t, route_idx_t> order, uint32_t first_idx) 
{
  permutation_.resize(order.size());
    for (auto i = 0U; i != permutation_.size(); ++i) {
      permutation_[i] = location_idx_t{i};
    }
    std::stable_sort(begin(permutation_) + first_idx, end(permutation_),
              [&](auto&& a, auto&& b) { 
                return order.at(location_idx_t{a}).size() > 0 && order.at(location_idx_t{b}).size() == 0; 
              });
  mapping_vec = create_mapping_vec(permutation_);
}

static vector<location_idx_t> get_permutation_vector()
{
  return permutation_;
}


template<typename T>
T apply_permutation_vec(T input) 
{
  T sorted;
  for(auto i = 0U; i < input.size(); ++i) 
  {
    auto temp = input.at(permutation_[i]);
    sorted.emplace_back(temp);
  }
  return sorted;
}

static vecvec<route_idx_t, stop::value_type> apply_permutation_to_route_loc_seq(vecvec<route_idx_t, stop::value_type> input) 
{
  vecvec<route_idx_t, stop::value_type> sorted;
  for(auto j = 0U; j < input.size(); ++j) 
  {
    vector<stop::value_type> new_stop_v_types;
    auto loc_seq = input.at(route_idx_t{j});
    for(auto s : loc_seq) { 
      auto const old_stop = stop{s};
      auto stop_loc_idx = old_stop.location_idx();
      auto new_idx = mapping_vec.at(stop_loc_idx.v_);
      stop new_stop = {new_idx, old_stop.in_allowed(), old_stop.out_allowed(), old_stop.in_allowed_wheelchair(), old_stop.out_allowed_wheelchair()};
      new_stop_v_types.emplace_back(new_stop.value());
    }
    sorted.emplace_back(new_stop_v_types);
  }
  return sorted;
}

static vector_map<location_idx_t, location_idx_t> apply_permutation_and_mapping_vec(vector_map<location_idx_t, location_idx_t> input) 
{
  vector_map<location_idx_t, location_idx_t> sorted;
  for(auto i = 0U; i < input.size(); ++i) 
  {
    auto temp = input.at(permutation_[i]);
    if(temp == std::numeric_limits<uint32_t>::max()) {
      sorted.emplace_back(temp);
    }
    else {
      sorted.emplace_back(mapping_vec.at(temp.v_));
    }
  }
  return sorted;
}


static mutable_fws_multimap<location_idx_t, location_idx_t> apply_permutation_multimap(mutable_fws_multimap<location_idx_t, location_idx_t> input)
{
    mutable_fws_multimap<location_idx_t, location_idx_t> sorted;
    for(auto i = 0U; i < input.size(); ++i) 
    {
      sorted.emplace_back();
    }
    for(auto i = 0U; i < input.size(); ++i) 
    {
      auto temp = input.at(permutation_[i]);
      if(temp.size() > 0) {
        for(auto j = 0U; j < temp.size(); ++j) {
          sorted.emplace_back_entry(i, mapping_vec.at(temp.at(j).v_));
        }
      }
    }
    return sorted;
}

static mutable_fws_multimap<location_idx_t, footpath> apply_permutation_multimap_foot(mutable_fws_multimap<location_idx_t, footpath> input)
{
    mutable_fws_multimap<location_idx_t, footpath> sorted;
    for(auto i = 0U; i < input.size(); ++i) 
    {
      sorted.emplace_back();
    }
    for(auto i = 0U; i < input.size(); ++i) 
    {
      auto temp = input.at(permutation_[i]);
      if(temp.size() > 0) {
        for(auto j = 0U; j < temp.size(); ++j) {
          auto old_footpath = temp.at(j);
          footpath new_footpath = {mapping_vec.at(old_footpath.target().v_), old_footpath.duration()};
          sorted.emplace_back_entry(i, new_footpath);
        }
      }
    }
    return sorted;
}

} // namespace nigiri 