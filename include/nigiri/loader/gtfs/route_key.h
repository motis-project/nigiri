#pragma once

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

struct route_key_t {
  clasz clasz_;
  stop_seq_t stop_seq_;
  std::array<bitvec, kNumRouteFlags> flags_;
};

struct route_key_ptr_t {
  clasz clasz_;
  stop_seq_t const* stop_seq_;
  std::array<bitvec const*, kNumRouteFlags> flags_;
};

struct route_key_hash {
  using is_transparent = void;

  cista::hash_t operator()(route_key_t const& x) const {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<stop_seq_t>{}(x.stop_seq_));
    h = cista::hash_combine(h, x.clasz_);
    for (auto const& f : x.flags_) {
      if (!f.empty()) {
        h = cista::hash_combine(h, cista::hashing<bitvec>{}(f));
      }
    }
    return h;
  }

  cista::hash_t operator()(route_key_ptr_t const& x) const {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<stop_seq_t>{}(*x.stop_seq_));
    h = cista::hash_combine(h, x.clasz_);
    for (auto const* f : x.flags_) {
      if (f != nullptr && !f->empty()) {
        h = cista::hash_combine(h, cista::hashing<bitvec>{}(*f));
      }
    }
    return h;
  }
};

struct route_key_equals {
  using is_transparent = void;

  bool operator()(route_key_t const& a, route_key_t const& b) const {
    return std::tie(a.clasz_, a.stop_seq_, a.flags_) ==
           std::tie(b.clasz_, b.stop_seq_, b.flags_);
  }

  bool operator()(route_key_ptr_t const& a, route_key_t const& b) const {
    if (a.clasz_ != b.clasz_ || *a.stop_seq_ != b.stop_seq_) {
      return false;
    }
    for (auto f = 0U; f != kNumRouteFlags; ++f) {
      auto const* pf = a.flags_[f];
      auto const empty_a = pf == nullptr || pf->empty();
      auto const empty_b = b.flags_[f].empty();
      if (empty_a != empty_b) {
        return false;
      }
      if (!empty_a && *pf != b.flags_[f]) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace nigiri::loader::gtfs
