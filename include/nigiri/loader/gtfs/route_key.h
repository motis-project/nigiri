#pragma once

#include "utl/helpers/algorithm.h"

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
  std::array<bitvec, kNumRouteFlags> const* flags_{nullptr};
};

struct route_key_hash {
  using is_transparent = void;

  static cista::hash_t hash(clasz const c,
                            stop_seq_t const& seq,
                            std::array<bitvec, kNumRouteFlags> const* flags) {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<stop_seq_t>{}(seq));
    h = cista::hash_combine(h, c);
    if (flags != nullptr) {
      for (auto const& f : *flags) {
        if (!f.empty()) {
          h = cista::hash_combine(h, cista::hashing<bitvec>{}(f));
        }
      }
    }
    return h;
  }

  cista::hash_t operator()(route_key_t const& x) const {
    return hash(x.clasz_, x.stop_seq_, &x.flags_);
  }

  cista::hash_t operator()(route_key_ptr_t const& x) const {
    return hash(x.clasz_, *x.stop_seq_, x.flags_);
  }
};

struct route_key_equals {
  using is_transparent = void;

  cista::hash_t operator()(route_key_t const& a, route_key_t const& b) const {
    return std::tie(a.clasz_, a.stop_seq_, a.flags_) ==
           std::tie(b.clasz_, b.stop_seq_, a.flags_);
  }

  cista::hash_t operator()(route_key_ptr_t const& a,
                           route_key_t const& b) const {
    if (a.flags_ == nullptr ||
        utl::all_of(*a.flags_, [&](auto f) { return f.empty(); }) !=
            utl::all_of(b.flags_, [&](auto f) { return f.empty(); })) {
      return false;
    }
    return a.clasz_ == b.clasz_ && *a.stop_seq_ == b.stop_seq_ &&
           (a.flags_ == nullptr || *a.flags_ == b.flags_);
  }
};

}  // namespace nigiri::loader::gtfs
