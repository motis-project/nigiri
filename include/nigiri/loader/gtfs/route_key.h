#pragma once

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

using route_key_t = std::pair<stop_seq_t, clasz>;

struct route_key_hash {
  using is_transparent = void;

  static cista::hash_t hash(stop_seq_t const& seq, clasz const c) {
    auto const h = cista::BASE_HASH;
    cista::hash_combine(h, cista::hashing<stop_seq_t>{}(seq));
    cista::hash_combine(h, c);
    return h;
  }

  cista::hash_t operator()(route_key_t const& x) const {
    return hash(x.first, x.second);
  }

  cista::hash_t operator()(std::pair<stop_seq_t const*, clasz> const& x) const {
    return hash(*x.first, x.second);
  }

  cista::hash_t operator()(std::pair<stop_seq_t*, clasz> const& x) const {
    return hash(*x.first, x.second);
  }
};

struct route_key_equals {
  using is_transparent = void;

  cista::hash_t operator()(route_key_t const& a, route_key_t const& b) const {
    return std::tie(a.first, a.second) == std::tie(b.first, b.second);
  }

  cista::hash_t operator()(std::pair<stop_seq_t const*, clasz> const& a,
                           std::pair<stop_seq_t, clasz> const& b) const {
    return std::tie(*a.first, a.second) == std::tie(b.first, b.second);
  }

  cista::hash_t operator()(std::pair<stop_seq_t, clasz> const& a,
                           std::pair<stop_seq_t const*, clasz> const& b) const {
    return std::tie(a.first, a.second) == std::tie(*b.first, b.second);
  }
};

}  // namespace nigiri::loader::gtfs