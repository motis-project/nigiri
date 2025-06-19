#pragma once

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

struct route_key_hash {
  using is_transparent = void;

  static bool seated_hash(std::vector<trip const*> const& trips,
                          hash_set<trip const*>& hashed) {
    auto h = cista::BASE_HASH;
    for (auto const* trp : trips) {
      h = cista::hash_combine(h, hash(trp, hashed));
    }
    return h;
  }

  static bool hash(trip const* trp, hash_set<trip const*>& hashed) {
    auto h = std::apply(
        [](auto&&... k) {
          return cista::hash_combine(cista::hashing<decltype(k)>{}(k)...);
        },
        trp->route_key());

    if (!trp->seated_in_.empty()) {
      [[unlikely]] h =
          cista::hash_combine(h, seated_hash(trp->seated_in_, hashed));
    }

    if (!trp->seated_out_.empty()) {
      [[unlikely]] h =
          cista::hash_combine(h, seated_hash(trp->seated_out_, hashed));
    }

    return h;
  }

  cista::hash_t operator()(trip const* trp) const {
    auto hashed = hash_set<trip const*>{};
    return hash(trp, hashed);
  }

  trip_data const& trip_data_;
};

struct route_key_equals {
  using is_transparent = void;

  static bool seated_eq(
      std::vector<trip const*> const& a,
      std::vector<trip const*> const& b,
      hash_set<std::pair<trip const*, trip const*>>& compared) {
    if (a.size() != b.size()) {
      return false;
    }

    return utl::all_of(utl::zip_no_size_check(a, b),
                       [&](std::pair<trip const*, trip const*> const& x) {
                         return eq(x.first, x.second, compared);
                       });
  }

  static bool eq(trip const* a,
                 trip const* b,
                 hash_set<std::pair<trip const*, trip const*>>& compared) {
    if (a->route_key() != b->route_key()) {
      return false;
    }

    if (a->seated_out_.empty() && b->seated_out_.empty() &&
        a->seated_in_.empty() && b->seated_in_.empty()) {
      [[likely]] return true;
    }

    if (!compared.emplace(a, b).second) {
      return true;
    }

    return seated_eq(a->seated_in_, b->seated_in_, compared) &&
           seated_eq(a->seated_out_, b->seated_out_, compared);
  }

  bool operator()(trip const* a, trip const* b) const {
    auto compared = hash_set<std::pair<trip const*, trip const*>>{};
    return eq(a, b, compared);
  }
};

}  // namespace nigiri::loader::gtfs
