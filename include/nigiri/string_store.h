#pragma once

#include "nigiri/types.h"

namespace nigiri {

using string_idx_t = cista::strong<std::uint32_t, struct _string_idx>;

struct string_idx_hash {
  using is_transparent = void;
  explicit string_idx_hash(vecvec<string_idx_t, char> const& s) : s_{s} {}
  cista::hash_t operator()(string_idx_t const i) const {
    return cista::hash(s_[i].view());
  }
  cista::hash_t operator()(std::string_view s) const { return cista::hash(s); }
  vecvec<string_idx_t, char> const& s_;
};

struct string_idx_equals {
  using is_transparent = void;
  explicit string_idx_equals(vecvec<string_idx_t, char> const& s) : s_{s} {}
  cista::hash_t operator()(std::string_view a, string_idx_t const b) const {
    return a == s_[b].view();
  }
  cista::hash_t operator()(string_idx_t const a, string_idx_t const b) const {
    return s_[a].view() == s_[b].view();
  }
  vecvec<string_idx_t, char> const& s_;
};

using string_cache_t =
    hash_set<string_idx_t, string_idx_hash, string_idx_equals>;

struct string_store {
  std::string_view get(string_idx_t const x) const {
    return strings_[x].view();
  }

  std::optional<std::string_view> try_get(string_idx_t const s) const {
    return s == string_idx_t::invalid() ? std::nullopt : std::optional{get(s)};
  }

  string_idx_t register_string(string_cache_t& cache, std::string_view s) {
    if (auto const it = cache.find(s); it != end(cache)) {
      return *it;
    } else {
      auto next = string_idx_t{strings_.size()};
      strings_.emplace_back(s);
      cache.emplace(next);
      return next;
    }
  }

  vecvec<string_idx_t, char> strings_;
};

}  // namespace nigiri