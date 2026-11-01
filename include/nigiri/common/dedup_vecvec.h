#pragma once

#include <cstdint>
#include <algorithm>

#include "nigiri/types.h"

namespace nigiri {

template <typename K, typename V>
struct dedup_vecvec {
  using sequence_idx_t = std::uint32_t;
  using sequences_t = vecvec<sequence_idx_t, V>;

  struct hash {
    using is_transparent = void;
    template <typename Seq>
    cista::hash_t operator()(Seq const& s) const {
      return cista::hashing<Seq>{}(s);
    }
    cista::hash_t operator()(sequence_idx_t const i) const {
      return (*this)((*s_)[i]);
    }
    sequences_t const* s_{nullptr};
  };

  struct equals {
    using is_transparent = void;
    template <typename Seq>
    bool operator()(Seq const& a, sequence_idx_t const b) const {
      return std::ranges::equal(a, (*s_)[b]);
    }
    bool operator()(sequence_idx_t const a, sequence_idx_t const b) const {
      return a == b;
    }
    sequences_t const* s_{nullptr};
  };

  using cache_t = hash_set<sequence_idx_t, hash, equals>;

  cache_t make_cache() const {
    return cache_t{0U, hash{&sequences_}, equals{&sequences_}};
  }

  template <typename Seq>
  void emplace_back(cache_t& cache, Seq const& s) {
    if (auto const it = cache.find(s); it != end(cache)) {
      sequence_.push_back(*it);
      return;
    }
    auto const idx = static_cast<sequence_idx_t>(sequences_.size());
    sequences_.emplace_back(s);
    cache.emplace(idx);
    sequence_.push_back(idx);
  }

  auto operator[](K const k) const { return sequences_[sequence_[k]]; }
  auto at(K const k) const { return sequences_.at(sequence_.at(k)); }
  std::size_t size() const { return sequence_.size(); }

  vector_map<K, sequence_idx_t> sequence_;
  sequences_t sequences_;
};

}  // namespace nigiri
