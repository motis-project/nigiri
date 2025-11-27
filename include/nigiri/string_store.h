#pragma once

#include "nigiri/types.h"

namespace nigiri {

template <typename Idx>
struct string_store {
  using idx_t = Idx;

  struct hash {
    using is_transparent = void;
    cista::hash_t operator()(idx_t const i) const {
      return cista::hash((*s_)[i].view());
    }
    cista::hash_t operator()(std::string_view s) const {
      return cista::hash(s);
    }
    ptr<vecvec<idx_t, char> const> s_;
  };

  struct equals {
    using is_transparent = void;
    cista::hash_t operator()(std::string_view a, idx_t const b) const {
      return a == (*s_)[b].view();
    }
    cista::hash_t operator()(idx_t const a, idx_t const b) const {
      return (*s_)[a].view() == (*s_)[b].view();
    }
    ptr<vecvec<idx_t, char> const> s_;
  };

  string_store() = default;
  string_store(string_store const& o) {
    if (&o != this) {
      strings_ = o.strings_;
      cache_ = o.cache_;
      resolve();
    }
  }
  string_store(string_store&& o) {
    if (&o != this) {
      strings_ = std::move(o.strings_);
      cache_ = std::move(o.cache_);
      resolve();
    }
  }
  string_store& operator=(string_store const& o) {
    if (&o != this) {
      strings_ = o.strings_;
      cache_ = o.cache_;
      resolve();
    }
    return *this;
  }
  string_store& operator=(string_store&& o) {
    if (&o != this) {
      strings_ = std::move(o.strings_);
      cache_ = std::move(o.cache_);
      resolve();
    }
    return *this;
  }

  template <typename Ctx, typename Fn>
  friend void recurse(Ctx&, string_store* el, Fn&& fn) {
    fn(&el->cache_);
    fn(&el->strings_);
    el->resolve();
  }

  auto cista_members() { return std::tie(cache_, strings_); }

  std::string_view get(idx_t const x) const {
    return x == idx_t::invalid() ? "" : strings_[x].view();
  }

  std::optional<std::string_view> try_get(idx_t const s) const {
    return s == idx_t::invalid() ? std::nullopt : std::optional{get(s)};
  }

  idx_t store(std::string_view s) {
    if (auto const it = cache_.find(s); it != end(cache_)) {
      return *it;
    } else {
      auto next = idx_t{strings_.size()};
      strings_.emplace_back(s);
      cache_.emplace(next);
      return next;
    }
  }

  std::optional<idx_t> find(std::string_view s) const {
    auto const it = cache_.find(s);
    return it == end(cache_) ? std::nullopt : std::optional{*it};
  }

  void resolve() {
    cache_.hash_function().s_ = &strings_;
    cache_.key_eq().s_ = &strings_;
  }

  vecvec<idx_t, char> strings_;
  hash_set<idx_t, hash, equals> cache_{0U, {&strings_}, {&strings_}};
};

}  // namespace nigiri