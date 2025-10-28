#pragma once

#include "nigiri/types.h"

namespace nigiri {

using none_tag_idx_t = cista::strong<std::uint8_t, struct _none_tag_idx_t>;

template <typename Idx, typename TagIdx = none_tag_idx_t>
struct string_store {
  using idx_t = Idx;
  using tag_idx_t = TagIdx;

  struct hash {
    using is_transparent = void;
    cista::hash_t operator()(idx_t const i) const {
      auto const h = cista::hash((*s_)[i].view());
      return cista::hash_combine(
          h, std::is_same<tag_idx_t, none_tag_idx_t>::value ? tag_idx_t{}.v_
                                                            : (*t_)[i].v_);
    }
    cista::hash_t operator()(std::pair<std::string_view, tag_idx_t> k) const {
      auto const h = cista::hash(k.first);
      return cista::hash_combine(h, k.second.v_);
    }
    ptr<vecvec<idx_t, char> const> s_;
    ptr<vector_map<idx_t, tag_idx_t> const> t_;
  };

  struct equals {
    using is_transparent = void;
    cista::hash_t operator()(std::pair<std::string_view, tag_idx_t> a,
                             idx_t const b) const {
      return a.first == (*s_)[b].view() &&
             (std::is_same<tag_idx_t, none_tag_idx_t>::value ||
              a.second == (*t_)[b]);
    }
    cista::hash_t operator()(idx_t const a, idx_t const b) const {
      return (*s_)[a].view() == (*s_)[b].view() &&
             (std::is_same<tag_idx_t, none_tag_idx_t>::value ||
              (*t_)[a] == (*t_)[b]);
    }
    ptr<vecvec<idx_t, char> const> s_;
    ptr<vector_map<idx_t, tag_idx_t> const> t_;
  };

  string_store() = default;
  string_store(string_store const& o) {
    if (&o != this) {
      strings_ = o.strings_;
      tags_ = o.tags_;
      cache_ = o.cache_;
      resolve();
    }
  }
  string_store(string_store&& o) {
    if (&o != this) {
      strings_ = std::move(o.strings_);
      tags_ = std::move(o.tags_);
      cache_ = std::move(o.cache_);
      resolve();
    }
  }
  string_store& operator=(string_store const& o) {
    if (&o != this) {
      strings_ = o.strings_;
      tags_ = o.tags_;
      cache_ = o.cache_;
      resolve();
    }
  }
  string_store& operator=(string_store&& o) {
    if (&o != this) {
      strings_ = std::move(o.strings_);
      tags_ = std::move(o.tags_);
      cache_ = std::move(o.cache_);
      resolve();
    }
  }

  template <typename Ctx, typename Fn>
  friend void recurse(Ctx&, string_store* el, Fn&& fn) {
    fn(&el->cache_);
    fn(&el->strings_);
    el->resolve();
  }

  auto cista_members() { return std::tie(cache_, strings_, tags_); }

  std::string_view get(idx_t const x) const {
    return x == idx_t::invalid() ? "" : strings_[x].view();
  }

  std::optional<std::string_view> try_get(idx_t const s) const {
    return s == idx_t::invalid() ? std::nullopt : std::optional{get(s)};
  }

  idx_t store(std::string_view str, tag_idx_t const tag = tag_idx_t{}) {
    if (auto const it = cache_.find(std::pair{str, tag}); it != end(cache_)) {
      return *it;
    } else {
      auto next = idx_t{strings_.size()};
      strings_.emplace_back(str);
      if (!std::is_same<tag_idx_t, none_tag_idx_t>::value) {
        tags_.emplace_back(tag);
      }
      cache_.emplace(next);
      return next;
    }
  }

  std::optional<idx_t> find(std::string_view str,
                            tag_idx_t const tag = tag_idx_t{}) const {
    auto const it = cache_.find(std::pair{str, tag});
    return it == end(cache_) ? std::nullopt : std::optional{*it};
  }

  void resolve() {
    cache_.hash_function().s_ = &strings_;
    cache_.key_eq().s_ = &strings_;
    cache_.hash_function().t_ = &tags_;
    cache_.key_eq().t_ = &tags_;
  }

  vecvec<idx_t, char> strings_;
  vector_map<idx_t, tag_idx_t> tags_;
  hash_set<idx_t, hash, equals> cache_{
      0U, {&strings_, &tags_}, {&strings_, &tags_}};
};

}  // namespace nigiri