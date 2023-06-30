#pragma once

#include <optional>

namespace nigiri {

template <typename Map>
struct cached_lookup {
  using iterator_t = typename Map::iterator;
  using key_t = typename Map::key_type;
  using value_t = typename Map::mapped_type;
  using opt_iterator_t = std::optional<iterator_t>;

  explicit cached_lookup(Map& map) : map_{map} {}

  template <typename Key, typename CreateFn>
  value_t& operator()(Key&& key, CreateFn&& create_fn) {
    if (!prev_it_.has_value() ||
        (prev_it_.has_value() && (*prev_it_)->first != key)) {
      if (auto const it = map_.find(key); it == end(map_)) {
        prev_it_ = map_.emplace(key_t{key}, create_fn()).first;
      } else {
        prev_it_ = it;
      }
    }
    return (*prev_it_)->second;
  }

  template <typename Key>
  value_t& operator()(Key&& key) {
    return this->operator()(std::forward<Key>(key), []() { return value_t{}; });
  }

  Map& map_;
  opt_iterator_t prev_it_;
};

template <typename Map>
cached_lookup(Map& map) -> cached_lookup<Map>;

}  // namespace nigiri