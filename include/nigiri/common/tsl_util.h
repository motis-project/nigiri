#pragma once

#include <string>
#include <string_view>

namespace nigiri::loader::gtfs {

struct equal_str {
  using is_transparent = void;

  bool operator()(std::string const& a, std::string_view b) const {
    return a == b;
  }

  bool operator()(std::string_view a, std::string const& b) const {
    return a == b;
  }

  bool operator()(std::string_view a, std::string_view const& b) const {
    return a == b;
  }

  bool operator()(std::string const& a, std::string const& b) const {
    return a == b;
  }
};

struct hash_str {
  std::size_t operator()(std::string const& x) const {
    return std::hash<std::string_view>()(std::string_view{x});
  }

  std::size_t operator()(std::string_view x) const {
    return std::hash<std::string_view>()(x);
  }
};

template <typename K,
          typename Key,
          typename T,
          typename Hash,
          typename KeyEqual,
          typename Allocator,
          unsigned NeighborhoodSize,
          bool StoreHash,
          typename GrowthPolicy,
          typename CreateFun>
auto& get_or_create(tsl::hopscotch_map<Key,
                                       T,
                                       Hash,
                                       KeyEqual,
                                       Allocator,
                                       NeighborhoodSize,
                                       StoreHash,
                                       GrowthPolicy>& m,
                    K&& key,
                    CreateFun&& f) {
  using Map = std::decay_t<decltype(m)>;
  if (auto const it = m.find(key); it == end(m)) {
    return m.emplace_hint(it, typename Map::key_type{std::forward<K>(key)}, f())
        .value();
  } else {
    return it.value();
  }
}

}  // namespace nigiri::loader::gtfs