#pragma once

namespace nigiri {

template <typename It, typename End, typename Key, typename Cmp>
It linear_lb(It&& it, End&& end, Key&& key, Cmp&& cmp) {
  for (; it != end; ++it) {
    if (!cmp(key, *it)) {
      return it;
    }
  }
  return end;
}

}  // namespace nigiri
