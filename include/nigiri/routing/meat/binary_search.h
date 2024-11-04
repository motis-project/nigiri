#pragma once

namespace nigiri::routing::meat {

template <class Iter, class Pred>
Iter binary_find_first_true(Iter begin, Iter end, Pred p) {
  if (begin == end) return end;
  if (!p(*(end - 1))) return end;

  while (end - begin > 1) {
    Iter mid = begin + (end - begin - 1) / 2;

    if (p(*mid))
      end = mid + 1;
    else
      begin = mid + 1;
  }
  return begin;
}

}  // namespace nigiri::routing::meat