#pragma once

#include <type_traits>

namespace nigiri {

template <typename T>
struct interval {
  template <typename X>
  interval operator+(X const& x) const {
    return {from_ + x, to_ + x};
  }

  template <typename X>
  interval operator-(X const& x) const {
    return {from_ - x, to_ - x};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  operator interval<X>() {
    return {from_, to_};
  }

  bool contains(T const t) const { return t >= from_ && t < to_; }

  T from_{}, to_{};
};

template <typename T>
interval(T, T) -> interval<T>;

}  // namespace nigiri