#pragma once

#include <type_traits>

namespace nigiri {

template <typename T>
struct interval {
  struct iterator {
    using value_type = T;
    friend auto operator<=>(iterator const&, iterator const&) = default;
    friend bool operator==(iterator const&, iterator const&) = default;
    friend bool operator!=(iterator const&, iterator const&) = default;
    friend bool operator<=(iterator const&, iterator const&) = default;
    friend bool operator<(iterator const&, iterator const&) = default;
    friend bool operator>=(iterator const&, iterator const&) = default;
    friend bool operator>(iterator const&, iterator const&) = default;
    value_type operator*() { return t_; }
    iterator operator++() { return iterator{++t_}; }
    T t_;
  };

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
  operator interval<X>() { return {from_, to_}; }

  bool contains(T const t) const { return t >= from_ && t < to_; }

  iterator begin() const { return {from_}; }
  iterator end() const { return {to_}; }
  friend iterator begin(interval const& r) { return r.begin(); }
  friend iterator end(interval const& r) { return r.end(); }

  T from_{}, to_{};
};

template <typename T>
interval(T, T) -> interval<T>;

}  // namespace nigiri
