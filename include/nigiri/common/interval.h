#pragma once

#include <cassert>
#include <concepts>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <ostream>
#include <type_traits>

#include "fmt/ostream.h"
#include "fmt/ranges.h"

#include "cista/strong.h"

namespace nigiri {

template <typename T>
struct interval {
  struct iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<value_type>;
    using reference = value_type;
    friend auto operator<=>(iterator const&, iterator const&) = default;
    friend bool operator==(iterator const&, iterator const&) = default;
    friend bool operator!=(iterator const&, iterator const&) = default;
    friend bool operator<=(iterator const&, iterator const&) = default;
    friend bool operator<(iterator const&, iterator const&) = default;
    friend bool operator>=(iterator const&, iterator const&) = default;
    friend bool operator>(iterator const&, iterator const&) = default;
    value_type operator*() const { return t_; }
    iterator& operator++() {
      ++t_;
      return *this;
    }
    iterator& operator--() {
      --t_;
      return *this;
    }
    iterator& operator+=(difference_type const x) {
      t_ += x;
      return *this;
    }
    iterator& operator-=(difference_type const x) {
      t_ -= x;
      return *this;
    }
    iterator operator+(difference_type const x) const { return {t_ + x}; }
    iterator operator-(difference_type const x) const { return {t_ - x}; }
    friend difference_type operator-(iterator const& a, iterator const& b) {
      return static_cast<difference_type>(cista::to_idx(a.t_) -
                                          cista::to_idx(b.t_));
    }
    T t_;
  };

  template <typename X>
  interval operator>>(X const& x) const {
    return {static_cast<T>(from_ + x), static_cast<T>(to_ + x)};
  }

  template <typename X>
  interval operator<<(X const& x) const {
    return {static_cast<T>(from_ - x), static_cast<T>(to_ - x)};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  operator interval<X>() {
    return {from_, to_};
  }

  T clamp(T const x) const { return std::clamp(x, from_, to_); }

  bool contains(T const t) const { return t >= from_ && t < to_; }

  bool contains(interval const& o) const {
    return from_ <= o.from_ && to_ >= o.to_;
  }

  bool overlaps(interval const& o) const {
    return from_ < o.to_ && to_ > o.from_;
  }

  interval intersect(interval const& o) const {
    if (overlaps(o)) {
      return {std::max(from_, o.from_), std::min(to_, o.to_)};
    } else {
      return {};
    }
  }

  iterator begin() const { return {from_}; }
  iterator end() const { return {to_}; }
  friend iterator begin(interval const& r) { return r.begin(); }
  friend iterator end(interval const& r) { return r.end(); }

  std::reverse_iterator<iterator> rbegin() const {
    return std::reverse_iterator<iterator>{iterator{to_}};
  }
  std::reverse_iterator<iterator> rend() const {
    return std::reverse_iterator<iterator>{iterator{from_}};
  }
  friend std::reverse_iterator<iterator> rbegin(interval const& r) {
    return r.begin();
  }
  friend std::reverse_iterator<iterator> rend(interval const& r) {
    return r.end();
  }

  auto size() const { return cista::to_idx(to_ - from_); }

  bool empty() const { return to_ - from_ == 0U; }

  T operator[](std::size_t const i) const {
    assert(contains(from_ + static_cast<T>(i)));
    return from_ + static_cast<T>(i);
  }

  friend std::ostream& operator<<(std::ostream& out, interval const& i) {
    return out << "[" << i.from_ << ", " << i.to_ << "[";
  }

  friend bool operator==(interval const& a, interval const& b) {
    return std::tie(a.from_, a.to_) == std::tie(b.from_, b.to_);
  }

  friend bool operator<(interval const& a, interval const& b) {
    return std::tie(a.from_, a.to_) < std::tie(b.from_, b.to_);
  }

  T from_{}, to_{};
};

template <typename T, typename T1, typename = std::common_type_t<T1, T>>
interval(T, T1) -> interval<std::common_type_t<T, T1>>;

}  // namespace nigiri

template <typename T>
struct fmt::formatter<nigiri::interval<T>> : ostream_formatter {};

namespace fmt {
template <typename T, typename Char>
struct range_format_kind<nigiri::interval<T>, Char, void> : std::false_type {};
}  // namespace fmt