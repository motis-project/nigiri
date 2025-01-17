#pragma once

#include <iterator>

#include "cista/cuda_check.h"

#include "nigiri/common/interval.h"

namespace nigiri {

template <typename BeginIt, typename EndIt = BeginIt>
struct it_range {
  using value_type =
      std::remove_reference_t<decltype(*std::declval<BeginIt>())>;
  using reference_type = std::add_lvalue_reference_t<value_type>;
  using const_iterator = BeginIt;
  using iterator = BeginIt;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;

  template <typename Collection>
  explicit CISTA_CUDA_COMPAT it_range(Collection const& c)
      : begin_{std::cbegin(c)}, end_{std::cend(c)} {}
  explicit CISTA_CUDA_COMPAT it_range(BeginIt begin, EndIt end)
      : begin_{std::move(begin)}, end_{std::move(end)} {}
  CISTA_CUDA_COMPAT BeginIt begin() const { return begin_; }
  CISTA_CUDA_COMPAT EndIt end() const { return end_; }
  CISTA_CUDA_COMPAT reference_type operator[](std::size_t const i) const {
    return *std::next(begin_, static_cast<difference_type>(i));
  }
  CISTA_CUDA_COMPAT value_type const* data() const { return begin_; }
  CISTA_CUDA_COMPAT friend BeginIt begin(it_range const& r) {
    return r.begin();
  }
  CISTA_CUDA_COMPAT friend EndIt end(it_range const& r) { return r.end(); }
  CISTA_CUDA_COMPAT reference_type front() const { return *begin_; }
  CISTA_CUDA_COMPAT reference_type back() const {
    return *std::next(begin_, static_cast<difference_type>(size() - 1U));
  }
  CISTA_CUDA_COMPAT std::size_t size() const {
    return static_cast<std::size_t>(std::distance(begin_, end_));
  }
  CISTA_CUDA_COMPAT bool empty() const { return begin_ == end_; }
  BeginIt begin_;
  EndIt end_;
};

template <typename Collection>
it_range(Collection const&) -> it_range<typename Collection::const_iterator>;

template <typename BeginIt, typename EndIt>
it_range(BeginIt, EndIt) -> it_range<BeginIt, EndIt>;

template <typename Collection, typename T>
it_range<typename Collection::const_iterator> make_it_range(
    Collection const& c, interval<T> const& i) {
  return it_range{std::next(std::begin(c), static_cast<long>(i.from_)),
                  std::next(std::begin(c), static_cast<long>(i.to_))};
}

}  // namespace nigiri
