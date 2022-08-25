#pragma once

#include <iterator>

namespace nigiri {

template <typename BeginIt, typename EndIt = BeginIt>
struct it_range {
  template <typename Collection>
  explicit it_range(Collection&& c)
      : begin_{std::begin(c)}, end_{std::end(c)} {}
  explicit it_range(BeginIt begin, EndIt end)
      : begin_{std::move(begin)}, end_{std::move(end)} {}
  BeginIt begin() const { return begin_; }
  EndIt end() const { return end_; }
  friend BeginIt begin(it_range const& r) { return r.begin(); }
  friend EndIt end(it_range const& r) { return r.end(); }
  BeginIt begin_;
  EndIt end_;
};

template <typename Collection>
it_range(Collection const&) -> it_range<typename Collection::iterator>;

template <typename BeginIt, typename EndIt>
it_range(BeginIt, EndIt) -> it_range<BeginIt, EndIt>;

}  // namespace nigiri