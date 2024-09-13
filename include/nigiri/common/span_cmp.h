#pragma once

#include <algorithm>
#include <span>
#include <vector>

namespace std {

template <typename T>
bool operator==(span<T const> a, vector<T> const& b) {
  return equal(begin(a), end(a), begin(b), end(b));
}

template <typename T>
bool operator==(vector<T> const& a, span<T const> b) {
  return std::equal(begin(a), end(a), begin(b), end(b));
}

}  // namespace std