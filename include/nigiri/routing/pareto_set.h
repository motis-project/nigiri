#pragma once

#include <cinttypes>
#include <concepts>
#include <tuple>
#include <vector>

namespace nigiri {

template <typename T>
concept HasDominates = requires(T x) {
  { x.dominates(x) } -> std::convertible_to<bool>;
};

template <typename T>
struct pareto_set {
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  size_t size() const { return els_.size(); }

  std::tuple<bool, iterator, iterator> add(T&& el)
    requires HasDominates<T>
  {
    return add(std::move(el),
               [](auto&& a, auto&& b) { return a.dominates(b); });
  }

  template <typename Fn>
  std::tuple<bool, iterator, iterator> add(T&& el, Fn&& dominates)
    requires(!HasDominates<T>)
  {
    auto n_removed = std::size_t{0};
    for (auto i = 0U; i < els_.size(); ++i) {
      if (dominates(els_[i], el)) {
        return {false, end(), std::next(begin(), i)};
      }
      if (dominates(el, els_[i])) {
        n_removed++;
        continue;
      }
      els_[i - n_removed] = els_[i];
    }
    els_.resize(els_.size() - n_removed + 1);
    els_.back() = std::move(el);
    return {true, std::next(begin(), static_cast<unsigned>(els_.size() - 1)),
            end()};
  }

  friend const_iterator begin(pareto_set const& s) { return s.begin(); }
  friend const_iterator end(pareto_set const& s) { return s.end(); }
  friend iterator begin(pareto_set& s) { return s.begin(); }
  friend iterator end(pareto_set& s) { return s.end(); }
  iterator begin() { return els_.begin(); }
  iterator end() { return els_.end(); }
  const_iterator begin() const { return els_.begin(); }
  const_iterator end() const { return els_.end(); }
  iterator erase(iterator const& it) { return els_.erase(it); }
  iterator erase(iterator const& from, iterator const& to) {
    return els_.erase(from, to);
  }
  void clear() { els_.clear(); }

private:
  std::vector<T> els_;
};

}  // namespace nigiri
