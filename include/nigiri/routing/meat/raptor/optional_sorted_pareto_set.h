#pragma once

#include <cinttypes>
#include <algorithm>
#include <vector>

namespace nigiri::routing::meat::raptor {

template <typename T>
struct optional_sorted_pareto_set {
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  using const_reference = typename std::vector<T>::const_reference;
  using access_type = typename std::vector<T>::size_type;

  optional_sorted_pareto_set() = default;

  optional_sorted_pareto_set(T&& init_el) {
    els_.push_back(std::forward<T>(init_el));
    is_sorted_ = true;
  }

  size_t size() const { return els_.size(); }

  bool unsorted_add(T&& el) {
    auto n_removed = std::size_t{0};
    for (auto i = 0U; i < els_.size(); ++i) {
      if (els_[i].dominates(el)) {
        return false;
      }
      if (el.dominates(els_[i])) {
        n_removed++;
        continue;
      }
      els_[i - n_removed] = els_[i];
    }
    els_.resize(els_.size() - n_removed + 1);
    els_.back() = std::forward<T>(el);
    is_sorted_ = false;
    return true;
  }

  bool unsorted_add_on_sorted_set(T&& el) {
    if (!is_sorted()) {
      sort();
    }
    return unsorted_add(std::forward<T>(el));
  }

  bool sorted_add(T&& el) {
    auto const b = unsorted_add(std::forward<T>(el));
    if (!is_sorted()) {
      sort();
    }
    return b;
  }

  bool sorted_add_on_sorted_set(T&& el) {
    if (!is_sorted()) {
      sort();
    }
    return sorted_add(std::forward<T>(el));
  }


  friend const_iterator begin(optional_sorted_pareto_set const& s) {
    return s.begin();
  }
  friend const_iterator end(optional_sorted_pareto_set const& s) {
    return s.end();
  }
  friend iterator begin(optional_sorted_pareto_set& s) { return s.begin(); }
  friend iterator end(optional_sorted_pareto_set& s) { return s.end(); }
  iterator begin() { return els_.begin(); }
  iterator end() { return els_.end(); }
  const_iterator begin() const { return els_.begin(); }
  const_iterator end() const { return els_.end(); }
  iterator erase(iterator const& it) { return els_.erase(it); }
  iterator erase(iterator const& from, iterator const& to) {
    return els_.erase(from, to);
  }

  T const& operator[](access_type const index) const noexcept {
    return els_[index];
  }

  auto lower_bound(T const& t) {
    return std::lower_bound(els_.begin(), els_.end(), t);
  }

  void clear() {
    els_.clear();
    els_.shrink_to_fit();
    is_sorted_ = true;
  }
  void clear(T&& init_el) {
    els_.clear();
    els_.push_back(std::forward<T>(init_el));
    els_.shrink_to_fit();
    is_sorted_ = true;
  }

  bool is_sorted() const { return is_sorted_; }
  const_reference back() const { return els_.back(); }
  void sort() {
    std::sort(els_.begin(), els_.end());
    is_sorted_ = true;
  }

private:
  std::vector<T> els_;
  bool is_sorted_;
};

}  // namespace nigiri::routing::meat::raptor
