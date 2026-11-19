#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "cista/cuda_check.h"

#include "nigiri/types.h"

namespace nigiri {

// The event times of all transports of one route at one event: first
// departure of the transport + the event's offset in the transport's pattern.
struct event_times {
  struct iterator;

  CISTA_CUDA_COMPAT delta operator[](std::size_t const i) const {
    return first_dep_[i] + offsets_[n_patterns_ == 1U ? 0U : pattern_[i]];
  }

  CISTA_CUDA_COMPAT transport_idx_t transport(std::uint32_t const i) const {
    return first_transport_ + i;
  }

  CISTA_CUDA_COMPAT std::size_t size() const { return n_transports_; }
  bool empty() const { return n_transports_ == 0U; }
  delta front() const { return (*this)[0U]; }
  delta back() const { return (*this)[n_transports_ - 1U]; }

  iterator begin() const;
  iterator end() const;

  delta const* first_dep_;  // of the route's transports
  std::uint16_t const* pattern_;  // of the route's transports
  delta const* offsets_;  // the event's offsets, one per pattern
  transport_idx_t first_transport_;
  std::uint32_t n_transports_;
  std::uint16_t n_patterns_;
};

struct event_times::iterator {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = delta;
  using difference_type = std::ptrdiff_t;
  using reference = delta;
  using pointer = void;

  delta operator*() const { return times_[i_]; }

  iterator& operator++() {
    ++i_;
    return *this;
  }
  iterator& operator--() {
    --i_;
    return *this;
  }
  iterator& operator+=(difference_type const n) {
    i_ = static_cast<std::uint32_t>(static_cast<difference_type>(i_) + n);
    return *this;
  }
  friend difference_type operator-(iterator const a, iterator const b) {
    return static_cast<difference_type>(a.i_) -
           static_cast<difference_type>(b.i_);
  }
  friend bool operator==(iterator const a, iterator const b) {
    return a.i_ == b.i_;
  }

  event_times times_;
  std::uint32_t i_;
};

inline event_times::iterator event_times::begin() const { return {*this, 0U}; }

inline event_times::iterator event_times::end() const {
  return {*this, n_transports_};
}

}  // namespace nigiri
