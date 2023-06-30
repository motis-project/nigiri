#pragma once

#include <span>

#include "utl/verify.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct stop_seq_number_range {
  enum class number_seq_type {
    kZeroBased,
    kOneBased,
    kTenBased,
    kFullySpecified,
    kInvalid
  };

  struct iterator {
    struct compact {
      bool operator==(compact const& o) const {
        return std::tie(curr_, inc_, num_stops_) ==
               std::tie(o.curr_, o.inc_, o.num_stops_);
      }
      stop_idx_t curr_, inc_, num_stops_;
    };

    struct fully_specified {
      bool operator==(fully_specified const& o) const {
        return std::tuple(idx_, seq_.data(), seq_.size()) ==
               std::tuple(o.idx_, o.seq_.data(), seq_.size());
      }
      stop_idx_t idx_;
      std::span<stop_idx_t const> seq_;
    };

    using difference_type = stop_idx_t;
    using value_type = stop_idx_t;
    using pointer = stop_idx_t const*;
    using reference = stop_idx_t const&;
    using iterator_category = std::forward_iterator_tag;

    iterator(stop_idx_t const idx,
             std::span<stop_idx_t const> const seq_numbers)
        : seq_numbers_{fully_specified{idx, seq_numbers}} {}

    iterator(stop_idx_t const curr,
             stop_idx_t const inc,
             stop_idx_t const num_stops)
        : seq_numbers_{compact{curr, inc, num_stops}} {}

    stop_idx_t operator*() {
      return std::visit(
          utl::overloaded{[](compact& c) { return c.curr_; },
                          [](fully_specified& f) { return f.seq_[f.idx_]; }},
          seq_numbers_);
    }

    iterator& operator++() {
      std::visit(utl::overloaded{[](compact& c) {
                                   utl::verify(c.num_stops_ >= 1U, "");
                                   c.curr_ += c.inc_;
                                   c.num_stops_ -= 1U;
                                 },
                                 [](fully_specified& f) {
                                   utl::verify(f.idx_ < f.seq_.size(), "");
                                   ++f.idx_;
                                 }},
                 seq_numbers_);
      return *this;
    }

    iterator operator++(int) noexcept {
      auto r = *this;
      ++(*this);
      return r;
    }

    bool operator==(iterator const& o) const {
      if (seq_numbers_.index() != o.seq_numbers_.index()) {
        return false;
      }
      return std::visit(
          utl::overloaded{[&](compact const& c) {
                            return c == std::get<compact>(o.seq_numbers_);
                          },
                          [&](fully_specified const& f) {
                            return f ==
                                   std::get<fully_specified>(o.seq_numbers_);
                          }},
          seq_numbers_);
    }

    bool operator!=(iterator o) const { return !(*this == o); }

    std::variant<compact, fully_specified> seq_numbers_;
  };

  stop_seq_number_range(std::span<stop_idx_t const> seq_numbers,
                        stop_idx_t const location_seq_size)
      : seq_numbers_{seq_numbers}, location_seq_size_{location_seq_size} {}

  iterator begin() const {
    switch (type()) {
      case number_seq_type::kZeroBased:
        return iterator{0U, 1U, location_seq_size_};
      case number_seq_type::kOneBased:
        return iterator{1U, 1U, location_seq_size_};
      case number_seq_type::kTenBased:
        return iterator{10U, 10U, location_seq_size_};
      case number_seq_type::kFullySpecified: return iterator{0U, seq_numbers_};
      default:;
    }
    throw std::runtime_error{"invalid seq"};
  }

  iterator end() const {
    switch (type()) {
      case number_seq_type::kZeroBased:
        return iterator{location_seq_size_, 1U, 0U};
      case number_seq_type::kOneBased:
        return iterator{static_cast<stop_idx_t>(location_seq_size_ + 1U), 1U,
                        0U};
      case number_seq_type::kTenBased:
        return iterator{static_cast<stop_idx_t>(10U + 10 * location_seq_size_),
                        10U, 0U};
      case number_seq_type::kFullySpecified:
        return iterator{static_cast<stop_idx_t>(seq_numbers_.size()),
                        seq_numbers_};
      default:;
    }
    throw std::runtime_error{"invalid seq"};
  }

  friend iterator begin(stop_seq_number_range const& r) { return r.begin(); }
  friend iterator end(stop_seq_number_range const& r) { return r.end(); }

  number_seq_type type() const {
    if (seq_numbers_.empty()) {
      return number_seq_type::kZeroBased;
    } else if (seq_numbers_.size() == 1U) {
      switch (seq_numbers_[0]) {
        case 1: return number_seq_type::kOneBased;
        case 10: return number_seq_type::kTenBased;
        default: return number_seq_type::kInvalid;
      }
    } else {
      return number_seq_type::kFullySpecified;
    }
  }

  std::span<stop_idx_t const> seq_numbers_;
  stop_idx_t location_seq_size_;
};

void encode_seq_numbers(std::span<stop_idx_t>,
                        std::basic_string<stop_idx_t>& out);

}  // namespace nigiri::loader::gtfs