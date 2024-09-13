#pragma once

#include <cassert>

namespace nigiri {

template <typename Span>
struct base_flat_matrix_view {
  using value_type = typename Span::value_type;
  using size_type = typename Span::size_type;

  struct row {
    row(base_flat_matrix_view& matrix, size_type const i)
        : matrix_(matrix), i_(i) {}

    using iterator = typename Span::iterator;
    // libc++ doesn't have std::span::const_iterator
    using const_iterator = iterator;

    const_iterator begin() const {
      return std::next(matrix_.entries_, matrix_.n_columns_ * i_);
    }
    const_iterator end() const {
      return std::next(matrix_.entries_, matrix_.n_columns_ * (i_ + 1));
    }
    iterator begin() {
      return std::next(matrix_.entries_, matrix_.n_columns_ * i_);
    }
    iterator end() {
      return std::next(matrix_.entries_, matrix_.n_columns_ * (i_ + 1));
    }
    friend const_iterator begin(row const& r) { return r.begin(); }
    friend const_iterator end(row const& r) { return r.end(); }
    friend iterator begin(row& r) { return r.begin(); }
    friend iterator end(row& r) { return r.end(); }

    value_type& operator[](size_type const j) {
      assert(j < matrix_.n_columns_);
      auto const pos = matrix_.n_columns_ * i_ + j;
      return matrix_.entries_[pos];
    }

    base_flat_matrix_view& matrix_;
    size_type i_;
  };

  struct const_row {
    const_row(base_flat_matrix_view const& matrix, size_type const i)
        : matrix_(matrix), i_(i) {}

    using iterator = typename Span::iterator;

    iterator begin() const {
      return std::next(matrix_.entries_, matrix_.n_columns_ * i_);
    }
    iterator end() const {
      return std::next(matrix_.entries_, matrix_.n_columns_ * (i_ + 1));
    }
    friend iterator begin(const_row const& r) { return r.begin(); }
    friend iterator end(const_row const& r) { return r.end(); }

    value_type const& operator[](size_type const j) const {
      assert(j < matrix_.n_columns_);
      auto const pos = matrix_.n_columns_ * i_ + j;
      return matrix_.entries_[pos];
    }

    base_flat_matrix_view const& matrix_;
    size_type i_;
  };

  base_flat_matrix_view() = default;

  base_flat_matrix_view(Span entries, size_type n_rows, size_type n_columns)
      : n_rows_{n_rows}, n_columns_{n_columns}, entries_{entries} {
    assert(entries_.size() == n_rows_ * n_columns_);
  }

  row operator[](size_type i) {
    assert(i < n_rows_);
    return {*this, i};
  }
  const_row operator[](size_type i) const {
    assert(i < n_rows_);
    return {*this, i};
  }

  value_type& operator()(size_type const i, size_type const j) {
    assert(i < n_rows_ && j < n_columns_);
    return entries_[n_columns_ * i + j];
  }

  row at(size_type const i) {
    verify(i < n_rows_, "matrix::at: index out of range");
    return {*this, i};
  }

  const_row at(size_type const i) const {
    verify(i < n_rows_, "matrix::at: index out of range");
    return {*this, i};
  }

  void reset(value_type const& t) {
    std::fill(begin(entries_), end(entries_), t);
  }

  size_type n_rows_{0U}, n_columns_{0U};
  Span entries_;
};

template <typename T>
using flat_matrix_view = base_flat_matrix_view<std::span<T>>;

}  // namespace nigiri
