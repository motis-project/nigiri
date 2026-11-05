#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <bit>
#include <iterator>
#include <stdexcept>
#include <utility>

#include "cista/containers/vecvec.h"

#include "nigiri/common/dedup_vecvec.h"
#include "nigiri/types.h"

namespace nigiri {

template <typename K, typename V, typename Data = vecvec<std::uint32_t, V>>
struct sparse_vecvec {
  using key_type = K;
  using data_idx_t = std::uint32_t;
  using const_bucket = decltype(std::declval<Data const&>()[data_idx_t{}]);
  static constexpr auto const kBlockSize = 512U;

  struct const_iterator {
    using value_type = const_bucket;
    using difference_type = std::ptrdiff_t;
    using reference = const_bucket;
    using iterator_category = std::forward_iterator_tag;

    const_bucket operator*() const {
      return v_->data_[v_->has_bucket_.test(i_) ? next_ : 0U];
    }

    const_iterator& operator++() {
      if (v_->has_bucket_.test(i_)) {
        ++next_;
      }
      ++i_;
      return *this;
    }

    const_iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const_iterator const& o) const { return i_ == o.i_; }

    sparse_vecvec const* v_{nullptr};
    std::uint32_t i_{0U};
    data_idx_t next_{1U};
  };

  static sparse_vecvec with_default(std::initializer_list<V> s) {
    auto v = sparse_vecvec{};
    v.data_.emplace_back(s);
    return v;
  }

  template <typename Seq>
  void emplace_back(Seq const& s) {
    assert(data_.size() != 0U);
    auto const has_bucket = !std::ranges::equal(s, data_[0U]);
    add_key(has_bucket);
    if (has_bucket) {
      data_.emplace_back(s);
    }
  }

  void emplace_back(std::initializer_list<V> s) {
    emplace_back<std::initializer_list<V>>(s);
  }

  void add_key(bool const has_bucket) {
    auto const i = size();
    if (i % kBlockSize == 0U) {
      key_frames_.push_back(static_cast<data_idx_t>(data_.size()));
    }
    has_bucket_.resize(i + 1U);
    has_bucket_.set(i, has_bucket);
  }

  data_idx_t data_idx(std::uint32_t const i) const {
    if (!has_bucket_.test(i)) {
      return 0U;
    }
    auto const block = i / kBlockSize;
    auto idx = key_frames_[block];
    for (auto w = block * (kBlockSize / 64U); w != i / 64U; ++w) {
      idx += static_cast<data_idx_t>(std::popcount(has_bucket_.blocks_[w]));
    }
    idx += static_cast<data_idx_t>(
        std::popcount(has_bucket_.blocks_[i / 64U] &
                      ((std::uint64_t{1U} << (i % 64U)) - 1U)));
    return idx;
  }

  auto operator[](K const k) const { return data_[data_idx(to_idx(k))]; }

  auto at(K const k) const {
    if (to_idx(k) >= size()) {
      throw std::out_of_range{"sparse_vecvec::at"};
    }
    return (*this)[k];
  }

  const_iterator begin() const { return {this, 0U, 1U}; }
  const_iterator end() const { return {this, size(), 0U}; }
  friend const_iterator begin(sparse_vecvec const& v) { return v.begin(); }
  friend const_iterator end(sparse_vecvec const& v) { return v.end(); }

  std::uint32_t size() const { return has_bucket_.size(); }

  bitvec has_bucket_;
  vector<data_idx_t> key_frames_;
  Data data_;
};

template <typename K, typename V>
using sparse_dedup_vecvec = sparse_vecvec<K, V, dedup_vecvec<std::uint32_t, V>>;

template <typename Buckets>
auto make_sparse_vecvec(Buckets const& buckets) {
  auto v =
      sparse_vecvec<typename Buckets::key_type,
                    cista::detail::bucket_value_t<Buckets>>::with_default({});
  for (auto const& b : buckets) {
    v.emplace_back(b);
  }
  return v;
}

template <typename K, typename V>
sparse_dedup_vecvec<K, V> make_sparse_vecvec(dedup_vecvec<K, V>&& d) {
  auto v = sparse_dedup_vecvec<K, V>{};
  if (d.sequence_.empty()) {
    return v;
  }

  auto counts = std::vector<std::uint32_t>(d.sequences_.size());
  for (auto const s : d.sequence_) {
    ++counts[s];
  }
  auto const top = static_cast<std::uint32_t>(
      std::distance(begin(counts), std::ranges::max_element(counts)));

  v.data_.sequences_ = std::move(d.sequences_);
  v.data_.sequence_.push_back(top);
  for (auto const s : d.sequence_) {
    v.add_key(s != top);
    if (s != top) {
      v.data_.sequence_.push_back(s);
    }
  }
  return v;
}

}  // namespace nigiri
