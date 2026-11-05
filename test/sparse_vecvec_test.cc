#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

#include "nigiri/common/sparse_vecvec.h"

using namespace nigiri;

TEST(sparse_vecvec, matches_reference) {
  auto ref = std::vector<std::vector<int>>{};
  for (auto i = 0U; i != 2000U; ++i) {
    if (i % 97U == 0U || i == 511U || i == 512U || i == 1999U) {
      ref.push_back({static_cast<int>(i), -static_cast<int>(i)});
    } else {
      ref.push_back({});
    }
  }

  auto v = sparse_vecvec<transport_idx_t, int>::with_default({});
  for (auto const& x : ref) {
    v.emplace_back(x);
  }

  ASSERT_EQ(ref.size(), v.size());
  EXPECT_EQ(25U, v.data_.size());  // default + 24 non-empty buckets
  for (auto i = 0U; i != ref.size(); ++i) {
    auto const b = v[transport_idx_t{i}];
    EXPECT_TRUE(std::ranges::equal(ref[i], b)) << "i=" << i;
  }
  auto i = 0U;
  for (auto const b : v) {
    EXPECT_TRUE(std::ranges::equal(ref[i], b)) << "i=" << i;
    ++i;
  }
  EXPECT_EQ(ref.size(), i);
  EXPECT_TRUE(std::ranges::equal(ref[1999U], v.at(transport_idx_t{1999U})));
  EXPECT_THROW(v.at(transport_idx_t{2000U}), std::out_of_range);
}

TEST(sparse_vecvec, non_empty_default) {
  auto v = sparse_vecvec<transport_idx_t, int>::with_default({7});
  v.emplace_back(std::vector<int>{7});
  v.emplace_back(std::vector<int>{1, 2});
  v.emplace_back(std::vector<int>{7});
  v.emplace_back(std::vector<int>{});

  ASSERT_EQ(4U, v.size());
  EXPECT_EQ(3U, v.data_.size());  // default + {1, 2} + {}
  EXPECT_TRUE(std::ranges::equal(std::vector<int>{7}, v[transport_idx_t{0U}]));
  EXPECT_TRUE(
      std::ranges::equal(std::vector<int>{1, 2}, v[transport_idx_t{1U}]));
  EXPECT_TRUE(std::ranges::equal(std::vector<int>{7}, v[transport_idx_t{2U}]));
  EXPECT_TRUE(v[transport_idx_t{3U}].empty());
}

TEST(sparse_vecvec, from_buckets) {
  auto src = vecvec<transport_idx_t, int>{};
  src.emplace_back(std::vector<int>{});
  src.emplace_back(std::vector<int>{3});
  src.emplace_back(std::vector<int>{});

  auto const v = make_sparse_vecvec(src);
  static_assert(std::is_same_v<std::decay_t<decltype(v)>,
                               sparse_vecvec<transport_idx_t, int>>);
  ASSERT_EQ(3U, v.size());
  EXPECT_TRUE(v[transport_idx_t{0U}].empty());
  EXPECT_TRUE(std::ranges::equal(std::vector<int>{3}, v[transport_idx_t{1U}]));
  EXPECT_TRUE(v[transport_idx_t{2U}].empty());
}

TEST(sparse_vecvec, from_dedup) {
  auto d = dedup_vecvec<transport_idx_t, int>{};
  auto cache = d.make_cache();
  auto const ref =
      std::vector<std::vector<int>>{{1}, {}, {1}, {4, 5}, {1}, {}, {4, 5}, {1}};
  for (auto const& x : ref) {
    d.emplace_back(cache, x);
  }

  auto const v = make_sparse_vecvec(std::move(d));
  static_assert(std::is_same_v<std::decay_t<decltype(v)>,
                               sparse_dedup_vecvec<transport_idx_t, int>>);
  ASSERT_EQ(ref.size(), v.size());
  EXPECT_EQ(3U, v.data_.sequences_.size());
  EXPECT_EQ(5U, v.data_.sequence_.size());  // default {1} + 4 others
  for (auto i = 0U; i != ref.size(); ++i) {
    EXPECT_TRUE(std::ranges::equal(ref[i], v[transport_idx_t{i}])) << "i=" << i;
  }
}
