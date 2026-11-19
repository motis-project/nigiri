#include "gtest/gtest.h"

#include <algorithm>

#include "nigiri/common/dedup_vecvec.h"

using namespace nigiri;

TEST(dedup_vecvec, identical_buckets_stored_once) {
  using seq = basic_string<provider_idx_t>;

  auto v = dedup_vecvec<transport_idx_t, provider_idx_t>{};
  auto cache = v.make_cache();

  auto const a = seq{provider_idx_t{1U}, provider_idx_t{2U}};
  auto const b = seq{provider_idx_t{3U}};
  auto const empty = seq{};
  v.emplace_back(cache, a);
  v.emplace_back(cache, b);
  v.emplace_back(cache, empty);
  v.emplace_back(cache, a);
  v.emplace_back(cache, empty);
  v.emplace_back(cache, b);

  ASSERT_EQ(6U, v.size());
  EXPECT_EQ(3U, v.sequences_.size());
  EXPECT_EQ(v.sequence_[transport_idx_t{0U}], v.sequence_[transport_idx_t{3U}]);
  EXPECT_EQ(v.sequence_[transport_idx_t{1U}], v.sequence_[transport_idx_t{5U}]);
  EXPECT_EQ(v.sequence_[transport_idx_t{2U}], v.sequence_[transport_idx_t{4U}]);
  EXPECT_TRUE(std::ranges::equal(a, v[transport_idx_t{3U}]));
  EXPECT_TRUE(std::ranges::equal(b, v.at(transport_idx_t{5U})));
  EXPECT_TRUE(v[transport_idx_t{4U}].empty());
}
