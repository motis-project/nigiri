#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/networks.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, networks_parse_non_empty) {
  auto const content = R"(network_id,network_name
network1,Metro
network2,Bus
network3,Rail)";

  auto const result = read_networks(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("network1", result[0].network_id_);
  EXPECT_EQ("Metro", result[0].network_name_.value());

  EXPECT_EQ("network2", result[1].network_id_);
  EXPECT_EQ("Bus", result[1].network_name_.value());

  EXPECT_EQ("network3", result[2].network_id_);
  EXPECT_EQ("Rail", result[2].network_name_.value());
}

TEST(fares, networks_parse_empty) {
  auto const content = R"(network_id,network_name
network1,
network2,Bus
network3,)";

  auto const result = read_networks(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("network1", result[0].network_id_);
  EXPECT_FALSE(result[0].network_name_.has_value());

  EXPECT_EQ("network2", result[1].network_id_);
  EXPECT_EQ("Bus", result[1].network_name_.value());

  EXPECT_EQ("network3", result[2].network_id_);
  EXPECT_FALSE(result[2].network_name_.has_value());
}

}  // namespace nigiri::loader::gtfs::fares