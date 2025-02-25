#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/route_networks.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, route_networks) {
  auto const content = R"(route_id,network_id
route1,network1
route2,network1
route3,network2
route4,network3)";

  auto const result = read_route_networks(content);
  ASSERT_EQ(4, result.size());

  EXPECT_EQ("route1", result[0].route_id_);
  EXPECT_EQ("network1", result[0].network_id_);

  EXPECT_EQ("route2", result[1].route_id_);
  EXPECT_EQ("network1", result[1].network_id_);

  EXPECT_EQ("route3", result[2].route_id_);
  EXPECT_EQ("network2", result[2].network_id_);

  EXPECT_EQ("route4", result[3].route_id_);
  EXPECT_EQ("network3", result[3].network_id_);
}

TEST(fares, route_networks_empty) {
  auto const content = R"(route_id,network_id)";

  auto const result = read_route_networks(content);
  EXPECT_TRUE(result.empty());
}

}  // namespace nigiri::loader::gtfs::fares