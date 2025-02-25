#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/fare_leg_rules.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, fare_leg_rules) {
  auto const content = R"(fare_leg_rule_id,fare_product_id
rule1,product1
rule2,product2)";

  auto const result = read_fare_leg_rules(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_FALSE(result[0].leg_group_id_.has_value());
  EXPECT_FALSE(result[0].network_id_.has_value());
  EXPECT_FALSE(result[0].from_area_id_.has_value());
  EXPECT_FALSE(result[0].to_area_id_.has_value());
  EXPECT_FALSE(result[0].route_id_.has_value());
  EXPECT_FALSE(result[0].contains_area_id_.has_value());
  EXPECT_FALSE(result[0].contains_area_type_.has_value());
  EXPECT_FALSE(result[0].contains_route_id_.has_value());

  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("product2", result[1].fare_product_id_);
}

TEST(fares, fare_leg_rules_parse_with_group_and_area) {
  auto const content =
      R"(fare_leg_rule_id,fare_product_id,leg_group_id,from_area_id,to_area_id
rule1,product1,group1,area1,area2
rule2,product2,group2,area3,area4)";

  auto const result = read_fare_leg_rules(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_EQ("group1", result[0].leg_group_id_.value());
  EXPECT_EQ("area1", result[0].from_area_id_.value());
  EXPECT_EQ("area2", result[0].to_area_id_.value());

  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_EQ("group2", result[1].leg_group_id_.value());
  EXPECT_EQ("area3", result[1].from_area_id_.value());
  EXPECT_EQ("area4", result[1].to_area_id_.value());
}

TEST(fares, fare_leg_rules_parse_with_contains_area_fields) {
  auto const content =
      R"(fare_leg_rule_id,fare_product_id,contains_area_id,contains_area_type
rule1,product1,area1,0
rule2,product2,area2,1)";

  auto const result = read_fare_leg_rules(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_EQ("area1", result[0].contains_area_id_.value());
  EXPECT_EQ(contains_area_type::kAny, result[0].contains_area_type_.value());

  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_EQ("area2", result[1].contains_area_id_.value());
  EXPECT_EQ(contains_area_type::kAll, result[1].contains_area_type_.value());
}

TEST(fares, fare_leg_rules_parse_all_fields) {
  auto const content =
      R"(fare_leg_rule_id,fare_product_id,leg_group_id,network_id,from_area_id,to_area_id,route_id,contains_area_id,contains_area_type,contains_route_id
rule1,product1,group1,network1,area1,area2,route1,area3,0,route2
rule2,product2,group2,network2,area4,area5,route3,area6,1,route4)";

  auto const result = read_fare_leg_rules(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_EQ("group1", result[0].leg_group_id_.value());
  EXPECT_EQ("network1", result[0].network_id_.value());
  EXPECT_EQ("area1", result[0].from_area_id_.value());
  EXPECT_EQ("area2", result[0].to_area_id_.value());
  EXPECT_EQ("route1", result[0].route_id_.value());
  EXPECT_EQ("area3", result[0].contains_area_id_.value());
  EXPECT_EQ(contains_area_type::kAny, result[0].contains_area_type_.value());
  EXPECT_EQ("route2", result[0].contains_route_id_.value());

  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_EQ("group2", result[1].leg_group_id_.value());
  EXPECT_EQ("network2", result[1].network_id_.value());
  EXPECT_EQ("area4", result[1].from_area_id_.value());
  EXPECT_EQ("area5", result[1].to_area_id_.value());
  EXPECT_EQ("route3", result[1].route_id_.value());
  EXPECT_EQ("area6", result[1].contains_area_id_.value());
  EXPECT_EQ(contains_area_type::kAll, result[1].contains_area_type_.value());
  EXPECT_EQ("route4", result[1].contains_route_id_.value());
}

TEST(fares, fare_leg_rules_parse_empty_optional_fields) {
  auto const content =
      R"(fare_leg_rule_id,fare_product_id,leg_group_id,network_id,from_area_id,to_area_id,route_id,contains_area_id,contains_area_type,contains_route_id
rule1,product1,,network1,,area2,,,0,
rule2,product2,group2,,area4,,route3,area6,,route4)";

  auto const result = read_fare_leg_rules(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_FALSE(result[0].leg_group_id_.has_value());
  EXPECT_EQ("network1", result[0].network_id_.value());
  EXPECT_FALSE(result[0].from_area_id_.has_value());
  EXPECT_EQ("area2", result[0].to_area_id_.value());
  EXPECT_FALSE(result[0].route_id_.has_value());
  EXPECT_FALSE(result[0].contains_area_id_.has_value());
  EXPECT_EQ(contains_area_type::kAny, result[0].contains_area_type_.value());
  EXPECT_FALSE(result[0].contains_route_id_.has_value());

  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_EQ("group2", result[1].leg_group_id_.value());
  EXPECT_FALSE(result[1].network_id_.has_value());
  EXPECT_EQ("area4", result[1].from_area_id_.value());
  EXPECT_FALSE(result[1].to_area_id_.has_value());
  EXPECT_EQ("route3", result[1].route_id_.value());
  EXPECT_EQ("area6", result[1].contains_area_id_.value());
  EXPECT_FALSE(result[1].contains_area_type_.has_value());
  EXPECT_EQ("route4", result[1].contains_route_id_.value());
}

}  // namespace nigiri::loader::gtfs::fares