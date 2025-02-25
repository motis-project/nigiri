#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/fare_rules.h"

using namespace nigiri::loader::gtfs::fares;

TEST(fares, fare_rules) {
  std::string_view fare_rules_content =
      R"(fare_id,route_id,origin_id,destination_id,contains_id
base_fare,R1,,,
base_fare,R2,,,
premium_fare,R3,O1,D1,
premium_fare,R3,O1,D2,C1
premium_fare,R3,O2,D1,C1
zone_fare,,Z1,Z2,
zone_fare,,Z2,Z3,
zone_fare,,Z1,Z3,)";

  auto const rules = read_fare_rules(fare_rules_content);

  ASSERT_EQ(8, rules.size());

  // Test base_fare with route R1
  {
    auto const& rule = rules[0];
    EXPECT_EQ("base_fare", rule.fare_id_);
    EXPECT_EQ("R1", rule.route_id_.value());
    EXPECT_FALSE(rule.origin_id_.has_value());
    EXPECT_FALSE(rule.destination_id_.has_value());
    EXPECT_FALSE(rule.contains_id_.has_value());
  }

  // Test base_fare with route R2
  {
    auto const& rule = rules[1];
    EXPECT_EQ("base_fare", rule.fare_id_);
    EXPECT_EQ("R2", rule.route_id_.value());
    EXPECT_FALSE(rule.origin_id_.has_value());
    EXPECT_FALSE(rule.destination_id_.has_value());
    EXPECT_FALSE(rule.contains_id_.has_value());
  }

  // Test premium_fare with origin O1, destination D1
  {
    auto const& rule = rules[2];
    EXPECT_EQ("premium_fare", rule.fare_id_);
    EXPECT_EQ("R3", rule.route_id_.value());
    EXPECT_EQ("O1", rule.origin_id_.value());
    EXPECT_EQ("D1", rule.destination_id_.value());
    EXPECT_FALSE(rule.contains_id_.has_value());
  }

  // Test premium_fare with origin O1, destination D2, contains C1
  {
    auto const& rule = rules[3];
    EXPECT_EQ("premium_fare", rule.fare_id_);
    EXPECT_EQ("R3", rule.route_id_.value());
    EXPECT_EQ("O1", rule.origin_id_.value());
    EXPECT_EQ("D2", rule.destination_id_.value());
    EXPECT_EQ("C1", rule.contains_id_.value());
  }

  // Test premium_fare with origin O2, destination D1, contains C1
  {
    auto const& rule = rules[4];
    EXPECT_EQ("premium_fare", rule.fare_id_);
    EXPECT_EQ("R3", rule.route_id_.value());
    EXPECT_EQ("O2", rule.origin_id_.value());
    EXPECT_EQ("D1", rule.destination_id_.value());
    EXPECT_EQ("C1", rule.contains_id_.value());
  }

  // Test zone_fare with origin Z1, destination Z2
  {
    auto const& rule = rules[5];
    EXPECT_EQ("zone_fare", rule.fare_id_);
    EXPECT_FALSE(rule.route_id_.has_value());
    EXPECT_EQ("Z1", rule.origin_id_.value());
    EXPECT_EQ("Z2", rule.destination_id_.value());
    EXPECT_FALSE(rule.contains_id_.has_value());
  }

  // Test zone_fare with origin Z2, destination Z3
  {
    auto const& rule = rules[6];
    EXPECT_EQ("zone_fare", rule.fare_id_);
    EXPECT_FALSE(rule.route_id_.has_value());
    EXPECT_EQ("Z2", rule.origin_id_.value());
    EXPECT_EQ("Z3", rule.destination_id_.value());
    EXPECT_FALSE(rule.contains_id_.has_value());
  }

  // Test zone_fare with origin Z1, destination Z3
  {
    auto const& rule = rules[7];
    EXPECT_EQ("zone_fare", rule.fare_id_);
    EXPECT_FALSE(rule.route_id_.has_value());
    EXPECT_EQ("Z1", rule.origin_id_.value());
    EXPECT_EQ("Z3", rule.destination_id_.value());
    EXPECT_FALSE(rule.contains_id_.has_value());
  }
}