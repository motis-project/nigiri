#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/fare_leg_join_rules.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, fare_leg_join_rules_parse_basic_fields) {
  auto const content = R"(fare_leg_rule_id,from_leg_price_group_id,to_leg_price_group_id,fare_leg_rule_sequence
rule1,group1,group2,1
rule2,group3,group4,2
rule3,group5,group6,3)";

  auto const result = read_fare_leg_join_rules(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_EQ("group1", result[0].from_leg_price_group_id_.value());
  EXPECT_EQ("group2", result[0].to_leg_price_group_id_.value());
  EXPECT_EQ(1U, result[0].fare_leg_rule_sequence_.value());
  
  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("group3", result[1].from_leg_price_group_id_.value());
  EXPECT_EQ("group4", result[1].to_leg_price_group_id_.value());
  EXPECT_EQ(2U, result[1].fare_leg_rule_sequence_.value());
  
  EXPECT_EQ("rule3", result[2].fare_leg_rule_id_);
  EXPECT_EQ("group5", result[2].from_leg_price_group_id_.value());
  EXPECT_EQ("group6", result[2].to_leg_price_group_id_.value());
  EXPECT_EQ(3U, result[2].fare_leg_rule_sequence_.value());
}

TEST(fares, fare_leg_join_rules_parse_empty_optional_fields) {
  auto const content = R"(fare_leg_rule_id,from_leg_price_group_id,to_leg_price_group_id,fare_leg_rule_sequence
rule1,,,
rule2,group3,,2
rule3,,group6,)";

  auto const result = read_fare_leg_join_rules(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("rule1", result[0].fare_leg_rule_id_);
  EXPECT_FALSE(result[0].from_leg_price_group_id_.has_value());
  EXPECT_FALSE(result[0].to_leg_price_group_id_.has_value());
  EXPECT_FALSE(result[0].fare_leg_rule_sequence_.has_value());
  
  EXPECT_EQ("rule2", result[1].fare_leg_rule_id_);
  EXPECT_EQ("group3", result[1].from_leg_price_group_id_.value());
  EXPECT_FALSE(result[1].to_leg_price_group_id_.has_value());
  EXPECT_EQ(2U, result[1].fare_leg_rule_sequence_.value());
  
  EXPECT_EQ("rule3", result[2].fare_leg_rule_id_);
  EXPECT_FALSE(result[2].from_leg_price_group_id_.has_value());
  EXPECT_EQ("group6", result[2].to_leg_price_group_id_.value());
  EXPECT_FALSE(result[2].fare_leg_rule_sequence_.has_value());
}

}  // namespace nigiri::loader::gtfs::fares