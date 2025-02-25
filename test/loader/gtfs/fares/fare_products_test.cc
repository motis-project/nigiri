#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/fare_products.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, fare_products) {
  auto const content = R"(fare_product_id,amount,currency
product1,10.5,USD
product2,15.75,EUR)";

  auto const result = read_fare_products(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_DOUBLE_EQ(10.5, result[0].amount_);
  EXPECT_EQ("USD", result[0].currency_);
  EXPECT_FALSE(result[0].fare_product_name_.has_value());
  EXPECT_FALSE(result[0].rider_category_id_.has_value());
  EXPECT_FALSE(result[0].timeframe_id_.has_value());
  EXPECT_FALSE(result[0].fare_media_id_.has_value());

  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_DOUBLE_EQ(15.75, result[1].amount_);
  EXPECT_EQ("EUR", result[1].currency_);
}

TEST(fares, fare_products_parse_all_fields) {
  auto const content =
      R"(fare_product_id,fare_product_name,amount,currency,rider_category_id,timeframe_id,fare_media_id
product1,Day Pass,10.5,USD,adult,day_pass,card
product2,Student Ticket,5.25,EUR,student,single_ride,app)";

  auto const result = read_fare_products(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_EQ("Day Pass", result[0].fare_product_name_.value());
  EXPECT_DOUBLE_EQ(10.5, result[0].amount_);
  EXPECT_EQ("USD", result[0].currency_);
  EXPECT_EQ("adult", result[0].rider_category_id_.value());
  EXPECT_EQ("day_pass", result[0].timeframe_id_.value());
  EXPECT_EQ("card", result[0].fare_media_id_.value());

  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_EQ("Student Ticket", result[1].fare_product_name_.value());
  EXPECT_DOUBLE_EQ(5.25, result[1].amount_);
  EXPECT_EQ("EUR", result[1].currency_);
  EXPECT_EQ("student", result[1].rider_category_id_.value());
  EXPECT_EQ("single_ride", result[1].timeframe_id_.value());
  EXPECT_EQ("app", result[1].fare_media_id_.value());
}

TEST(fares, fare_products_parse_empty_optional_fields) {
  auto const content =
      R"(fare_product_id,fare_product_name,amount,currency,rider_category_id,timeframe_id,fare_media_id
product1,,10.5,USD,,,
product2,Student Ticket,15.75,EUR,,single_ride,)";

  auto const result = read_fare_products(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("product1", result[0].fare_product_id_);
  EXPECT_FALSE(result[0].fare_product_name_.has_value());
  EXPECT_DOUBLE_EQ(10.5, result[0].amount_);
  EXPECT_EQ("USD", result[0].currency_);
  EXPECT_FALSE(result[0].rider_category_id_.has_value());
  EXPECT_FALSE(result[0].timeframe_id_.has_value());
  EXPECT_FALSE(result[0].fare_media_id_.has_value());

  EXPECT_EQ("product2", result[1].fare_product_id_);
  EXPECT_EQ("Student Ticket", result[1].fare_product_name_.value());
  EXPECT_DOUBLE_EQ(15.75, result[1].amount_);
  EXPECT_EQ("EUR", result[1].currency_);
  EXPECT_FALSE(result[1].rider_category_id_.has_value());
  EXPECT_EQ("single_ride", result[1].timeframe_id_.value());
  EXPECT_FALSE(result[1].fare_media_id_.has_value());
}

}  // namespace nigiri::loader::gtfs::fares