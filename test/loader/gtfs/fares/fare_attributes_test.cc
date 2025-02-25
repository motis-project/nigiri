#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/fare_attributes.h"

using namespace nigiri::loader::gtfs::fares;

TEST(fares, fare_attributes) {
  std::string_view fare_attributes_content =
      R"(fare_id,price,currency_type,payment_method,transfers,agency_id,transfer_duration
base_fare,2.50,USD,1,0,agency1,
day_pass,10.00,USD,0,0,agency1,86400
premium_fare,3.25,USD,0,2,agency2,7200
limited_fare,1.75,USD,1,1,,3600
empty_fare,,,,,agency1,
)";

  auto const fare_attrs = read_fare_attributes(fare_attributes_content);

  ASSERT_EQ(5, fare_attrs.size());

  // Test base_fare
  {
    auto const& fare = fare_attrs[0];
    EXPECT_EQ("base_fare", fare.fare_id_);
    EXPECT_DOUBLE_EQ(2.5, fare.price_);
    EXPECT_EQ("USD", fare.currency_type_);
    EXPECT_EQ(payment_method::kBeforeBoarding, fare.payment_method_);
    EXPECT_EQ(transfers_type::kUnlimited, fare.transfers_);
    EXPECT_EQ("agency1", fare.agency_id_.value());
    EXPECT_FALSE(fare.transfer_duration_.has_value());
  }

  // Test day_pass
  {
    auto const& fare = fare_attrs[1];
    EXPECT_EQ("day_pass", fare.fare_id_);
    EXPECT_DOUBLE_EQ(10.0, fare.price_);
    EXPECT_EQ("USD", fare.currency_type_);
    EXPECT_EQ(payment_method::kOnBoard, fare.payment_method_);
    EXPECT_EQ(transfers_type::kUnlimited, fare.transfers_);
    EXPECT_EQ("agency1", fare.agency_id_.value());
    EXPECT_TRUE(fare.transfer_duration_.has_value());
    EXPECT_EQ(86400u, fare.transfer_duration_.value());
  }

  // Test premium_fare
  {
    auto const& fare = fare_attrs[2];
    EXPECT_EQ("premium_fare", fare.fare_id_);
    EXPECT_DOUBLE_EQ(3.25, fare.price_);
    EXPECT_EQ("USD", fare.currency_type_);
    EXPECT_EQ(payment_method::kOnBoard, fare.payment_method_);
    EXPECT_EQ(transfers_type::kOneTransfer, fare.transfers_);
    EXPECT_EQ("agency2", fare.agency_id_.value());
    EXPECT_TRUE(fare.transfer_duration_.has_value());
    EXPECT_EQ(7200u, fare.transfer_duration_.value());
  }

  // Test limited_fare
  {
    auto const& fare = fare_attrs[3];
    EXPECT_EQ("limited_fare", fare.fare_id_);
    EXPECT_DOUBLE_EQ(1.75, fare.price_);
    EXPECT_EQ("USD", fare.currency_type_);
    EXPECT_EQ(payment_method::kBeforeBoarding, fare.payment_method_);
    EXPECT_EQ(transfers_type::kNoTransfers, fare.transfers_);
    EXPECT_FALSE(fare.agency_id_.has_value());
    EXPECT_TRUE(fare.transfer_duration_.has_value());
    EXPECT_EQ(3600u, fare.transfer_duration_.value());
  }

  // Test empty_fare with defaults
  {
    auto const& fare = fare_attrs[4];
    EXPECT_EQ("empty_fare", fare.fare_id_);
    EXPECT_DOUBLE_EQ(0.0, fare.price_);
    EXPECT_TRUE(fare.currency_type_.empty());
    EXPECT_EQ(payment_method::kOnBoard, fare.payment_method_);
    EXPECT_EQ(transfers_type::kUnlimited, fare.transfers_);
    EXPECT_EQ("agency1", fare.agency_id_.value());
    EXPECT_FALSE(fare.transfer_duration_.has_value());
  }
}