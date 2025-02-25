#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/fare_media.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, fare_media) {
  auto const content = R"(fare_media_id,fare_media_name,fare_media_type
media1,Smart Card,0
media2,Mobile App,1)";

  auto const result = read_fare_media(content);
  ASSERT_EQ(2, result.size());

  EXPECT_EQ("media1", result[0].fare_media_id_);
  EXPECT_EQ("Smart Card", result[0].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kPhysical, result[0].media_type_);
  EXPECT_FALSE(result[0].restrictions_.has_value());

  EXPECT_EQ("media2", result[1].fare_media_id_);
  EXPECT_EQ("Mobile App", result[1].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kVirtual, result[1].media_type_);
  EXPECT_FALSE(result[1].restrictions_.has_value());
}

TEST(fares, fare_media_parse_with_restrictions) {
  auto const content =
      R"(fare_media_id,fare_media_name,fare_media_type,fare_media_restrictions
media1,Smart Card,0,0
media2,Mobile App,1,1
media3,Paper Ticket,0,2)";

  auto const result = read_fare_media(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("media1", result[0].fare_media_id_);
  EXPECT_EQ("Smart Card", result[0].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kPhysical, result[0].media_type_);
  EXPECT_EQ(fare_media_restriction::kNone, result[0].restrictions_.value());

  EXPECT_EQ("media2", result[1].fare_media_id_);
  EXPECT_EQ("Mobile App", result[1].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kVirtual, result[1].media_type_);
  EXPECT_EQ(fare_media_restriction::kReserveFirstUse,
            result[1].restrictions_.value());

  EXPECT_EQ("media3", result[2].fare_media_id_);
  EXPECT_EQ("Paper Ticket", result[2].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kPhysical, result[2].media_type_);
  EXPECT_EQ(fare_media_restriction::kReserveBeforeUse,
            result[2].restrictions_.value());
}

TEST(fares, fare_media_parse_with_empty_optional_fields) {
  auto const content =
      R"(fare_media_id,fare_media_name,fare_media_type,fare_media_restrictions
media1,,0,
media2,Mobile App,1,1
media3,,0,2)";

  auto const result = read_fare_media(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("media1", result[0].fare_media_id_);
  EXPECT_FALSE(result[0].fare_media_name_.has_value());
  EXPECT_EQ(fare_media_type::kPhysical, result[0].media_type_);
  EXPECT_FALSE(result[0].restrictions_.has_value());

  EXPECT_EQ("media2", result[1].fare_media_id_);
  EXPECT_EQ("Mobile App", result[1].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kVirtual, result[1].media_type_);
  EXPECT_EQ(fare_media_restriction::kReserveFirstUse,
            result[1].restrictions_.value());

  EXPECT_EQ("media3", result[2].fare_media_id_);
  EXPECT_FALSE(result[2].fare_media_name_.has_value());
  EXPECT_EQ(fare_media_type::kPhysical, result[2].media_type_);
  EXPECT_EQ(fare_media_restriction::kReserveBeforeUse,
            result[2].restrictions_.value());
}

TEST(fares, fare_media_parse_invalid_media_type) {
  auto const content = R"(fare_media_id,fare_media_name,fare_media_type
media1,Invalid Type,3)";

  auto const result = read_fare_media(content);
  ASSERT_EQ(1, result.size());

  // Invalid media type should default to kPhysical
  EXPECT_EQ("media1", result[0].fare_media_id_);
  EXPECT_EQ("Invalid Type", result[0].fare_media_name_.value());
  EXPECT_EQ(fare_media_type::kPhysical, result[0].media_type_);
}

}  // namespace nigiri::loader::gtfs::fares