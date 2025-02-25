#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

enum class payment_method : std::uint8_t { kOnBoard = 0, kBeforeBoarding = 1 };

enum class transfers_type : std::uint8_t {
  kUnlimited = 0,
  kNoTransfers = 1,
  kOneTransfer = 2,
  kTwoTransfers = 3
};

struct fare_attribute {
  std::string_view fare_id_;
  double price_{0.0};
  std::string_view currency_type_;
  payment_method payment_method_{payment_method::kOnBoard};
  transfers_type transfers_{transfers_type::kUnlimited};
  std::optional<unsigned> transfer_duration_;  // in seconds
  std::optional<std::string_view> agency_id_;
};

std::vector<fare_attribute> read_fare_attributes(std::string_view);

}  // namespace nigiri::loader::gtfs::fares