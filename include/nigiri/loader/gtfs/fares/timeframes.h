#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs::fares {

struct timeframe {
  std::string_view timeframe_id_;
  std::optional<std::string_view> timeframe_name_;
  std::optional<duration_t> timeframe_start_time_;
  std::optional<duration_t> timeframe_end_time_;
  std::optional<unsigned> timeframe_duration_;  // in seconds
  std::optional<bool> timeframe_disable_after_purchase_{false};
};

std::vector<timeframe> read_timeframes(std::string_view);

}  // namespace nigiri::loader::gtfs::fares