#pragma once

#include <string>
#include <string_view>

#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct transfer {
  enum class type : std::uint8_t {
    kRecommended = 0U,
    kTimed = 1U,
    kMinimumChangeTime = 2U,
    kNotPossible = 3U,
    kStaySeated = 4U,
    kNoStaySeated = 5U,
    kGenerated = std::numeric_limits<std::uint8_t>::max()
  } type_;
  u8_minutes minutes_;
};

using stop_pair = std::pair<stop const*, stop const*>;
hash_map<stop_pair, transfer> read_transfers(stop_map const&,
                                             std::string_view file_content);

}  // namespace nigiri::loader::gtfs
