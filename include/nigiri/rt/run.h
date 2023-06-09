#pragma once

#include <variant>

#include "nigiri/types.h"

namespace nigiri::rt {

struct run {
  bool is_rt() const noexcept { return rt_.has_value(); }
  bool valid() const noexcept { return t_.has_value() || rt_.has_value(); }

  // from static timetable, not set for additional services
  std::optional<transport> t_;

  // real-time instance, not set if no real-time info available
  std::optional<rt_transport_idx_t> rt_;
};

}  // namespace nigiri::rt