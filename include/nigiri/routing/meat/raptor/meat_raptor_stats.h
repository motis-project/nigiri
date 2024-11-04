#pragma once

#include <cstdint>

namespace nigiri::routing::meat::raptor {

struct meat_raptor_stats {
  void reset() {
    total_duration_ = 0ULL;
  }

  std::uint64_t total_duration_{0ULL};
};

}  // namespace nigiri::routing::meat::raptor
