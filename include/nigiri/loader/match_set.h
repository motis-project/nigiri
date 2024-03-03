#include "nigiri/types.h"

namespace nigiri::loader {

using match_set_t = hash_set<pair<location_idx_t, location_idx_t>>;

inline pair<location_idx_t, location_idx_t> make_match_pair(
    location_idx_t const a, location_idx_t const b) {
  return {std::min(a, b), std::max(a, b)};
}

}  // namespace nigiri::loader