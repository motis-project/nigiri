#pragma once

#include "nigiri/loader/virt/route_stop_pairs.h"
#include "nigiri/loader/virt/transfer_durations.h"
#include "nigiri/types.h"

namespace nigiri::loader {

vector_map<route_stop_idx_t, route_stop_idx_t> merge_locations(
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs&);

}  // namespace nigiri::loader
