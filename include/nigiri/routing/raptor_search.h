#pragma once

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

enum class raptor_variant { kAdvanced, kSimple };

routing_result<raptor_stats> raptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    direction search_dir,
    raptor_variant const variant = raptor_variant::kAdvanced,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace nigiri::routing
