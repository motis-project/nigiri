#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct query;
struct search_state;
struct journey;

template <direction SearchDir>
void reconstruct_journey(timetable const&,
                         query const&,
                         search_state const&,
                         journey&);

}  // namespace nigiri::routing
