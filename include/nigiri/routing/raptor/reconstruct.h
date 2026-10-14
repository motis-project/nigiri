#pragma once

#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct query;
struct search_state;
struct raptor_state;
struct journey;

void map_to_bases(timetable const&, rt_timetable const*, journey&);

bool is_journey_start(timetable const&,
                      query const&,
                      location_idx_t candidate_l);

template <direction SearchDir>
void reconstruct_journey(timetable const&,
                         rt_timetable const*,
                         query const&,
                         raptor_state const&,
                         journey&,
                         date::sys_days const base,
                         day_idx_t const base_day_idx);

// Where the vias of q (in journey order) hold on j: each at the first place
// it can - passing one of its stops on a trip (no stay), or a change or walk
// at one of its stops (from_ before to_) that leaves time for its stay.
struct via_counts {
  std::vector<via_offset_t> before_;  // vias counted before leg i
  std::vector<duration_t> stay_;  // via stay at the start of walk/change i
};

via_counts count_vias(timetable const&,
                      rt_timetable const*,
                      query const&,
                      journey const&);

void optimize_footpaths(timetable const&,
                        rt_timetable const*,
                        query const&,
                        journey&);

void specify_td_offsets(query const&, journey&);

}  // namespace nigiri::routing
