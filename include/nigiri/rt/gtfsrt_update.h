#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

struct statistics {
  bool parser_error_{false};
  bool no_header_{false};
  int total_entities_{0};
  int total_entities_success_{0};
  int total_entities_fail_{0};
  int unsupported_deleted_{0};
  int unsupported_vehicle_{0};
  int unsupported_alert_{0};
  int unsupported_no_trip_id_{0};
  int no_trip_update_{0};
  int trip_update_without_trip_{0};
};

statistics gtfsrt_update(timetable const&,
                         rt_timetable&,
                         source_idx_t const,
                         std::string_view tag,
                         std::string_view protobuf);

}  // namespace nigiri::rt
