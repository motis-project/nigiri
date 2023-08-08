#pragma once

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

struct statistics {
  friend std::ostream& operator<<(std::ostream& out, statistics const& s) {
    auto first = true;
    auto const print_if_no_empty = [&](char const* name, auto const& value,
                                       bool print_percent = false) {
      if (!value) {
        return;
      }
      if (!first) {
        out << ", ";
      }
      first = false;
      out << name << "=" << value;
      if (print_percent && value) {
        out << " (" << value / static_cast<float>(s.total_entities_) * 100
            << "%)";
      }
    };

    print_if_no_empty("parser_error", s.parser_error_);
    print_if_no_empty("no_header", s.no_header_);
    print_if_no_empty("total_entities", s.total_entities_);
    print_if_no_empty("total_entities_success", s.total_entities_success_,
                      true);
    print_if_no_empty("total_entities_fail", s.total_entities_fail_, true);
    print_if_no_empty("unsupported_deleted", s.unsupported_deleted_, true);
    print_if_no_empty("unsupported_vehicle", s.unsupported_vehicle_, true);
    print_if_no_empty("unsupported_alert", s.unsupported_alert_, true);
    print_if_no_empty("unsupported_no_trip_id", s.unsupported_no_trip_id_,
                      true);
    print_if_no_empty("no_trip_update", s.no_trip_update_, true);
    print_if_no_empty("trip_update_without_trip", s.trip_update_without_trip_,
                      true);
    print_if_no_empty("trip_resolve_error", s.trip_resolve_error_, true);
    print_if_no_empty("unsupported_schedule_relationship",
                      s.unsupported_schedule_relationship_, true);

    return out;
  }

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
  int trip_resolve_error_{0};
  int unsupported_schedule_relationship_{0};
};

statistics gtfsrt_update_msg(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             transit_realtime::FeedMessage const&);

statistics gtfsrt_update_buf(timetable const&,
                             rt_timetable&,
                             source_idx_t const,
                             std::string_view tag,
                             std::string_view protobuf);

}  // namespace nigiri::rt
