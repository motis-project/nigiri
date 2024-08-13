#pragma once

#include "pugixml.hpp"

#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt::vdv {

struct statistics {
  friend std::ostream& operator<<(std::ostream& out, statistics const& s) {
    out << "unsupported additional runs: " << s.unsupported_additional_runs_
        << "\nunsupported cancelled runs: " << s.unsupported_cancelled_runs_
        << "\ntotal stops: " << s.total_stops_
        << "\nresolved stops: " << s.resolved_stops_
        << "\nunknown stops: " << s.unknown_stops_
        << "\nunsupported additional stops: " << s.unsupported_additional_stops_
        << "\nno transport found at stop: " << s.no_transport_found_at_stop_
        << "\nmatch prevented by line id: " << s.match_prevented_by_line_id_
        << "\ntotal runs: " << s.total_runs_
        << "\nmatched runs: " << s.matched_runs_
        << "\nunmatchable runs: " << s.unmatchable_runs_
        << "\nskipped vdv stops: " << s.skipped_vdv_stops_
        << "\nexcess vdv stops: " << s.excess_vdv_stops_
        << "\nupdated events: " << s.updated_events_
        << "\npropagated delays: " << s.propagated_delays_ << "\n";
    return out;
  }

  friend statistics& operator+=(statistics& lhs, statistics const& rhs) {
    lhs.unsupported_additional_runs_ += rhs.unsupported_additional_runs_;
    lhs.unsupported_cancelled_runs_ += rhs.unsupported_cancelled_runs_;
    lhs.total_stops_ += rhs.total_stops_;
    lhs.resolved_stops_ += rhs.resolved_stops_;
    lhs.unknown_stops_ += rhs.unknown_stops_;
    lhs.unsupported_additional_stops_ += rhs.unsupported_additional_stops_;
    lhs.total_runs_ += rhs.total_runs_;
    lhs.no_transport_found_at_stop_ += rhs.no_transport_found_at_stop_;
    lhs.match_prevented_by_line_id_ += rhs.match_prevented_by_line_id_;
    lhs.matched_runs_ += rhs.matched_runs_;
    lhs.unmatchable_runs_ += rhs.unmatchable_runs_;
    lhs.skipped_vdv_stops_ += rhs.skipped_vdv_stops_;
    lhs.excess_vdv_stops_ += rhs.excess_vdv_stops_;
    lhs.updated_events_ += rhs.updated_events_;
    lhs.propagated_delays_ += rhs.propagated_delays_;
    return lhs;
  }

  std::uint32_t unsupported_additional_runs_{0U};
  std::uint32_t unsupported_cancelled_runs_{0U};
  std::uint32_t total_stops_{0U};
  std::uint32_t resolved_stops_{0U};
  std::uint32_t unknown_stops_{0U};
  std::uint32_t unsupported_additional_stops_{0U};
  std::uint32_t total_runs_{0U};
  std::uint32_t no_transport_found_at_stop_{0U};
  std::uint32_t match_prevented_by_line_id_{0U};
  std::uint32_t matched_runs_{0U};
  std::uint32_t unmatchable_runs_{0U};
  std::uint32_t skipped_vdv_stops_{0U};
  std::uint32_t excess_vdv_stops_{0U};
  std::uint32_t updated_events_{0U};
  std::uint32_t propagated_delays_{0U};
};

statistics vdv_update(timetable const&,
                      rt_timetable&,
                      source_idx_t,
                      pugi::xml_document const&);

}  // namespace nigiri::rt::vdv