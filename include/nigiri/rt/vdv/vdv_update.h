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
    out << "unsupported additional run: " << s.unsupported_additional_run_
        << "\nunsupported cancelled run: " << s.unsupported_cancelled_run_
        << "\ntotal stops: " << s.total_stops_
        << "\nmatched stop: " << s.matched_stops_
        << "\nunknown stops: " << s.unknown_stops_
        << "\nunsupported additional stop: " << s.unsupported_additional_stops_
        << "\nno transport found at stop: " << s.no_transport_found_at_stop_
        << "\nunmatchable run: " << s.unmatchable_run_
        << "\nexcess vdv stop: " << s.excess_vdv_stop_
        << "\nupdated event: " << s.updated_event_
        << "\npropagated delay: " << s.propagated_delay_ << "\n";
    return out;
  }

  friend statistics operator+(statistics lhs, statistics const& rhs) {
    return {
        lhs.unsupported_additional_run_ += rhs.unsupported_additional_run_,
        lhs.unsupported_cancelled_run_ += rhs.unsupported_cancelled_run_,
        lhs.total_stops_ += rhs.total_stops_,
        lhs.matched_stops_ += rhs.matched_stops_,
        lhs.unknown_stops_ += rhs.unknown_stops_,
        lhs.unsupported_additional_stops_ += rhs.unsupported_additional_run_,
        lhs.no_transport_found_at_stop_ += rhs.no_transport_found_at_stop_,
        lhs.unmatchable_run_ += rhs.unmatchable_run_,
        lhs.excess_vdv_stop_ += rhs.excess_vdv_stop_,
        lhs.updated_event_ += rhs.updated_event_,
        lhs.propagated_delay_ += rhs.propagated_delay_};
  }

  std::uint32_t unsupported_additional_run_{0U};
  std::uint32_t unsupported_cancelled_run_{0U};
  std::uint32_t total_stops_{0U};
  std::uint32_t matched_stops_{0U};
  std::uint32_t unknown_stops_{0};
  std::uint32_t unsupported_additional_stops_{0U};
  std::uint32_t no_transport_found_at_stop_{0};
  std::uint32_t unmatchable_run_{0U};
  std::uint32_t excess_vdv_stop_{0U};
  std::uint32_t updated_event_{0U};
  std::uint32_t propagated_delay_{0U};
};

statistics vdv_update(timetable const&,
                      rt_timetable&,
                      source_idx_t,
                      pugi::xml_document const&);

}  // namespace nigiri::rt::vdv