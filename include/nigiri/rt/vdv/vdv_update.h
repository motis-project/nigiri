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
        << "\nunsupported additional stop: " << s.unsupported_additional_stop_
        << "\nrun with additional stops only: "
        << s.run_with_additional_stops_only_
        << "\ntext81_stop: " << s.text81_stop_
        << "\nunknown stop id: " << s.unknown_stop_id_
        << "\nfound stop but no transport: " << s.found_stop_but_no_transport_
        << "\nunmatchable run: " << s.unmatchable_run_
        << "\nexcess vdv stop: " << s.excess_vdv_stop_
        << "\nmatched stop: " << s.full_match_stop_
        << "\nprefix match stop: " << s.prefix_match_stop_
        << "\npropagated delay: " << s.propagated_delay_ << "\n";
    return out;
  }

  friend statistics operator+(statistics lhs, statistics const& rhs) {
    return {
        lhs.unsupported_additional_run_ += rhs.unsupported_additional_run_,
        lhs.unsupported_cancelled_run_ += rhs.unsupported_cancelled_run_,
        lhs.unsupported_additional_stop_ += rhs.unsupported_additional_run_,
        lhs.run_with_additional_stops_only_ +=
        rhs.run_with_additional_stops_only_,
        lhs.text81_stop_ += rhs.text81_stop_,
        lhs.unknown_stop_id_ += rhs.unknown_stop_id_,
        lhs.found_stop_but_no_transport_ += rhs.found_stop_but_no_transport_,
        lhs.unmatchable_run_ += rhs.unmatchable_run_,
        lhs.excess_vdv_stop_ += rhs.excess_vdv_stop_,
        lhs.full_match_stop_ += rhs.full_match_stop_,
        lhs.prefix_match_stop_ += rhs.prefix_match_stop_,
        lhs.propagated_delay_ += rhs.propagated_delay_};
  }

  std::uint32_t unsupported_additional_run_{0U};
  std::uint32_t unsupported_cancelled_run_{0U};
  std::uint32_t unsupported_additional_stop_{0U};
  std::uint32_t run_with_additional_stops_only_{0U};
  std::uint32_t text81_stop_{0U};
  std::uint32_t unknown_stop_id_{0};
  std::uint32_t found_stop_but_no_transport_{0};
  std::uint32_t unmatchable_run_{0U};
  std::uint32_t excess_vdv_stop_{0U};
  std::uint32_t full_match_stop_{0U};
  std::uint32_t prefix_match_stop_{0U};
  std::uint32_t propagated_delay_{0U};
};

statistics vdv_update(timetable const&,
                      rt_timetable&,
                      source_idx_t,
                      pugi::xml_document const&);

}  // namespace nigiri::rt::vdv