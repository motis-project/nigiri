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
        << "\nunmatchable run: " << s.unmatchable_run_
        << "\nexcess vdv stop: " << s.excess_vdv_stop_
        << "\nmatched stop: " << s.matched_stop_
        << "\npropagated delay: " << s.propagated_delay_ << "\n";
    return out;
  }

  friend statistics operator+(statistics lhs, statistics const& rhs) {
    return {lhs.unsupported_additional_run_ += rhs.unsupported_additional_run_,
            lhs.unsupported_cancelled_run_ += rhs.unsupported_cancelled_run_,
            lhs.unsupported_additional_stop_ += rhs.unsupported_additional_run_,
            lhs.unmatchable_run_ += rhs.unmatchable_run_,
            lhs.excess_vdv_stop_ += rhs.excess_vdv_stop_,
            lhs.matched_stop_ += rhs.matched_stop_,
            lhs.propagated_delay_ += rhs.propagated_delay_};
  }

  std::uint32_t unsupported_additional_run_{0U};
  std::uint32_t unsupported_cancelled_run_{0U};
  std::uint32_t unsupported_additional_stop_{0U};
  std::uint32_t unmatchable_run_{0U};
  std::uint32_t excess_vdv_stop_{0U};
  std::uint32_t matched_stop_{0U};
  std::uint32_t propagated_delay_{0U};
};

statistics vdv_update(timetable const&,
                      rt_timetable&,
                      source_idx_t,
                      pugi::xml_document const&);

}  // namespace nigiri::rt::vdv