#pragma once

#include <ostream>

#include "nigiri/timetable.h"

namespace nigiri::routing {

struct pt {
  friend std::ostream& operator<<(std::ostream& out, pt const& pt) {
    out << "(src=" << pt.tt_.dbg(pt.t_.t_idx_)
        << ", name=" << pt.tt_.transport_name(pt.t_.t_idx_) << ")";
    return out;
  }
  timetable const& tt_;
  transport t_;
};

}  // namespace nigiri::routing