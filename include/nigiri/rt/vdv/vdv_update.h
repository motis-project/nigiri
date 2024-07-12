#pragma once

#include "pugixml.hpp"

#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt::vdv {

struct statistics {
  friend std::ostream& operator<<(std::ostream& out, statistics const& s);

  int unsupported_additional_{0};
  int unsupported_cancelled_{0};
};

statistics vdv_update(timetable const&,
                      rt_timetable&,
                      source_idx_t,
                      pugi::xml_document const&);

}  // namespace nigiri::rt::vdv