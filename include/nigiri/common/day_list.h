#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct day_list {
  day_list(bitfield const& bf, date::sys_days base);
  friend std::ostream& operator<<(std::ostream& out, day_list const& l);
  bitfield const& bf_;
  date::sys_days base_;
};

}  // namespace nigiri