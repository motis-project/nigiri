#include "nigiri/common/day_list.h"

namespace nigiri {

day_list::day_list(bitfield const& bf, date::sys_days base)
    : bf_{bf}, base_{base} {}

std::ostream& operator<<(std::ostream& out, day_list const& l) {
  out << "{";
  auto first = true;
  for (auto i = 0U; i != kMaxDays; ++i) {
    if (l.bf_.test(i)) {
      if (!first) {
        out << ", ";
      }
      date::to_stream(out, "%F", l.base_ + i * date::days{1});
      first = false;
    }
  }
  out << "}";
  return out;
}

}  // namespace nigiri