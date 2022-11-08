#include "nigiri/routing/journey.h"

#include "utl/enumerate.h"
#include "utl/overloaded.h"

#include "nigiri/common/indent.h"
#include "nigiri/print_transport.h"

namespace nigiri::routing {

void journey::leg::print(std::ostream& out,
                         nigiri::timetable const& tt,
                         unsigned const n_indent,
                         bool const debug) const {
  std::visit(utl::overloaded{[&](transport_enter_exit const& t) {
                               print_transport(tt, out, t.t_, t.stop_range_,
                                               n_indent, debug);
                             },
                             [&](footpath_idx_t const) {
                               indent(out, n_indent);
                               out << "FOOTPATH\n";
                             },
                             [&](std::uint8_t const x) {
                               indent(out, n_indent);
                               out << "MUMO ID " << static_cast<int>(x) << "\n";
                             }},
             uses_);
}

void journey::print(std::ostream& out,
                    nigiri::timetable const& tt,
                    bool const debug) const {
  if (legs_.empty()) {
    out << "no legs\n";
    return;
  }

  out << "[" << start_time_ << ", " << dest_time_ << "]\n";
  out << "TRANSFERS: " << static_cast<int>(transfers_) << "\n";
  out << "     FROM: " << location{tt, legs_.front().from_} << " ["
      << legs_.front().dep_time_ << "]\n";
  out << "       TO: " << location{tt, legs_.back().to_} << " ["
      << legs_.back().arr_time_ << "]\n";
  for (auto const [i, l] : utl::enumerate(legs_)) {
    out << "leg " << i << ": " << location{tt, l.from_} << " [" << l.dep_time_
        << "] -> " << location{tt, l.to_} << " [" << l.arr_time_ << "]\n";
    l.print(out, tt, 1, debug);
  }
}

}  // namespace nigiri::routing
