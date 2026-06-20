#include "nigiri/routing/tb/tb_data.h"

#include "nigiri/common/day_list.h"
#include "nigiri/routing/tb/segment_info.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

void tb_data::print(std::ostream& out, timetable const& tt) const {
  for (auto t = transport_idx_t{0U}; t != tt.next_transport_idx(); ++t) {
    out << "transport " << tt.transport_name(t) << " @ "
        << tt.days(tt.bitfields_[tt.transport_traffic_days_[t]]) << ":\n";
    for (auto const s : get_segment_range(t)) {
      out << "  " << segment_info{tt, *this, s}
          << ", #transfers=" << segment_transfers_[s].size() << "\n";
      for (auto const transfer : segment_transfers_[s]) {
        out << "    -> " << segment_info{tt, *this, transfer.to_segment_}
            << "\n";
      }
    }
  }
}

}  // namespace nigiri::routing::tb