#include "nigiri/routing/ontrip_train.h"

#include "utl/verify.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void generate_ontrip_train_query(timetable const& tt,
                                 transport const& t,
                                 stop_idx_t const stop_idx,
                                 query& q) {
  utl::verify(stop_idx != 0U, "first arrival time not defined");

  auto const route = tt.transport_route_.at(t.t_idx_);
  auto const location_seq = tt.route_location_seq_.at(route);
  utl::verify(stop_idx < location_seq.size(),
              "invalid stop index {} [{} stops]", stop_idx,
              location_seq.size());
  auto const time_at_first = tt.check_event_time(t, stop_idx, event_type::kArr);
  for (auto i = stop_idx; i != location_seq.size(); ++i) {
    auto const l_idx = stop{location_seq[i]}.location_idx();
    auto const arrival_time_with_transfer =
        tt.check_event_time(t, i, event_type::kArr) +
        tt.locations_.transfer_time_[l_idx];
    q.start_.emplace_back(l_idx, arrival_time_with_transfer - time_at_first,
                          static_cast<std::uint8_t>(i));
  }
  q.start_time_ = time_at_first;
}

}  // namespace nigiri::routing
