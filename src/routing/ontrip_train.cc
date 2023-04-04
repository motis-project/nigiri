#include "nigiri/routing/ontrip_train.h"

#include "utl/verify.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

query generate_ontrip_train_query(timetable const& tt,
                                  transport const& t,
                                  unsigned const stop_idx,
                                  query const& q) {
  auto ontrip_train_query = q;
  auto const route = tt.transport_route_.at(t.t_idx_);
  auto const location_seq = tt.route_location_seq_.at(route);
  utl::verify(stop_idx < location_seq.size(),
              "invalid stop index {} [{} stops]", stop_idx,
              location_seq.size());
  for (auto i = stop_idx; i != location_seq.size(); ++i) {
    fmt::print("{}\n", tt.event_time(t, i, event_type::kArr));
  }
  return q;
}

}  // namespace nigiri::routing