#pragma once

#include "nigiri/loader/virt/signatures.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

struct route_stop_pairs;

struct transport_routes {
  route_idx_t get_route(timetable const&,
                        route_idx_t,
                        route_transport_idx_t) const;

  vector_map<route_idx_t, route_idx_t> base_route_;
  vector_map<transport_idx_t, route_idx_t> route_;
};

transport_routes split_routes(timetable const&, route_stop_pairs&);

}  // namespace nigiri::loader
