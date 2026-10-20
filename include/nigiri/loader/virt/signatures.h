#pragma once

#include <cstdint>

#include "nigiri/transfer_rules.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

using signature_idx_t = cista::strong<std::uint32_t, struct signature_idx_>;

constexpr auto const kEmptySig = signature_idx_t{0U};

struct signatures {
  vecvec<signature_idx_t, transfer_rule_side_idx> rules_;

  // route_stop_sigs_[R][I][T] = active rules for route R, stop I, transport T
  //
  // compaction rules:
  // [R].empty()            => no transport of route R has rules
  // [R].size >= 1          => stop_seq[R].size() stop entries
  // [R][I].empty()         => no transport at stop I has rules
  // [R][I].size() == 1     => transports at stop I all have the same rules
  // [R][I].size() = n > 1  => different rules for all N transports of the route
  nvec<route_idx_t, signature_idx_t, 2U> route_stop_sigs_;
};

signatures get_signatures(timetable const&);

}  // namespace nigiri::loader
