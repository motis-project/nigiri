#pragma once

#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/types.h"

#define NIGIRI_DUMP_ROUND_TIMES_DIR "./round_times"
#ifdef NIGIRI_DUMP_ROUND_TIMES_DIR
#define dump_round_times(dbg_dir, tag, state, k, Vias) \
  dump_round_times_fun<Vias>(dbg_dir, tag, state, k)

namespace nigiri::routing {

template <via_offset_t Vias>
void dump_round_times_fun(std::optional<std::string> const& dbg_dir,
                          std::string_view tag,
                          raptor_state const& rs,
                          unsigned const k) {
  if (dbg_dir) {
    auto ostrm = std::ofstream{fmt::format("{}/k{}_{}.bin", *dbg_dir, k, tag),
                               std::ios::binary};
    ostrm.write(reinterpret_cast<char const*>(
                    &rs.round_times_storage_[k * rs.n_locations_ * (Vias + 1)]),
                rs.n_locations_ * (Vias + 1) * sizeof(delta_t));
  }
}

}  // namespace nigiri::routing

#else
#define dump_round_times(dbg_dir, tag, state, k, Vias)
#endif
