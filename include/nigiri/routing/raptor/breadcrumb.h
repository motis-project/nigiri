#pragma once

#include "nigiri/routing/gpu/breadcrumb.h"

namespace nigiri::routing {

// The CPU mcraptor packs its transit breadcrumbs exactly like the GPU engines
// (see gpu/breadcrumb.h). Unlike the single-criterion raptor, whose round times
// hold one cell per (round, stop) so the previous leg is just (k - 1, board),
// mcraptor keeps a pareto set per stop: the predecessor is ambiguous, so its
// breadcrumb also carries an explicit parent arena index (next to the payload,
// not in this word).
using gpu::bc_alight;
using gpu::bc_board;
using gpu::bc_transport;
using gpu::bc_transport_space_fits;
using gpu::breadcrumb_t;
using gpu::decode_rt_bc_transport;
using gpu::encode_rt_bc_transport;
using gpu::is_rt_bc_transport;
using gpu::kBcStopMask;
using gpu::kStartSentinel;
using gpu::make_transport_payload;

}  // namespace nigiri::routing
