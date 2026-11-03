#pragma once

#include "nigiri/routing/gpu/breadcrumb.h"

namespace nigiri::routing {

// The CPU mcraptor uses the GPU breadcrumb encoding; its breadcrumbs also store
// the parent index, as a stop holds several labels per round.
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
