#pragma once

// Compile-time toggles for GPU RAPTOR perf experiments. Override via -D on the
// nvcc command line, otherwise the defaults below apply.
//
//   NIGIRI_GPU_COMPUTE_ET : 0 = inline get_earliest_transport (baseline)
//                           1 = parallel compact-task-list pre-pass (#1)
//   NIGIRI_GPU_SPLIT_TIMES: 0 = 64-bit packed round_times reads (baseline)
//                           1 = 16-bit time mirror for the hot reads (#3)

#ifndef NIGIRI_GPU_COMPUTE_ET
#define NIGIRI_GPU_COMPUTE_ET 0
#endif

#ifndef NIGIRI_GPU_SPLIT_TIMES
#define NIGIRI_GPU_SPLIT_TIMES 0
#endif
