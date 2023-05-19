#pragma once

#include "fmt/core.h"

#define NIGIRI_TRACING
#if defined(NIGIRI_TRACING)

// #define NIGIRI_RAPTOR_TRACING_ONLY_UPDATES
// #define NIGIRI_RAPTOR_INTERVAL_TRACING

#define trace_upd(...) fmt::print(__VA_ARGS__)

#ifdef NIGIRI_RAPTOR_TRACING_ONLY_UPDATES
#define trace(...)
#else
#define trace(...) fmt::print(__VA_ARGS__)
#endif

#define trace_print_state(...)         \
  fmt::print(__VA_ARGS__);             \
  state_.print(tt_, base(), kInvalid); \
  fmt::print("\n")

#define trace_print_state_after_round() \
  trace_print_state("STATE AFTER ROUND {}\n", k)

#define trace_print_init_state(...) trace_print_state("INIT\n")

#else
#define trace_print_state_after_round()
#define trace_print_init_state()
#define trace_upd(...)
#define trace(...)
#endif