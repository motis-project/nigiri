#pragma once

#include <cstdint>

#include "nigiri/routing/raptor/raptor.h"

namespace nigiri::routing::meat::raptor {

struct meat_raptor_stats {
  void reset() {
    esa_stats_ = routing_result<raptor_stats> {};
    ea_stats_ = routing_result<raptor_stats> {};
    meat_n_active_transports_iterated_ = 0ULL;
    meat_n_stops_iterated_ = 0ULL;
    meat_n_fp_added_to_profile_ = 0ULL;
    meat_n_e_added_to_profile_ = 0ULL;
    meat_n_e_in_profile_ = 0ULL;
    esa_duration_ = 0ULL;
    ea_duration_ = 0ULL;
    meat_duration_ = 0ULL;
    extract_graph_duration_ = 0ULL;
    total_duration_ = 0ULL;
  }

  routing_result<raptor_stats> esa_stats_;
  routing_result<raptor_stats> ea_stats_;
  std::uint64_t meat_n_active_transports_iterated_{0ULL};
  std::uint64_t meat_n_stops_iterated_{0ULL};
  std::uint64_t meat_n_fp_added_to_profile_{0ULL};
  std::uint64_t meat_n_e_added_to_profile_{0ULL};
  std::uint64_t meat_n_e_in_profile_{0ULL};
  std::uint64_t esa_duration_{0ULL};
  std::uint64_t ea_duration_{0ULL};
  std::uint64_t meat_duration_{0ULL};
  std::uint64_t extract_graph_duration_{0ULL};
  std::uint64_t total_duration_{0ULL};
};

}  // namespace nigiri::routing::meat::raptor
