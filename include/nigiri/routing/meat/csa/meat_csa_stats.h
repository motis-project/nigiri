#pragma once

#include <cstdint>

namespace nigiri::routing::meat::csa {

struct meat_csa_stats {
  void reset() {
    esa_n_connections_scanned_ = 0ULL;
    esa_n_update_arr_time_ = 0ULL;
    ea_n_connections_scanned_ = 0ULL;
    ea_n_update_arr_time_ = 0ULL;
    meat_n_connections_scanned_ = 0ULL;
    meat_n_fp_added_to_que_ = 0ULL;
    meat_n_fp_added_to_profile_ = 0ULL;
    meat_n_e_added_or_replaced_to_profile_ = 0ULL;
    meat_n_e_in_profile_ = 0ULL;
    esa_duration_ = 0ULL;
    ea_duration_ = 0ULL;
    meat_duration_ = 0ULL;
    extract_graph_duration_ = 0ULL;
    total_duration_ = 0ULL;
  }

  std::uint64_t esa_n_connections_scanned_{0ULL};
  std::uint64_t esa_n_update_arr_time_{0ULL};
  std::uint64_t ea_n_connections_scanned_{0ULL};
  std::uint64_t ea_n_update_arr_time_{0ULL};
  std::uint64_t meat_n_connections_scanned_{0ULL};
  std::uint64_t meat_n_fp_added_to_que_{0ULL};
  std::uint64_t meat_n_fp_added_to_profile_{0ULL};
  std::uint64_t meat_n_e_added_or_replaced_to_profile_{0ULL};
  std::uint64_t meat_n_e_in_profile_{0ULL};
  std::uint64_t esa_duration_{0ULL};
  std::uint64_t ea_duration_{0ULL};
  std::uint64_t meat_duration_{0ULL};
  std::uint64_t extract_graph_duration_{0ULL};
  std::uint64_t total_duration_{0ULL};
};

}  // namespace nigiri::routing::meat::csa
