#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/service/progress_update_fn.h"
#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/common/interval.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

struct service_builder {
  explicit service_builder(stamm&, timetable&);

  void add_services(config const& c,
                    char const* filename,
                    std::string_view file_content,
                    progress_update_fn const&);

  void write_services(source_idx_t);

  void write_location_routes();

private:
  void add_service(ref_service&&);

  stamm& stamm_;
  timetable& tt_;
  hash_map<pair<std::basic_string<stop::value_type>, std::basic_string<clasz>>,
           vector<vector<ref_service>>>
      route_services_;
  service_store store_;
  hash_map<std::basic_string<attribute_idx_t>, attribute_combination_idx_t>
      attribute_combinations_;
  hash_map<bitfield, bitfield_idx_t> bitfield_indices_;
  interval<std::chrono::sys_days> selection_;

  mutable_fws_multimap<location_idx_t, route_idx_t> location_routes_;

  // Reused memory buffers to prevent temporary allocations:
  pair<std::basic_string<stop::value_type>, std::basic_string<clasz>>
      route_key_;
  std::basic_string<attribute_idx_t> attribute_combination_;
  std::basic_string<provider_idx_t> section_providers_;
  std::basic_string<attribute_combination_idx_t> section_attributes_;
  std::basic_string<trip_direction_idx_t> section_directions_;
  std::basic_string<trip_line_idx_t> section_lines_;
  std::basic_string<stop_idx_t> stop_seq_numbers_;
  fmt::memory_buffer trip_id_buf_;
};

}  // namespace nigiri::loader::hrd
