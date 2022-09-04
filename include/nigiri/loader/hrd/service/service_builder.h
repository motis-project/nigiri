#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/service/read_services.h"
#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/timetable.h"
#include "utl/is_uniform.h"

namespace nigiri::loader::hrd {

struct service_builder {

  explicit service_builder(stamm& s, timetable& tt) : stamm_{s}, tt_{tt} {}

  void add_services(config const& c,
                    char const* filename,
                    std::string_view file_content,
                    progress_update_fn const&);

  void write_services(source_idx_t const src);

  stamm& stamm_;
  timetable& tt_;
  hash_map<pair<std::basic_string<timetable::stop::value_type>,
                std::basic_string<clasz>>,
           vector<vector<ref_service>>>
      route_services_;
  service_store store_;
  hash_map<vector<attribute_idx_t>, attribute_combination_idx_t>
      attribute_combinations_;
  hash_map<bitfield, bitfield_idx_t> bitfield_indices_;
};

}  // namespace nigiri::loader::hrd
