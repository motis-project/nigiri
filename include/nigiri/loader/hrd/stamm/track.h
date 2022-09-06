#pragma once

#include <string>

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct stamm;

struct track_rule_key {
  CISTA_COMPARABLE()
  static constexpr minutes_after_midnight_t kTimeNotSet = -1_minutes;
  location_idx_t location_{0U};
  unsigned train_num_{0U};
  provider_idx_t admin_{provider_idx_t::invalid()};
  minutes_after_midnight_t time_{-1_minutes};
};

struct track_rule {
  unsigned bitfield_num_{0};
  location_idx_t track_location_;
};

struct track_at_station {
  CISTA_COMPARABLE()
  location_idx_t parent_station_;
  track_name_idx_t track_;
};

using track_rule_map_t = hash_map<track_rule_key, std::vector<track_rule>>;
using track_location_map_t = hash_map<track_at_station, location_idx_t>;

void parse_track_rules(config const&,
                       stamm&,
                       timetable&,
                       std::string_view file_content,
                       track_rule_map_t&,
                       track_location_map_t&);

}  // namespace nigiri::loader::hrd