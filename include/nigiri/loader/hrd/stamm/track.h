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
  location_idx_t location_{0U};
  std::uint32_t train_num_{0U};
  provider_idx_t admin_{provider_idx_t::invalid()};
};

struct track_rule {
  static constexpr minutes_after_midnight_t kTimeNotSet = -1_minutes;
  unsigned bitfield_num_{0};
  location_idx_t track_location_;
  minutes_after_midnight_t mam_{-1_minutes};
};

struct track_at_station {
  CISTA_FRIEND_COMPARABLE(track_at_station)
  location_idx_t parent_station_;
  std::string track_name_;
};

struct tracks {
  location_idx_t lookup(track_rule_key const&,
                        minutes_after_midnight_t,
                        day_idx_t) const;
  hash_map<track_rule_key, std::vector<track_rule>> track_rules_;
  hash_map<track_at_station, location_idx_t> rack_locations_;
};

tracks parse_track_rules(config const&,
                         stamm&,
                         timetable&,
                         std::string_view file_content);

}  // namespace nigiri::loader::hrd
