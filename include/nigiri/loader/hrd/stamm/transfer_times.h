#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/category.h"
#include "nigiri/common/interval.h"
#include "nigiri/transfer_rules.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::hrd {

struct stamm;
struct service;

// === transfer time rules (UMSTEIGB / UMSTEIGV / UMSTEIGL / UMSTEIGZ) ======
//
// HAFAS precedence (HRDF 5.40.41, ch. 8):
//   1. trip pair @ station (umsteigz)
//   2. line pair @ station (umsteigl), ascending number of wildcards
//   3. admin pair @ station (umsteigv)
//   4. station transfer time (umsteigb)
//   5. line pair global (umsteigl), ascending number of wildcards
//   6. admin pair global (umsteigv)
//   7. default transfer time (umsteigb, first line)
//
// The rules become transfer rules of the timetable: trip pairs with the trips
// matching their sides (trip_rules_), line pairs with the route ids (one per
// admin, category, line and direction) matching their sides (route_rules_),
// admin pairs with the route ids of the two admins at the station. A global
// rule is written once per station without an own transfer time where both of
// its sides occur.

struct line_rule_side {
  provider_idx_t admin_{provider_idx_t::invalid()};  // invalid = '*'
  bool cat_any_{true};
  category const* cat_{nullptr};
  bool line_any_{true};
  std::array<char, 8U> line_{};
  char dir_{'*'};  // '*' = any, else normalized flag (H -> 1, R -> 2)
};

struct line_rule {
  line_rule_side from_, to_;
  u8_minutes time_{};
  bool guaranteed_{false};
  std::uint8_t n_wildcards_{0U};
  transfer_rule_idx_t rule_{transfer_rule_idx_t::invalid()};
};

struct admin_rule {
  provider_idx_t from_{provider_idx_t::invalid()};
  provider_idx_t to_{provider_idx_t::invalid()};
  u8_minutes time_{};
  transfer_rule_idx_t rule_{transfer_rule_idx_t::invalid()};
};

struct trip_rule {
  std::uint32_t from_nr_{0U};
  provider_idx_t from_admin_{provider_idx_t::invalid()};
  std::uint32_t to_nr_{0U};
  provider_idx_t to_admin_{provider_idx_t::invalid()};
  u8_minutes time_{};
  bool guaranteed_{false};
  unsigned bitfield_num_{0U};  // 0 = all days
  transfer_rule_idx_t rule_{transfer_rule_idx_t::invalid()};
};

// per-side attributes of a stop event used to match transfer rules
struct event_attrs {
  CISTA_COMPARABLE()
  bool valid_{false};  // false = no arrival/departure at this stop
  std::uint32_t train_nr_{0U};
  provider_idx_t admin_{provider_idx_t::invalid()};
  category const* cat_{nullptr};
  std::array<char, 8U> line_{};
  char dir_{' '};  // ' ' = unknown
};

struct stop_attrs {
  CISTA_COMPARABLE()
  event_attrs arr_, dep_;
};

// attributes at stop index `stop_idx` (absolute index into s.stops_)
stop_attrs get_stop_attrs(service const&, std::size_t stop_idx);

struct transfer_times {
  // any pair rules present? (station transfer times alone need no rules)
  bool has_pair_rules() const {
    return !admin_.empty() || !admin_global_.empty() || !line_.empty() ||
           !line_global_.empty() || !trip_.empty();
  }

  std::optional<u8_minutes> default_;  // umsteigb first line, non-IC column
  hash_map<eva_number, u8_minutes> station_;  // umsteigb
  hash_map<eva_number, std::vector<admin_rule>> admin_;  // umsteigv @ station
  std::vector<admin_rule> admin_global_;  // umsteigv global
  hash_map<eva_number, std::vector<line_rule>> line_;  // umsteigl @ station
  std::vector<line_rule> line_global_;  // umsteigl global
  hash_map<eva_number, std::vector<trip_rule>> trip_;  // umsteigz

  // the global rules by the admin of a side (line rules with admin '*' apart)
  hash_map<provider_idx_t, std::vector<std::size_t>> admin_global_from_;
  hash_set<provider_idx_t> admin_global_to_;
  hash_map<provider_idx_t, std::vector<std::size_t>> line_global_from_,
      line_global_to_;
  std::vector<std::size_t> line_global_from_any_, line_global_to_any_;

  // the route ids: one per admin, category, line and direction
  hash_map<std::string, route_id_idx_t> route_ids_;
  vector_map<route_id_idx_t, event_attrs> route_attrs_;

  // collected while the services are written
  transfer_rule_idx_t first_rule_{transfer_rule_idx_t::invalid()};
  hash_set<pair<location_idx_t, provider_idx_t>> arr_admins_, dep_admins_;
  hash_set<pair<location_idx_t, route_id_idx_t>> arr_routes_, dep_routes_;
  qualified_sides_t<trip_idx_t> trip_sides_;
  qualified_sides_t<pair<source_idx_t, route_id_idx_t>> route_sides_;
  std::size_t n_station_rules_{0U};
};

// order: station (umsteigb), admin (umsteigv), line (umsteigl),
//        trip (umsteigz); missing files are empty
std::vector<file> load_transfer_time_files(config const&, dir const&);

// parses only the station transfer times (umsteigb) - independent of
// providers/categories, needed before station registration
void parse_station_transfer_times(transfer_times&, std::string_view);

// parses the pair rules (umsteigv/umsteigl/umsteigz) and writes the rules at
// a station into the timetable; requires stations, providers and categories
void parse_transfer_time_rules(transfer_times&,
                               stamm&,
                               timetable&,
                               source_idx_t,
                               std::string_view admin_file_content,
                               std::string_view line_file_content,
                               std::string_view trip_file_content);

// the route id of a service: its admin, category, line and direction at its
// first stop, registered on first use
route_id_idx_t get_or_create_route_id(transfer_times&,
                                      timetable&,
                                      source_idx_t,
                                      service const&,
                                      std::size_t first_stop);

// the sides of a trip: the trip pair rules at its stops that match it; the
// admins and route ids arriving and departing at each station
void add_trip_sides(transfer_times&,
                    stamm const&,
                    trip_idx_t,
                    route_id_idx_t,
                    service const&,
                    interval<std::size_t> stop_range);

// the line pair rules as route sides, the global rules per station where both
// sides occur; the lookups
void write_transfer_rules(transfer_times&,
                          stamm const&,
                          timetable&,
                          source_idx_t);

}  // namespace nigiri::loader::hrd
