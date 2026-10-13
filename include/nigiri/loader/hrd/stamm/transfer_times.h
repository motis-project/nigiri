#pragma once

#include <array>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

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

// HRDF transfer time rules (UMSTEIGB / UMSTEIGV / UMSTEIGL / UMSTEIGZ) as
// transfer rules: one rule per row at its station (global rows once per
// station where they match), its sides qualified by the trips whose stop
// events match. The virtual locations are decided in finalize like for
// transfers.txt.
//
// HAFAS precedence (HRDF 5.40.41, ch. 8) = rule specificity:
//   1. trip pair @ station (umsteigz)
//   2. line pair @ station (umsteigl), ascending number of wildcards
//   3. admin pair @ station (umsteigv)
//   4. station transfer time (umsteigb) - shadows 5. and 6. at the station
//   5. line pair global (umsteigl), ascending number of wildcards
//   6. admin pair global (umsteigv)
//   7. default transfer time (umsteigb, first line)

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
  std::uint8_t n_wildcards_{0U};
};

struct admin_rule {
  provider_idx_t from_{provider_idx_t::invalid()};
  provider_idx_t to_{provider_idx_t::invalid()};
  u8_minutes time_{};
};

struct trip_rule {
  std::uint32_t from_nr_{0U};
  provider_idx_t from_admin_{provider_idx_t::invalid()};
  std::uint32_t to_nr_{0U};
  provider_idx_t to_admin_{provider_idx_t::invalid()};
  u8_minutes time_{};
};

struct transfer_times {
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
};

// order: station (umsteigb), admin (umsteigv), line (umsteigl),
//        trip (umsteigz); missing files are empty
std::vector<file> load_transfer_time_files(config const&, dir const&);

void parse_station_transfer_times(transfer_times&, std::string_view);

// requires providers and categories
void parse_transfer_time_rules(transfer_times&,
                               stamm&,
                               std::string_view admin_file_content,
                               std::string_view line_file_content,
                               std::string_view trip_file_content);

// per-side attributes of a stop event used to match transfer rules
struct event_attrs {
  std::uint32_t train_nr_{0U};
  provider_idx_t admin_{provider_idx_t::invalid()};
  category const* cat_{nullptr};
  std::array<char, 8U> line_{};
  char dir_{' '};  // ' ' = unknown
};

struct transfer_rule_builder {
  enum class tier : std::uint8_t {
    kTrip,
    kLineStation,
    kAdminStation,
    kLineGlobal,
    kAdminGlobal
  };

  struct instance {
    tier tier_;
    std::uint32_t row_;
    location_idx_t station_;  // invalid = global
    u8_minutes time_;
    transfer_rule_specificity_t specificity_;
    std::array<bool, 2U> has_side_{};
  };

  transfer_rule_builder(timetable&,
                        transfer_times const&,
                        stamm const&,
                        source_idx_t);

  // The trip's attribute group (admin, category, line, direction) as a route
  // id, carrying the sides of the global rows it matches. Invalid if the
  // attributes change along the trip (no global rules then).
  route_id_idx_t get_group(service const&, interval<std::size_t> stop_range);

  // the station rule sides the stop events of trip t match
  void add_trip(trip_idx_t, service const&, interval<std::size_t> stop_range);

  // appends the rules with trips on both sides to the timetable
  void write();

  timetable& tt_;
  transfer_times const& r_;
  stamm const& st_;
  source_idx_t src_;
  // global rows by the admin of their side ('*' = any)
  using global_row = std::pair<tier, std::uint32_t>;
  hash_map<provider_idx_t, std::vector<global_row>> global_from_, global_to_;
  std::vector<global_row> global_from_any_, global_to_any_;
  hash_map<std::tuple<tier, std::uint32_t, location_idx_t>, std::uint32_t>
      instance_idx_;
  std::vector<instance> instances_;
  std::vector<std::tuple<trip_idx_t, std::uint32_t, event_type>> trip_sides_;

  using group_key =
      std::tuple<provider_idx_t, category const*, std::array<char, 8U>, char>;
  hash_map<group_key, route_id_idx_t> groups_;
  std::vector<std::tuple<route_id_idx_t, std::uint32_t, event_type>>
      group_sides_;
  std::size_t n_varying_{0U};  // trips whose attributes change
};

}  // namespace nigiri::loader::hrd
