#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/category.h"
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
  bool guaranteed_{false};
  unsigned bitfield_num_{0U};  // 0 = all days
};

struct transfer_times {
  // any pair rules present? (station transfer times alone need no groups)
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

  // lookup helpers for the event scan
  hash_set<eva_number> rule_stations_;  // stations with any pair rule
  hash_set<provider_idx_t> global_admins_;  // admins in global rule sides
  hash_set<std::string> global_admin_strings_;
  bool global_matches_any_admin_{false};  // global rule side with admin '*'
};

// order: station (umsteigb), admin (umsteigv), line (umsteigl),
//        trip (umsteigz); missing files are empty
std::vector<file> load_transfer_time_files(config const&, dir const&);

// parses only the station transfer times (umsteigb) - independent of
// providers/categories, needed before station registration
void parse_station_transfer_times(transfer_times&, std::string_view);

// parses the pair rules (umsteigv/umsteigl/umsteigz);
// requires providers/categories to be parsed already
void parse_transfer_time_rules(transfer_times&,
                               stamm&,
                               std::string_view admin_file_content,
                               std::string_view line_file_content,
                               std::string_view trip_file_content);

// === transfer groups ======================================================
//
// Trips at a station are partitioned into groups by the set of rule sides
// matching them. Each group that behaves differently from the default gets a
// virtual child location (location_type::kGeneratedTransfer). The pairwise
// transfer times between the station and its groups are emitted as directed
// footpaths into timetable::locations::transfer_rule_fps_.

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

struct transfer_groups {
  struct assignment {
    location_idx_t static_group_{location_idx_t::invalid()};
    std::vector<location_idx_t> by_day_;  // non-empty overrides static_group_
  };

  struct station {
    eva_number eva_{0U};
    std::set<stop_attrs> tuples_;  // phase 1, cleared by build()
    std::map<stop_attrs, assignment> assignment_;  // phase 2 lookup
  };

  location_idx_t resolve(location_idx_t const base,
                         stop_attrs const& attrs,
                         std::size_t const day) const {
    auto const station_it = stations_.find(base);
    if (station_it == end(stations_)) {
      return base;
    }
    auto const it = station_it->second.assignment_.find(attrs);
    if (it == end(station_it->second.assignment_)) {
      return base;
    }
    auto const& a = it->second;
    return !a.by_day_.empty() && day < a.by_day_.size() ? a.by_day_[day]
                                                        : a.static_group_;
  }

  bool active_{false};
  hash_map<location_idx_t, station> stations_;
};

// phase 1: collect stop event attributes relevant for transfer rules
void scan_transfer_events(config const&,
                          stamm&,
                          transfer_times const&,
                          transfer_groups&,
                          std::string_view fplan_file_content);

// between phase 1 and 2: partition events into groups, create virtual
// locations and emit the pairwise transfer time matrix
void build_transfer_groups(stamm&,
                           timetable&,
                           transfer_times const&,
                           transfer_groups&);

}  // namespace nigiri::loader::hrd
