#pragma once

#include "nigiri/types.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/string_store.h"

namespace nigiri::loader::gtfs {

using fare_media_idx_t = cista::strong<std::uint32_t, struct _fare_media_idx>;
using fare_product_idx_t =
    cista::strong<std::uint32_t, struct _fare_product_idx>;
using fare_leg_join_rule_idx_t =
    cista::strong<std::uint32_t, struct _fare_leg_join_rule_idx>;
using fare_transfer_rule_idx_t =
    cista::strong<std::uint32_t, struct _fare_transfer_rule_idx>;
using area_idx_t = cista::strong<std::uint32_t, struct _area_idx_t>;
using network_idx_t = cista::strong<std::uint32_t, struct _network_idx_t>;
using timeframe_group_idx_t =
    cista::strong<std::uint32_t, struct _timeframe_group_idx_t>;
using leg_group_idx_t = cista::strong<std::uint32_t, struct _leg_group_idx_t>;

struct fares {
  struct fare_media {
    enum class fare_media_type : std::uint8_t {
      kNone,
      kPaper,
      kCard,
      kContactless,
      kApp
    };
    string_idx_t name_;
    fare_media_type type_;
  };

  struct fare_product {
    string_idx_t name_;
    fare_media_idx_t media_;
    std::uint32_t amount_;
    string_idx_t currency_code_;
  };

  struct fare_leg_rule {
    leg_group_idx_t leg_group_idx_;
    network_idx_t network_id_;
    area_idx_t from_area_id_;
    area_idx_t to_area_id_;
    timeframe_group_idx_t from_timeframe_group_id_;
    timeframe_group_idx_t to_timeframe_group_id_;
    fare_product_idx_t fare_product_id_;
    unsigned rule_priority_;
  };

  struct fare_leg_join_rule {
    network_idx_t from_network_id_;
    network_idx_t to_network_id_;
    location_idx_t from_stop_id_;
    location_idx_t to_stop_id_;
  };

  struct fare_transfer_rule {
    static constexpr auto const kNoDurationLimit =
        duration_t{std::numeric_limits<duration_t::rep>::max()};

    enum class duration_limit_type : std::uint8_t {
      kCurrDepNextArr,
      kCurrDepNextDep,
      kCurrArrNextDep,
      kCurrArrNextArr
    };

    enum class fare_transfer_type : std::uint8_t {
      kAPlusAB,  // fare_leg_rules.fare_product_id +
                 // fare_transfer_rules.fare_product_id
      kAPlusABPlusB,  // fare_leg_rules.fare_product_id +
                      // fare_transfer_rules.fare_product_id +
                      // fare_leg_rules.fare_product_id
      kAB  // fare_transfer_rules.fare_product_id
    };

    leg_group_idx_t from_leg_group_{leg_group_idx_t::invalid()};
    leg_group_idx_t to_leg_group_{leg_group_idx_t::invalid()};
    std::int8_t transfer_count_{-1};
    duration_t duration_limit_{kNoDurationLimit};
    duration_limit_type duration_limit_type_{
        duration_limit_type::kCurrDepNextArr};
    fare_transfer_type fare_transfer_type_{fare_transfer_type::kAPlusAB};
    fare_product_idx_t fare_product_{fare_product_idx_t::invalid()};
  };

  struct timeframe {
    duration_t start_time_;
    duration_t end_time_;
    bitfield service_;
  };

  struct area {
    string_idx_t name_;
  };

  struct network {
    string_idx_t name_;
  };

  vector_map<fare_media_idx_t, fare_media> fare_media_;
  vector_map<fare_product_idx_t, fare_product> fare_products_;
  vector<fare_leg_rule> fare_leg_rules_;
  vector<fare_leg_join_rule> fare_leg_join_rules_;
  vector<fare_transfer_rule> fare_transfer_rules_;
  vecvec<timeframe_group_idx_t, timeframe> timeframes_;
  vecvec<location_idx_t, area_idx_t> location_areas_;
  vector_map<area_idx_t, area> areas_;
  hash_map<route_id_idx_t, network_idx_t> route_networks_;
  vector_map<network_idx_t, network> networks_;
  string_store strings_;
};

hash_map<std::string, fare_media_idx_t> parse_media(
    std::string_view file_content, string_cache_t&, fares&);

hash_map<std::string, network_idx_t> parse_networks(
    std::string_view file_content, string_cache_t&, fares&);

hash_map<std::string, route_idx_t> parse_route_networks(
    std::string_view file_content, fares&);

hash_map<std::string, location_idx_t> parse_stop_areas(
    std::string_view file_content, fares&);

hash_map<std::string, fare_leg_join_rule_idx_t> parse_fare_leg_join_rules(
    std::string_view file_content, fares&);

void parse_fare_transfer_rules(std::string_view file_content,
                               fares&,
                               hash_map<std::string, fare_product_idx_t> const&,
                               hash_map<std::string, leg_group_idx_t> const&);

void parse_fares(timetable const& tt,
                 dir const&,
                 traffic_days_t const&,
                 route_map_t const&,
                 locations_map const&);

}  // namespace nigiri::loader::gtfs