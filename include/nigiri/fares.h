#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/string_store.h"
#include "nigiri/types.h"

namespace nigiri {

using fare_media_idx_t = cista::strong<std::uint32_t, struct _fare_media_idx>;
using fare_product_idx_t =
    cista::strong<std::uint32_t, struct _fare_product_idx>;
using fare_leg_join_rule_idx_t =
    cista::strong<std::uint32_t, struct _fare_leg_join_rule_idx>;
using rider_category_idx_t =
    cista::strong<std::uint32_t, struct _rider_category_idx>;
using area_idx_t = cista::strong<std::uint32_t, struct _area_idx_t>;
using network_idx_t = cista::strong<std::uint32_t, struct _network_idx_t>;
using timeframe_group_idx_t =
    cista::strong<std::uint32_t, struct _timeframe_group_idx_t>;
using leg_group_idx_t = cista::strong<std::uint32_t, struct _leg_group_idx_t>;
using area_set_idx_t = cista::strong<std::uint32_t, struct _area_set_idx_t>;

struct area {
  string_idx_t id_;
  string_idx_t name_;
};

struct fares {
  struct fare_media {
    enum class fare_media_type : std::uint8_t {
      kNone,
      kPaper,
      kCard,
      kContactless,
      kApp
    };
    friend std::ostream& operator<<(std::ostream&, fare_media_type);
    string_idx_t name_;
    fare_media_type type_;
  };

  struct fare_product {
    float amount_;
    string_idx_t name_;
    fare_media_idx_t media_;
    string_idx_t currency_code_;
    rider_category_idx_t rider_category_{rider_category_idx_t::invalid()};
  };

  struct fare_leg_rule {
    auto match_members() const;
    friend bool operator==(fare_leg_rule const&, fare_leg_rule const&);

    friend std::ostream& operator<<(std::ostream&, fare_leg_rule const&);

    std::int32_t rule_priority_{0};
    network_idx_t network_;
    area_idx_t from_area_;
    area_idx_t to_area_;
    timeframe_group_idx_t from_timeframe_group_;
    timeframe_group_idx_t to_timeframe_group_;
    fare_product_idx_t fare_product_{fare_product_idx_t::invalid()};
    leg_group_idx_t leg_group_idx_{leg_group_idx_t::invalid()};
    area_set_idx_t contains_exactly_area_set_id_{area_set_idx_t::invalid()};
    area_set_idx_t contains_area_set_id_{area_set_idx_t::invalid()};
  };

  struct fare_leg_join_rule {
    CISTA_FRIEND_COMPARABLE(fare_leg_join_rule)
    network_idx_t from_network_;
    network_idx_t to_network_;
    location_idx_t from_stop_{location_idx_t::invalid()};
    location_idx_t to_stop_{location_idx_t::invalid()};
  };

  struct fare_transfer_rule {
    static constexpr auto const kNoDurationLimit =
        i32_minutes{std::numeric_limits<i32_minutes::rep>::max()};

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

    friend std::ostream& operator<<(std::ostream&, fare_transfer_rule const&);
    friend std::ostream& operator<<(std::ostream&, fare_transfer_type);
    friend std::ostream& operator<<(std::ostream&, duration_limit_type);

    leg_group_idx_t from_leg_group_{leg_group_idx_t::invalid()};
    leg_group_idx_t to_leg_group_{leg_group_idx_t::invalid()};
    std::int8_t transfer_count_{-1};
    i32_minutes duration_limit_{kNoDurationLimit};
    duration_limit_type duration_limit_type_{
        duration_limit_type::kCurrDepNextArr};
    fare_transfer_type fare_transfer_type_{fare_transfer_type::kAPlusAB};
    fare_product_idx_t fare_product_{fare_product_idx_t::invalid()};
  };

  struct rider_category {
    string_idx_t name_;
    string_idx_t eligibility_url_;
    bool is_default_fare_category_;
  };

  struct timeframe {
    duration_t start_time_;
    duration_t end_time_;
    bitfield service_;
    string_idx_t service_id_;
  };

  struct network {
    string_idx_t id_;
    string_idx_t name_;
  };

  vector_map<leg_group_idx_t, string_idx_t> leg_group_name_;
  vector_map<fare_media_idx_t, fare_media> fare_media_;
  vecvec<fare_product_idx_t, fare_product> fare_products_;
  vector_map<fare_product_idx_t, string_idx_t> fare_product_id_;
  vector<fare_leg_rule> fare_leg_rules_;
  vector<fare_leg_join_rule> fare_leg_join_rules_;
  vector<fare_transfer_rule> fare_transfer_rules_;
  vector_map<rider_category_idx_t, rider_category> rider_categories_;
  vecvec<timeframe_group_idx_t, timeframe> timeframes_;
  vector_map<timeframe_group_idx_t, string_idx_t> timeframe_id_;
  hash_map<route_id_idx_t, network_idx_t> route_networks_;
  vector_map<network_idx_t, network> networks_;
  vecvec<area_set_idx_t, area_idx_t> area_sets_;
  vector_map<area_set_idx_t, string_idx_t> area_set_ids_;
  bool has_priority_{false};
};

struct timetable;

using effective_fare_leg_t = std::vector<routing::journey::leg const*>;

struct fare_leg {
  source_idx_t src_;
  effective_fare_leg_t joined_leg_;
  std::vector<fares::fare_leg_rule> rule_;
};

struct fare_transfer {
  std::optional<fares::fare_transfer_rule> rule_;
  std::vector<fare_leg> legs_;
};

std::vector<fare_transfer> get_fares(timetable const&,
                                     rt_timetable const*,
                                     routing::journey const&);

}  // namespace nigiri