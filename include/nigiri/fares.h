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
using area_idx_t = cista::strong<std::uint32_t, struct _area_idx_t>;
using network_idx_t = cista::strong<std::uint32_t, struct _network_idx_t>;
using timeframe_group_idx_t =
    cista::strong<std::uint32_t, struct _timeframe_group_idx_t>;
using leg_group_idx_t = cista::strong<std::uint32_t, struct _leg_group_idx_t>;

struct area {
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
    string_idx_t name_;
    fare_media_type type_;
  };

  struct fare_product {
    float amount_;
    string_idx_t name_;
    fare_media_idx_t media_;
    string_idx_t currency_code_;
  };

  struct fare_leg_rule {
    auto match_members() const {
      return std::tie(network_id_, from_area_id_, to_area_id_,
                      from_timeframe_group_id_, to_timeframe_group_id_);
    }

    friend bool operator==(fare_leg_rule const& a, fare_leg_rule const& b) {
      return a.match_members() == b.match_members();
    }

    bool fuzzy_matches(fare_leg_rule const& x) const {
      return (network_id_ == network_idx_t ::invalid() ||
              network_id_ == x.network_id_) &&
             (from_area_id_ == area_idx_t::invalid() ||
              from_area_id_ == x.from_area_id_) &&
             (to_area_id_ == area_idx_t ::invalid() ||
              to_area_id_ == x.to_area_id_) &&
             (from_timeframe_group_id_ == timeframe_group_idx_t::invalid() ||
              from_timeframe_group_id_ == x.from_timeframe_group_id_) &&
             (to_timeframe_group_id_ == timeframe_group_idx_t::invalid() ||
              to_timeframe_group_id_ == x.to_timeframe_group_id_);
    }

    friend std::ostream& operator<<(std::ostream& out, fare_leg_rule const& r) {
      return out << "FROM_AREA=" << r.from_area_id_
                 << ", TO_AREA=" << r.to_area_id_
                 << ", NETWORK=" << r.network_id_
                 << ", FROM_TIMEFRAME_GROUP=" << r.from_timeframe_group_id_
                 << ", TO_TIMEFRAME_GROUP=" << r.to_timeframe_group_id_;
    }

    unsigned rule_priority_{0U};
    network_idx_t network_id_;
    area_idx_t from_area_id_;
    area_idx_t to_area_id_;
    timeframe_group_idx_t from_timeframe_group_id_;
    timeframe_group_idx_t to_timeframe_group_id_;
    fare_product_idx_t fare_product_id_{fare_product_idx_t::invalid()};
    leg_group_idx_t leg_group_idx_{leg_group_idx_t::invalid()};
  };

  struct fare_leg_join_rule {
    CISTA_FRIEND_COMPARABLE(fare_leg_join_rule)
    network_idx_t from_network_id_;
    network_idx_t to_network_id_;
    location_idx_t from_stop_id_{location_idx_t::invalid()};
    location_idx_t to_stop_id_{location_idx_t::invalid()};
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

  struct network {
    string_idx_t name_;
  };

  vector_map<fare_media_idx_t, fare_media> fare_media_;
  vector_map<fare_product_idx_t, fare_product> fare_products_;
  vector<fare_leg_rule> fare_leg_rules_;
  vector<fare_leg_join_rule> fare_leg_join_rules_;
  vector<fare_transfer_rule> fare_transfer_rules_;
  vecvec<timeframe_group_idx_t, timeframe> timeframes_;
  hash_map<route_id_idx_t, network_idx_t> route_networks_;
  vector_map<network_idx_t, network> networks_;
};

inline std::ostream& operator<<(
    std::ostream& out, fares::fare_transfer_rule::fare_transfer_type const t) {
  using fare_transfer_type = fares::fare_transfer_rule::fare_transfer_type;
  switch (t) {
    case fare_transfer_type::kAPlusAB: return out << "APlusAB";
    case fare_transfer_type::kAPlusABPlusB: return out << "APlusABPlusB";
    case fare_transfer_type::kAB: return out << "AB";
  }
  std::unreachable();
}

struct timetable;

namespace routing {

struct fare_leg {
  source_idx_t src_;
  std::vector<journey::leg> joined_leg_;
  std::vector<fares::fare_leg_rule> rule_;
};

struct fare_transfer {
  std::optional<fares::fare_transfer_rule> rule_;
  std::vector<fare_leg> legs_;
};

std::vector<fare_transfer> compute_price(timetable const&, journey const&);

}  // namespace routing

}  // namespace nigiri