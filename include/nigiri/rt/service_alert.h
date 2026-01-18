#pragma once

#include "nigiri/string_store.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

using alert_str_idx_t = cista::strong<std::uint32_t, struct _alert_string_idx>;
using alert_idx_t = cista::strong<std::uint32_t, struct _alert_idx>;

enum class alert_cause : std::uint8_t {
  kUnknownCause,
  kOtherCause,
  kTechnicalProblem,
  kStrike,
  kDemonstration,
  kAccident,
  kHoliday,
  kWeather,
  kMaintenance,
  kConstruction,
  kPoliceActivity,
  kMedicalEmergency
};

std::string_view to_str(alert_cause);
std::ostream& operator<<(std::ostream&, alert_cause);

enum class alert_effect : std::uint8_t {
  kNoService,
  kReducedService,
  kSignificantDelays,
  kDetour,
  kAdditionalService,
  kModifiedService,
  kOtherEffect,
  kUnknownEffect,
  kStopMoved,
  kNoEffect,
  kAccessibilityIssue
};

std::string_view to_str(alert_effect);
std::ostream& operator<<(std::ostream&, alert_effect);

enum class alert_severity : std::uint8_t {
  kUnknownSeverity,
  kInfo,
  kWarning,
  kSevere
};

std::string_view to_str(alert_severity);
std::ostream& operator<<(std::ostream&, alert_severity);

struct alert_translation {
  alert_str_idx_t text_;
  alert_str_idx_t language_;
};

struct localized_image {
  alert_str_idx_t url_;
  alert_str_idx_t media_type_;
  alert_str_idx_t language_;
};

struct alerts {
  struct by_route_id {
    direction_id_t direction_;
    location_idx_t l_;
    alert_idx_t alert_;
  };
  struct by_agency {
    route_type_t route_type_;
    location_idx_t l_;
    alert_idx_t alert_;
  };
  struct by_rt_transport {
    location_idx_t l_;
    alert_idx_t alert_;
  };
  using by_route = by_rt_transport;
  using by_route_type = by_rt_transport;

  // fuzzy_stop parameter:
  //   - true: alert.l_=invalid matches everything
  //     => used for stop times
  //   - false: alert.l_=invalid matches iff l=invalid
  //     => used for itineraries
  //     - leg (overall trip):
  //         l == invalid => matches only not stop specific alerts
  //         (addressing route/trip/agency)
  //     - from/to/intermediateStop:
  //         l != invalid => matches only concrete stop
  hash_set<alert_idx_t> get_alerts(timetable const& tt,
                                   source_idx_t const src,
                                   trip_idx_t const t,
                                   rt_transport_idx_t const rt_t,
                                   location_idx_t const l,
                                   bool const fuzzy_stop) const {
    auto const route_id_idx = tt.trip_route_id_[t];
    auto const route_type = tt.route_ids_[src].route_id_type_[route_id_idx];
    auto const agency = tt.route_ids_[src].route_id_provider_[route_id_idx];
    auto const direction = tt.trip_direction_id_.test(t);
    auto const parent =
        l == location_idx_t::invalid() ? l : tt.locations_.parents_[l];
    auto const grandparent = parent == location_idx_t::invalid()
                                 ? location_idx_t::invalid()
                                 : tt.locations_.parents_[parent];
    auto const matches_location = [&](location_idx_t const x) {
      return (fuzzy_stop ? (x == location_idx_t::invalid() || x == l)
                         : (x == l)) ||
             (parent != location_idx_t::invalid() && parent == x) ||
             (grandparent != location_idx_t::invalid() && grandparent == x);
    };

    auto alerts = hash_set<alert_idx_t>{};

    if (rt_t != rt_transport_idx_t::invalid()) {
      for (auto const& a : rt_transport_[rt_t]) {
        if (matches_location(a.l_)) {
          alerts.insert(a.alert_);
        }
      }
    }

    for (auto const& a : route_id_[src][route_id_idx]) {
      if ((a.direction_ == direction_id_t::invalid() ||
           a.direction_ == direction) &&
          matches_location(a.l_)) {
        alerts.insert(a.alert_);
      }
    }

    for (auto const& a : agency_[agency]) {
      if ((a.route_type_ == route_type_t::invalid() ||
           a.route_type_ == route_type) &&
          matches_location(a.l_)) {
        alerts.insert(a.alert_);
      }
    }

    if (l != location_idx_t::invalid()) {
      for (auto const& a : location_[l]) {
        alerts.insert(a);
      }
      if (parent != location_idx_t::invalid()) {
        for (auto const& a : location_[parent]) {
          alerts.insert(a);
        }
      }
    }

    return alerts;
  }

  paged_vecvec<rt_transport_idx_t, by_rt_transport> rt_transport_;
  vector_map<source_idx_t, paged_vecvec<route_id_idx_t, by_route_id>> route_id_;
  paged_vecvec<provider_idx_t, by_agency> agency_;
  vector_map<source_idx_t, paged_vecvec<route_type_t, by_route_type>>
      route_type_;
  paged_vecvec<location_idx_t, alert_idx_t> location_;

  vecvec<alert_idx_t, interval<unixtime_t>> communication_period_;
  vecvec<alert_idx_t, interval<unixtime_t>> impact_period_;
  vecvec<alert_idx_t, alert_translation> cause_detail_;
  vecvec<alert_idx_t, alert_translation> effect_detail_;
  vecvec<alert_idx_t, alert_translation> url_;
  vecvec<alert_idx_t, alert_translation> header_text_;
  vecvec<alert_idx_t, alert_translation> description_text_;
  vecvec<alert_idx_t, alert_translation> tts_header_text_;
  vecvec<alert_idx_t, alert_translation> tts_description_text_;
  vecvec<alert_idx_t, alert_translation> image_alternative_text_;
  vecvec<alert_idx_t, localized_image> image_;
  vector_map<alert_idx_t, alert_cause> cause_;
  vector_map<alert_idx_t, alert_effect> effect_;
  vector_map<alert_idx_t, alert_severity> severity_level_;
  string_store<alert_str_idx_t> strings_;
};

}  // namespace nigiri
