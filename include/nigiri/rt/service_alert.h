#pragma once

#include "nigiri/string_store.h"
#include "nigiri/types.h"

namespace nigiri {

using service_alert_idx_t =
    cista::strong<std::uint32_t, struct _service_alert_idx>;

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

struct translation {
  string_idx_t text_;
  string_idx_t language_;
};

struct localized_image {
  string_idx_t url_;
  string_idx_t media_type_;
  string_idx_t language_;
};

struct service_alerts {
  paged_vecvec<rt_transport_idx_t, pair<service_alert_idx_t, location_idx_t>>
      rt_service_alerts_;
  vecvec<service_alert_idx_t, interval<unixtime_t>> communication_period_;
  vecvec<service_alert_idx_t, interval<unixtime_t>> impact_period_;
  vecvec<service_alert_idx_t, translation> cause_detail_;
  vecvec<service_alert_idx_t, translation> effect_detail_;
  vecvec<service_alert_idx_t, translation> url_;
  vecvec<service_alert_idx_t, translation> header_text_;
  vecvec<service_alert_idx_t, translation> description_text_;
  vecvec<service_alert_idx_t, translation> tts_header_text_;
  vecvec<service_alert_idx_t, translation> tts_description_text_;
  vecvec<service_alert_idx_t, translation> image_alternative_text_;
  vecvec<service_alert_idx_t, localized_image> image_;
  vector_map<service_alert_idx_t, alert_cause> cause_;
  vector_map<service_alert_idx_t, alert_effect> effect_;
  vector_map<service_alert_idx_t, alert_severity> severity_level_;
  string_store strings_;
};

}  // namespace nigiri