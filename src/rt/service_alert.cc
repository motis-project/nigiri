#include "nigiri/rt/service_alert.h"

namespace nigiri {

std::string_view to_str(alert_cause const x) {
  switch (x) {
    case alert_cause::kUnknownCause: return "UNKNOWN_CAUSE";
    case alert_cause::kOtherCause: return "OTHER_CAUSE";
    case alert_cause::kTechnicalProblem: return "TECHNICAL_PROBLEM";
    case alert_cause::kStrike: return "STRIKE";
    case alert_cause::kDemonstration: return "DEMONSTRATION";
    case alert_cause::kAccident: return "ACCIDENT";
    case alert_cause::kHoliday: return "HOLIDAY";
    case alert_cause::kWeather: return "WEATHER";
    case alert_cause::kMaintenance: return "MAINTENANCE";
    case alert_cause::kConstruction: return "CONSTRUCTION";
    case alert_cause::kPoliceActivity: return "POLICE_ACTIVITY";
    case alert_cause::kMedicalEmergency: return "MEDICAL_EMERGENCY";
  }
  std::unreachable();
}

std::ostream& operator<<(std::ostream& out, alert_cause const x) {
  return out << to_str(x);
}

std::string_view to_str(alert_effect const x) {
  switch (x) {
    case alert_effect::kNoService: return "NO_SERVICE";
    case alert_effect::kReducedService: return "REDUCED_SERVICE";
    case alert_effect::kSignificantDelays: return "SIGNIFICANT_DELAYS";
    case alert_effect::kDetour: return "DETOUR";
    case alert_effect::kAdditionalService: return "ADDITIONAL_SERVICE";
    case alert_effect::kModifiedService: return "MODIFIED_SERVICE";
    case alert_effect::kOtherEffect: return "OTHER_EFFECT";
    case alert_effect::kUnknownEffect: return "UNKNOWN_EFFECT";
    case alert_effect::kStopMoved: return "STOP_MOVED";
    case alert_effect::kNoEffect: return "NO_EFFECT";
    case alert_effect::kAccessibilityIssue: return "ACCESSIBILITY_ISSUE";
  }
  std::unreachable();
}

std::ostream& operator<<(std::ostream& out, alert_effect const x) {
  return out << to_str(x);
}

std::string_view to_str(alert_severity const x) {

  switch (x) {
    case alert_severity::kUnknownSeverity: return "UNKNOWN_SEVERITY";
    case alert_severity::kInfo: return "INFO";
    case alert_severity::kWarning: return "WARNING";
    case alert_severity::kSevere: return "SEVERE";
  }
  std::unreachable();
}

std::ostream& operator<<(std::ostream& out, alert_severity const x) {
  return out << to_str(x);
}

}  // namespace nigiri