#include "nigiri/rt/gtfsrt_alert.h"

#include <ranges>

#include "utl/verify.h"

#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

namespace nigiri::rt {

std::string remove_nl(std::string);

unixtime_t sec_to_unixtime(std::uint64_t const s) {
  return unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
      std::chrono::seconds{s})};
}

interval<unixtime_t> to_interval(transit_realtime::TimeRange const& t) {
  return {sec_to_unixtime(t.start()), sec_to_unixtime(t.end())};
}

alert_cause convert(transit_realtime::Alert_Cause x) {
  using namespace transit_realtime;
  switch (x) {
    case Alert_Cause_UNKNOWN_CAUSE: return alert_cause::kUnknownCause;
    case Alert_Cause_OTHER_CAUSE: return alert_cause::kOtherCause;
    case Alert_Cause_TECHNICAL_PROBLEM: return alert_cause::kTechnicalProblem;
    case Alert_Cause_STRIKE: return alert_cause::kStrike;
    case Alert_Cause_DEMONSTRATION: return alert_cause::kDemonstration;
    case Alert_Cause_ACCIDENT: return alert_cause::kAccident;
    case Alert_Cause_HOLIDAY: return alert_cause::kHoliday;
    case Alert_Cause_WEATHER: return alert_cause::kWeather;
    case Alert_Cause_MAINTENANCE: return alert_cause::kMaintenance;
    case Alert_Cause_CONSTRUCTION: return alert_cause::kConstruction;
    case Alert_Cause_POLICE_ACTIVITY: return alert_cause::kPoliceActivity;
    case Alert_Cause_MEDICAL_EMERGENCY: return alert_cause::kMedicalEmergency;
  }
  throw utl::fail("unknown Alert_Cause");
}

alert_effect convert(transit_realtime::Alert_Effect x) {
  using namespace transit_realtime;
  switch (x) {
    case Alert_Effect_NO_SERVICE: return alert_effect::kNoService;
    case Alert_Effect_REDUCED_SERVICE: return alert_effect::kReducedService;
    case Alert_Effect_SIGNIFICANT_DELAYS:
      return alert_effect::kSignificantDelays;
    case Alert_Effect_DETOUR: return alert_effect::kDetour;
    case Alert_Effect_ADDITIONAL_SERVICE:
      return alert_effect::kAdditionalService;
    case Alert_Effect_MODIFIED_SERVICE: return alert_effect::kModifiedService;
    case Alert_Effect_OTHER_EFFECT: return alert_effect::kOtherEffect;
    case Alert_Effect_UNKNOWN_EFFECT: return alert_effect::kUnknownEffect;
    case Alert_Effect_STOP_MOVED: return alert_effect::kStopMoved;
    case Alert_Effect_NO_EFFECT: return alert_effect::kNoEffect;
    case Alert_Effect_ACCESSIBILITY_ISSUE:
      return alert_effect::kAccessibilityIssue;
  }
  throw utl::fail("unknown Alert_Effect");
}

alert_severity convert(transit_realtime::Alert_SeverityLevel x) {
  using namespace transit_realtime;
  switch (x) {
    case Alert_SeverityLevel_UNKNOWN_SEVERITY:
      return alert_severity::kUnknownSeverity;
    case Alert_SeverityLevel_INFO: return alert_severity::kInfo;
    case Alert_SeverityLevel_WARNING: return alert_severity::kWarning;
    case Alert_SeverityLevel_SEVERE: return alert_severity::kSevere;
  }
  throw utl::fail("unknown Alert_SeverityLevel");
}

void handle_alert(date::sys_days const today,
                  timetable const& tt,
                  rt_timetable& rtt,
                  string_cache_t& c,
                  source_idx_t const src,
                  std::string_view tag,
                  transit_realtime::Alert const& a,
                  statistics& stats) {
  ++stats.total_alerts_;

  auto& alerts = rtt.service_alerts_;

  auto const alert_idx =
      service_alert_idx_t{alerts.communication_period_.size()};

  // =========================
  // Resolve informed entities
  // -------------------------
  auto any_resolved = false;
  stats.alert_total_informed_entities_ += a.informed_entity_size();
  for (auto const& x : a.informed_entity()) {
    if (x.has_trip()) {
      auto [r, trip] = gtfsrt_resolve_run(today, tt, &rtt, src, x.trip());
      if (!r.valid()) {
        log(log_lvl::error, "rt.gtfs.resolve.alert",
            "could not resolve (tag={}) {}", tag,
            remove_nl(x.trip().DebugString()));
        ++stats.alert_informed_entity_resolve_error_;
        continue;
      }
      if (!r.is_rt()) {
        r.rt_ = rtt.add_rt_transport(src, tt, r.t_);
      }
      auto const stop = x.has_stop_id()
                            ? tt.locations_.find({x.stop_id(), src})
                                  .and_then([](location const& l) {
                                    return std::optional{l.l_};
                                  })
                                  .value_or(location_idx_t::invalid())
                            : location_idx_t::invalid();
      alerts.rt_service_alerts_[r.rt_].push_back({alert_idx, stop});
      any_resolved = true;
    } else {
      ++stats.alert_informed_entity_resolve_error_;
    }
  }

  if (!any_resolved) {
    return;
  }

  // =============
  // Create alert.
  // -------------
  auto& s = alerts.strings_;

  auto const to_translation =
      [&](transit_realtime::TranslatedString_Translation const& x) {
        utl::verify(x.has_text(), "GTFS RT Translation requires text");
        return translation{.text_ = s.register_string(c, x.text()),
                           .language_ = x.has_language()
                                            ? s.register_string(c, x.language())
                                            : string_idx_t::invalid()};
      };

  auto const to_localized_image =
      [&](transit_realtime::TranslatedImage_LocalizedImage const& x) {
        utl::verify(x.has_url() && x.has_media_type(),
                    "GTFS RT LocatizedImage requires URL and media_type");
        return localized_image{
            .url_ = s.register_string(c, x.url()),
            .media_type_ = s.register_string(c, x.media_type()),
            .language_ = x.has_language() ? s.register_string(c, x.language())
                                          : string_idx_t::invalid()};
      };

  auto const translate = [&](auto&& x) {
    return x.translation() | std::views::transform(to_translation);
  };

  auto const period = a.active_period() | std::views::transform(to_interval);
  alerts.communication_period_.emplace_back(period);
  alerts.impact_period_.emplace_back(period);

  if (a.has_cause_detail()) {
    alerts.cause_detail_.emplace_back(translate(a.cause_detail()));
  } else {
    alerts.cause_detail_.emplace_back(std::initializer_list<translation>{});
  }

  if (a.has_effect_detail()) {
    alerts.effect_detail_.emplace_back(translate(a.effect_detail()));
  } else {
    alerts.effect_detail_.emplace_back(std::initializer_list<translation>{});
  }

  if (a.has_url()) {
    alerts.url_.emplace_back(translate(a.url()));
  } else {
    alerts.url_.emplace_back(std::initializer_list<translation>{});
  }

  if (a.has_header_text()) {
    alerts.header_text_.emplace_back(translate(a.header_text()));
  } else {
    alerts.header_text_.emplace_back(std::initializer_list<translation>{});
  }

  if (a.has_description_text()) {
    alerts.description_text_.emplace_back(translate(a.description_text()));
  } else {
    alerts.description_text_.emplace_back(std::initializer_list<translation>{});
  }

  if (a.has_tts_header_text()) {
    alerts.tts_header_text_.emplace_back(translate(a.tts_header_text()));
  } else {
    alerts.tts_header_text_.emplace_back(std::initializer_list<translation>{});
  }

  if (a.has_tts_description_text()) {
    alerts.tts_description_text_.emplace_back(
        translate(a.tts_description_text()));
  } else {
    alerts.tts_description_text_.emplace_back(
        std::initializer_list<translation>{});
  }

  if (a.has_image_alternative_text()) {
    alerts.image_alternative_text_.emplace_back(
        translate(a.image_alternative_text()));
  } else {
    alerts.image_alternative_text_.emplace_back(
        std::initializer_list<translation>{});
  }

  if (a.has_image()) {
    alerts.image_.emplace_back(a.image().localized_image() |
                               std::views::transform(to_localized_image));
  } else {
    alerts.image_.emplace_back(std::initializer_list<localized_image>{});
  }

  alerts.cause_.emplace_back(convert(
      a.has_cause() ? a.cause() : transit_realtime::Alert_Cause_UNKNOWN_CAUSE));
  alerts.effect_.emplace_back(
      convert(a.has_effect() ? a.effect()
                             : transit_realtime::Alert_Effect_UNKNOWN_EFFECT));
  alerts.severity_level_.emplace_back(
      convert(a.has_severity_level()
                  ? a.severity_level()
                  : transit_realtime::Alert_SeverityLevel_UNKNOWN_SEVERITY));
}

}  // namespace nigiri::rt