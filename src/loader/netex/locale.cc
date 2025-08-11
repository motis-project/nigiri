#include "nigiri/loader/netex/locale.h"

#include <functional>
#include <string_view>

#include "utl/get_or_create.h"
#include "utl/verify.h"

namespace nigiri::loader::netex {

std::string_view guess_time_zone(std::string_view const offset,
                                 std::string_view const summer_offset) {
  char* wo_end = nullptr;
  auto const wo = std::strtod(offset.data(), &wo_end);
  utl::verify(wo_end != offset.data(), "Invalid time zone offset: {}", offset);
  char* so_end = nullptr;
  auto const so = std::strtod(summer_offset.data(), &so_end);
  utl::verify(so_end != summer_offset.data(), "Invalid time zone offset: {}",
              summer_offset);

  // European Time Zones
  // All these time zones (currently) have the same DST rules
  if (std::equal_to<>()(wo, 0.0) && std::equal_to<>()(so, 0.0)) {
    return "UTC";
  } else if (std::equal_to<>()(wo, 1.0) && std::equal_to<>()(so, 2.0)) {
    return "CET";
  } else if (std::equal_to<>()(wo, 0.0) && std::equal_to<>()(so, 1.0)) {
    return "Europe/London";
  } else if (std::equal_to<>()(wo, 2.0) && std::equal_to<>()(so, 3.0)) {
    return "Europe/Helsinki";
  } else if (std::equal_to<>()(wo, -1.0) && std::equal_to<>()(so, 0.0)) {
    return "Atlantic/Azores";
  }
  throw utl::fail("Unknown timezone for offset {} and summer offset {}", offset,
                  summer_offset);
}

timezone_idx_t get_tz_idx(netex_data& data, std::string_view tz_name) {
  utl::verify(!tz_name.empty(), "timezone not set");
  return utl::get_or_create(data.timezones_, tz_name, [&]() {
    return data.tt_.locations_.register_timezone(timezone{
        cista::pair{string{tz_name},
                    static_cast<void const*>(date::locate_zone(tz_name))}});
  });
}

netex_locale parse_locale(netex_data& data,
                          netex_ctx const& ctx,
                          pugi::xml_node const& locale_node) {
  auto const& tz_name = locale_node.child("TimeZone");
  auto const& tz_offset = locale_node.child("TimeZoneOffset");
  auto const& tz_summer_offset = locale_node.child("SummerTimeZoneOffset");

  auto locale = netex_locale{
      .language_ = locale_node.child("DefaultLanguage").child_value(),
      .tz_name_ = tz_name.child_value(),
      .tz_offset_ = tz_offset.child_value(),
      .tz_summer_offset_ = tz_summer_offset.child_value(),
  };

  if (ctx.locale_ && !tz_name && !tz_offset && !tz_summer_offset) {
    locale.tz_name_ = ctx.locale_->tz_name_;
    locale.tz_offset_ = ctx.locale_->tz_offset_;
    locale.tz_summer_offset_ = ctx.locale_->tz_summer_offset_;
  }

  if (!locale.tz_name_.empty()) {
    locale.tz_idx_ = get_tz_idx(data, locale.tz_name_);
  } else if (!locale.tz_offset_.empty() && !locale.tz_summer_offset_.empty()) {
    auto const guessed_tz_name =
        guess_time_zone(locale.tz_offset_, locale.tz_summer_offset_);
    locale.tz_idx_ = get_tz_idx(data, guessed_tz_name);
  }

  return locale;
}

std::optional<netex_locale> get_default_locale(netex_data& data,
                                               loader_config const& config) {
  return config.default_tz_.empty()
             ? std::nullopt
             : std::optional{netex_locale{
                   .tz_name_ = config.default_tz_,
                   .tz_idx_ = get_tz_idx(data, config.default_tz_)}};
}

}  // namespace nigiri::loader::netex
