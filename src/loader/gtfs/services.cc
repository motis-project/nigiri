#include "nigiri/loader/gtfs/services.h"

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

enum class bound { kFirst, kLast };

bitfield calendar_to_bitfield(
    interval<date::sys_days> const& tt_interval,
    calendar const& c,
    std::optional<date::sys_days> const& feed_end_date) {
  auto const extend = feed_end_date.has_value() &&
                      *feed_end_date == c.interval_.to_ - date::days{1};
  if (!tt_interval.overlaps(c.interval_) && !extend) {
    return {};
  }
  auto const from = tt_interval.clamp(c.interval_.from_);
  auto const to = extend ? tt_interval.to_ : tt_interval.clamp(c.interval_.to_);
  auto bit = (from - tt_interval.from_).count();
  auto traffic_days = bitfield{};
  for (auto d = from; d < to && bit < kMaxDays; d = d + date::days{1}, ++bit) {
    traffic_days.set(
        static_cast<std::size_t>(bit),
        c.week_days_.test(date::year_month_weekday{d}.weekday().c_encoding()));
  }
  return traffic_days;
}

void add_exception(interval<date::sys_days> const& tt_interval,
                   calendar_date const& exception,
                   bitfield& b) {
  auto const day_idx = (exception.day_ - tt_interval.from_).count();
  if (day_idx < 0 || day_idx >= kMaxDays) {
    return;
  }
  b.set(static_cast<unsigned>(day_idx), exception.type_ == calendar_date::kAdd);
}

traffic_days_t merge_traffic_days(
    interval<date::sys_days> const& tt_interval,
    hash_map<std::string, calendar> const& base,
    hash_map<std::string, std::vector<calendar_date>> const& exceptions,
    std::optional<date::sys_days> const& feed_end_date) {
  auto const timer = nigiri::scoped_timer{"loader.gtfs.services"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Build Base Services")
      .out_bounds(33.F, 35.F)
      .in_high(base.size());

  auto s = traffic_days_t{};
  for (auto const& [service_name, calendar] : base) {
    s[service_name] = std::make_unique<bitfield>(
        calendar_to_bitfield(tt_interval, calendar, feed_end_date));
  }

  progress_tracker->status("Add Service Exceptions")
      .out_bounds(35.F, 37.F)
      .in_high(base.size());
  for (auto const& [service_name, service_exceptions] : exceptions) {
    for (auto const& day : service_exceptions) {
      add_exception(tt_interval, day,
                    *utl::get_or_create(s, service_name, []() {
                      return std::make_unique<bitfield>();
                    }));
    }
  }

  return s;
}

}  // namespace nigiri::loader::gtfs
