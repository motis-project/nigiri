#include "nigiri/loader/gtfs/services.h"

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

enum class bound { kFirst, kLast };

bitfield calendar_to_bitfield(interval<date::sys_days> const& tt_interval,
                              std::string const& service_name,
                              calendar const& c) {
  assert((c.interval_.from_ - gtfs_interval.from_).count() >= 0);

  auto const from = std::max(c.interval_.from_, tt_interval.from_);
  auto const to = std::min(c.interval_.to_, tt_interval.to_);
  auto bit = (from - tt_interval.from_).count();
  auto traffic_days = bitfield{};
  for (auto d = from; d != to; d = d + date::days{1}, ++bit) {
    if (bit >= kMaxDays) {
      log(log_lvl::error, "loader.gtfs.services",
          "date {} for service {} out of range [tt_interval={}, calendar={}, "
          "iterating={}]",
          d, service_name, tt_interval, c.interval_, interval{from, to});
      break;
    }
    auto const weekday_index =
        date::year_month_weekday{d}.weekday().c_encoding();
    traffic_days.set(bit, c.week_days_.test(weekday_index));
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
    hash_map<std::string, std::vector<calendar_date>> const& exceptions) {
  auto const timer = nigiri::scoped_timer{"loader.gtfs.services"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Build Base Services")
      .out_bounds(36.F, 38.F)
      .in_high(base.size());

  auto s = traffic_days_t{};
  for (auto const& [service_name, calendar] : base) {
    s[service_name] = std::make_unique<bitfield>(
        calendar_to_bitfield(tt_interval, service_name, calendar));
  }

  progress_tracker->status("Add Service Exceptions")
      .out_bounds(38.F, 40.F)
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
