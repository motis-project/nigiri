#include "nigiri/loader/gtfs/services.h"

#include "nigiri/logging.h"
#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

namespace nigiri::loader::gtfs {

enum class bound { kFirst, kLast };

date::sys_days bound_date(
    hash_map<std::string, calendar> const& base,
    hash_map<std::string, std::vector<calendar_date>> const& exceptions,
    bound const b) {
  constexpr auto const kMin = date::sys_days{date::sys_days ::duration{
      std::numeric_limits<date::sys_days::rep>::max()}};
  constexpr auto const kMax = date::sys_days{date::sys_days ::duration{
      std::numeric_limits<date::sys_days::rep>::min()}};

  auto const min_base_day = [&]() {
    auto const it =
        std::min_element(begin(base), end(base), [](auto&& lhs, auto&& rhs) {
          return lhs.second.interval_.from_ < rhs.second.interval_.from_;
        });
    return it == end(base) ? std::pair{"", kMin}
                           : std::pair{it->first, it->second.interval_.from_};
  };

  auto const max_base_day = [&]() {
    auto const it =
        std::max_element(begin(base), end(base), [](auto&& lhs, auto&& rhs) {
          return lhs.second.interval_.to_ < rhs.second.interval_.to_;
        });
    return it == end(base) ? std::pair{"", kMax}
                           : std::pair{it->first, it->second.interval_.to_};
  };

  switch (b) {
    case bound::kFirst: {
      auto [min_id, min] = min_base_day();
      for (auto const& [id, dates] : exceptions) {
        for (auto const& date : dates) {
          if (date.type_ == calendar_date::kAdd && date.day_ < min) {
            min = date.day_;
            min_id = id;
          }
        }
      }
      log(log_lvl::info, "loader.gtfs.services",
          "first date {} from services {}", min, min_id);
      return min;
    }
    case bound::kLast: {
      auto [max_id, max] = max_base_day();
      for (auto const& [id, dates] : exceptions) {
        for (auto const& date : dates) {
          if (date.type_ == calendar_date::kAdd &&
              date.day_ + date::days{1} > max) {
            max = date.day_ + date::days{1};
            max_id = id;
          }
        }
      }
      log(log_lvl::info, "loader.gtfs.services",
          "last date {} from services {}", max, max_id);
      return max;
    }
  }

  assert(false);
  throw std::runtime_error{"unreachable"};
}

bitfield calendar_to_bitfield(std::string const& service_name,
                              interval<date::sys_days> const& gtfs_interval,
                              calendar const& c) {
  assert((c.interval_.from_ - gtfs_interval.from_).count() >= 0);

  bitfield traffic_days;
  auto bit = static_cast<std::size_t>(
      (c.interval_.from_ - gtfs_interval.from_).count());
  for (auto d = c.interval_.from_; d != c.interval_.to_;
       d = d + date::days{1}, ++bit) {
    if (bit >= kMaxDays) {
      log(log_lvl::error, "loader.gtfs.services",
          "date {} for service {} out of range", d, service_name);
      break;
    }
    auto const weekday_index =
        date::year_month_weekday{d}.weekday().c_encoding();
    traffic_days.set(bit, c.week_days_.test(weekday_index));
  }
  return traffic_days;
}

void add_exception(std::string const& service_name,
                   date::sys_days const& start,
                   calendar_date const& exception,
                   bitfield& b) {
  auto const day_idx = (exception.day_ - start).count();
  if (day_idx < 0 || day_idx >= kMaxDays) {
    log(log_lvl::error, "loader.gtfs.services",
        "date {} for service {} out of range", exception.day_, service_name);
    return;
  }
  b.set(static_cast<unsigned>(day_idx), exception.type_ == calendar_date::kAdd);
}

traffic_days merge_traffic_days(
    hash_map<std::string, calendar> const& base,
    hash_map<std::string, std::vector<calendar_date>> const& exceptions) {
  auto const timer = nigiri::scoped_timer{"loader.gtfs.services"};

  traffic_days s;
  s.interval_ = {bound_date(base, exceptions, bound::kFirst),
                 bound_date(base, exceptions, bound::kLast)};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Build Base Services")
      .out_bounds(36.F, 38.F)
      .in_high(base.size());
  for (auto const& [service_name, calendar] : base) {
    s.traffic_days_[service_name] = std::make_unique<bitfield>(
        calendar_to_bitfield(service_name, s.interval_, calendar));
  }

  progress_tracker->status("Add Service Exceptions")
      .out_bounds(38.F, 40.F)
      .in_high(base.size());
  for (auto const& [service_name, service_exceptions] : exceptions) {
    for (auto const& day : service_exceptions) {
      add_exception(service_name, s.interval_.from_, day,
                    *utl::get_or_create(s.traffic_days_, service_name, []() {
                      return std::make_unique<bitfield>();
                    }));
    }
  }

  return s;
}

}  // namespace nigiri::loader::gtfs
