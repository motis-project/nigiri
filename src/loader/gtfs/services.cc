#include "nigiri/loader/gtfs/services.h"

#include "nigiri/logging.h"
#include "utl/get_or_create.h"

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
          return lhs.second.first_day_ < rhs.second.first_day_;
        });
    return it == end(base) ? std::pair{"", kMin}
                           : std::pair{it->first, it->second.first_day_};
  };

  auto const max_base_day = [&]() {
    auto const it =
        std::max_element(begin(base), end(base), [](auto&& lhs, auto&& rhs) {
          return lhs.second.last_day_ < rhs.second.last_day_;
        });
    return it == end(base) ? std::pair{"", kMax}
                           : std::pair{it->first, it->second.last_day_};
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
          if (date.type_ == calendar_date::kAdd && date.day_ > max) {
            max = date.day_;
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
                              date::sys_days const& start,
                              calendar const& c) {
  auto const first = std::min(start, c.first_day_);
  auto const last = std::min(start + kMaxDays * 1_days, c.last_day_ + 1_days);

  bitfield traffic_days;
  auto bit = static_cast<std::size_t>((first - start).count());
  for (auto d = first; d != last; d = d + date::sys_days::duration{1}, ++bit) {
    if (bit >= traffic_days.size()) {
      log(log_lvl::error, "loader.gtfs.services",
          "date {} for servcie {} out of range", d, service_name);
      continue;
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
  if (day_idx < 0 || day_idx >= static_cast<int>(b.size())) {
    log(log_lvl::error, "loader.gtfs.services",
        "date {} for service {} out of range", exception.day_, service_name);
    return;
  }
  b.set(static_cast<unsigned>(day_idx), exception.type_ == calendar_date::kAdd);
}

traffic_days merge_traffic_days(
    hash_map<std::string, calendar> const& base,
    hash_map<std::string, std::vector<calendar_date>> const& exceptions) {
  nigiri::scoped_timer timer{"traffic days"};

  traffic_days s;
  s.first_day_ = bound_date(base, exceptions, bound::kFirst);
  s.last_day_ = bound_date(base, exceptions, bound::kLast);

  for (auto const& [service_name, calendar] : base) {
    s.traffic_days_[service_name] = std::make_unique<bitfield>(
        calendar_to_bitfield(service_name, s.first_day_, calendar));
  }

  for (auto const& [service_name, service_exceptions] : exceptions) {
    for (auto const& day : service_exceptions) {
      add_exception(service_name, s.first_day_, day,
                    *utl::get_or_create(s.traffic_days_, service_name, []() {
                      return std::make_unique<bitfield>();
                    }));
    }
  }

  return s;
}

}  // namespace nigiri::loader::gtfs
