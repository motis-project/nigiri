#include "nigiri/loader/hrd/stamm/timezone.h"

#include "utl/parser/arg_parser.h"

#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"

namespace nigiri::loader::hrd {

duration_t parse_mam(utl::cstr s) {
  return duration_t{hhmm_to_min(parse<int>(s))};
}

unixtime_t parse_date(utl::cstr s) {
  auto const date = std::chrono::year_month_day{
      std::chrono::year{utl::parse_verify<int>(s.substr(4, utl::size(4)))},
      std::chrono::month{
          utl::parse_verify<unsigned>(s.substr(2, utl::size(2)))},
      std::chrono::day{utl::parse_verify<unsigned>(s.substr(0, utl::size(2)))}};
  return unixtime_t{std::chrono::sys_days{date}};
}

vector<tz_offsets::season> parse_seasons(utl::cstr const line) {
  enum state {
    kSeasonOffset,
    kSeasonBeginDate,
    kSeasonBeginHour,
    kSeasonEndDate,
    kSeasonEndHour,
    kNumStates
  } s{kSeasonOffset};
  vector<tz_offsets::season> seasons;
  auto e = tz_offsets::season{};
  for_each_token(line, ' ', [&](utl::cstr const t) {
    switch (s) {
      case kSeasonOffset: e.offset_ = parse_mam(t); break;
      case kSeasonBeginDate: e.begin_ = parse_date(t); break;
      case kSeasonBeginHour: e.season_begin_mam_ = parse_mam(t); break;
      case kSeasonEndDate: e.end_ = parse_date(t); break;
      case kSeasonEndHour:
        e.season_end_mam_ = parse_mam(t);
        seasons.push_back(e);
        break;
      default:;
    }
    s = static_cast<state>((s + 1) % kNumStates);
  });
  return seasons;
}

timezone_map_t parse_timezones(config const& c,
                               timetable& tt,
                               std::string_view file_content) {
  auto const timer = scoped_timer{"parse timezones"};

  timezone_map_t tz;
  utl::for_each_line(file_content, [&](utl::cstr line) {
    if (auto const comment_start = line.view().find('%');
        comment_start != std::string::npos) {
      line = line.substr(0, comment_start);
    }

    if (line.length() == 15) {
      auto const first_valid_eva_number =
          parse_eva_number(line.substr(c.tz_.type1_first_valid_eva_));
      auto const it = tz.find(first_valid_eva_number);
      if (it != end(tz)) {
        tz[parse_eva_number(line.substr(c.tz_.type1_eva_))] = it->second;
      } else {
        log(log_lvl::error, "loader.hrd.timezone",
            "no timezone for eva number: {}", first_valid_eva_number);
      }
      return;
    }

    if ((std::isdigit(line[0]) != 0)) {
      auto const is_season =
          !line.substr(14, 22).empty() &&
          utl::all_of(line.substr(14, 22).view(),
                      [](char const x) { return x >= '0' && x <= '9'; });
      auto const eva = parse_eva_number(line.substr(c.tz_.type2_eva_));
      auto& t = tz[eva].second;
      if (is_season) {
        t.seasons_ = parse_seasons(line.substr(8));
        return;
      } else {
        t.offset_ =
            duration_t{parse_mam(line.substr(c.tz_.type2_dst_to_midnight_))};
        if (!line.substr(14, utl::size(33)).trim().empty()) {
          t.seasons_ = parse_seasons(line.substr(14));
        }
      }
    }
  });

  for (auto& [eva, t] : tz) {
    auto& [tz_idx, offsets] = t;
    tz_idx = tt.locations_.register_timezone(offsets);
  }

  return tz;
}

}  // namespace nigiri::loader::hrd
