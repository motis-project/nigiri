#include "nigiri/loader/hrd/timezone.h"

#include "nigiri/logging.h"

#include "nigiri/loader/hrd/util.h"
#include "utl/parser/arg_parser.h"

namespace nigiri::loader::hrd {

duration_t distance_to_midnight(utl::cstr s) {
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

tz_offsets const& get_tz(timezone_map_t const& tz,
                         eva_number const eva_number) {
  utl::verify(!tz.empty(), "no timezones");
  auto const it = tz.upper_bound(eva_number);
  utl::verify(it != end(tz) || std::prev(it)->first <= eva_number,
              "no timezone for eva number {}", eva_number);
  return std::prev(it)->second;
}

timezone_map_t parse_timezones(config const& c, std::string_view file_content) {
  timezone_map_t tz;
  utl::for_each_line(file_content, [&](utl::cstr line) {
    if (line.length() == 15) {
      auto const first_valid_eva_number =
          parse_eva_number(line.substr(c.tz_.type1_first_valid_eva_));
      auto const it = tz.find(first_valid_eva_number);
      if (it != end(tz)) {
        tz[eva_number(line.substr(c.tz_.type1_eva_))] = it->second;
      } else {
        log(log_lvl::error, "nigiri.loader.hrd.timezone",
            "no timezone for eva number: {}", first_valid_eva_number);
      }
      return;
    }

    if ((isdigit(line[0]) != 0) && line.length() >= 47) {
      auto const eva_num = eva_number(line.substr(c.tz_.type2_eva_));
      tz_offsets t;
      t.offset_ = duration_t{
          distance_to_midnight(line.substr(c.tz_.type2_dst_to_midnight_))};
      if (!line.substr(14, utl::size(33)).trim().empty()) {
        t.season_ = tz_offsets::season{
            .offset_ = distance_to_midnight(
                line.substr(c.tz_.type3_dst_to_midnight1_)),
            .begin_ = parse_date(line.substr(c.tz_.type3_bitfield_idx1_)),
            .end_ = parse_date(line.substr(c.tz_.type3_bitfield_idx2_)),
            .season_begin_mam_ = distance_to_midnight(
                line.substr(c.tz_.type3_dst_to_midnight2_)),
            .season_end_mam_ = distance_to_midnight(
                line.substr(c.tz_.type3_dst_to_midnight3_))};
      }
      std::cout << t << "\n";
      tz[eva_num] = t;
    }
  });

  return tz;
}

}  // namespace nigiri::loader::hrd
