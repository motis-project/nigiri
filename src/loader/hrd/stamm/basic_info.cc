#include "utl/parser/arg_parser.h"

#include "nigiri/loader/hrd/stamm/basic_info.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

std::chrono::year_month_day yyyymmdd(utl::cstr s) {
  return {std::chrono::year{parse_verify<int>(s.substr(6, utl::size(4)))},
          std::chrono::month{parse_verify<unsigned>(s.substr(3, utl::size(2)))},
          std::chrono::day{parse_verify<unsigned>(s.substr(0, utl::size(2)))}};
}

std::pair<utl::cstr, utl::cstr> mask_dates(utl::cstr str) {
  utl::cstr from_line, to_line;

  from_line = get_line(str).substr(0, utl::size(10));
  while (from_line.starts_with("%")) {
    skip_line(str);
    from_line = get_line(str).substr(0, utl::size(10));
  }

  skip_line(str);

  to_line = get_line(str).substr(0, utl::size(10));
  while (to_line.starts_with("%")) {
    skip_line(str);
    to_line = get_line(str).substr(0, utl::size(10));
  }

  return {from_line, to_line};
}

interval<std::chrono::sys_days> parse_interval(std::string_view file_content) {
  auto const [first_date, last_date] = mask_dates(file_content);
  return {{std::chrono::sys_days{yyyymmdd(first_date)}},
          {std::chrono::sys_days{yyyymmdd(last_date)}}};
}

std::string parse_schedule_name(std::string_view file_content) {
  auto basic_info_file = utl::cstr{file_content};
  utl::skip_line(basic_info_file);  // from
  utl::skip_line(basic_info_file);  // to
  return iso_8859_1_to_utf8(get_line(basic_info_file).view());  // schedule name
}

}  // namespace nigiri::loader::hrd
