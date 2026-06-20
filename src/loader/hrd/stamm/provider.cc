#include "nigiri/loader/hrd/stamm/provider.h"

#include "nigiri/loader/hrd/util.h"
#include "nigiri/loader/register.h"
#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

namespace nigiri::loader::hrd {

void verify_line_format(utl::cstr line,
                        char const* filename,
                        unsigned const line_number) {
  // Verify that the provider number has 5 digits.
  auto const provider_number = line.substr(0, utl::size(5));
  utl::verify(std::all_of(begin(provider_number), end(provider_number),
                          [](char c) { return std::isdigit(c); }),
              "provider line format mismatch in {}:{}", filename, line_number);

  utl::verify(line[6] == 'K' || line[6] == ':',
              "provider line format mismatch in {}:{}", filename, line_number);
}

std::string_view parse_name(utl::cstr s) {
  auto const start_is_quote = (s[0] == '\'' || s[0] == '\"');
  auto const end = start_is_quote ? s[0] : ' ';
  auto i = start_is_quote ? 1U : 0U;
  while (s && s[i] != end) {
    ++i;
  }
  auto region = s.substr(start_is_quote ? 1 : 0, utl::size(i - 1));
  return {region.str, static_cast<unsigned>(region.len)};
}

provider read_provider_names(timetable& tt,
                             utl::cstr line,
                             source_idx_t const src) {
  auto const long_name = line.substr_offset(" L ");
  utl::verify(long_name != std::numeric_limits<size_t>::max(),
              "no long name found: {}", line.view());
  auto const full_name = line.substr_offset(" V ");
  utl::verify(long_name != std::numeric_limits<size_t>::max(),
              "no full name found: {}", line.view());
  return provider{.id_ = tt.strings_.store(iso_8859_1_to_utf8(
                      parse_name(line.substr(long_name + 3U)))),
                  .name_ = tt.register_translation(iso_8859_1_to_utf8(
                      parse_name(line.substr(full_name + 3U)))),
                  .url_ = kEmptyTranslation,
                  .src_ = src};
}

provider_map_t parse_providers(config const& c,
                               source_idx_t const src,
                               timetable& tt,
                               std::string_view file_content) {
  auto const timer = scoped_timer{"parse providers"};

  provider_map_t providers;
  provider current_info;
  int previous_provider_number = 0;

  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, unsigned const line_number) {
        auto provider_number = utl::parse<int>(line.substr(c.track_.prov_nr_));
        if (line.length() > 6 && line[6] == 'K') {
          current_info = read_provider_names(tt, line, src);
          previous_provider_number = provider_number;
        } else if (line.length() > 8) {
          utl::verify(previous_provider_number == provider_number,
                      "provider line format mismatch in line {}", line_number);
          for_each_token(line.substr(8), ' ', [&](utl::cstr token) {
            providers[token.to_str()] = register_agency(
                tt, agency{tt, src, tt.strings_.get(current_info.id_),
                           current_info.name_, kEmptyTranslation,
                           current_info.tz_, dummy_tz_map});
          });
        }
      });

  return providers;
}

}  // namespace nigiri::loader::hrd
