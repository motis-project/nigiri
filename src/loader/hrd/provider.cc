#include "nigiri/loader/hrd/provider.h"

#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

namespace nigiri::loader::hrd {

using provider_map_t = hash_map<string, provider>;

void verify_line_format(utl::cstr line, char const* filename, int line_number) {
  // Verify that the provider number has 5 digits.
  auto const provider_number = line.substr(0, utl::size(5));
  utl::verify(std::all_of(begin(provider_number), end(provider_number),
                          [](char c) { return std::isdigit(c); }),
              "provider line format mismatch in {}:{}", filename, line_number);

  utl::verify(line[6] == 'K' || line[6] == ':',
              "provider line format mismatch in {}:{}", filename, line_number);
}

std::string parse_name(utl::cstr s) {
  auto const start_is_quote = (s[0] == '\'' || s[0] == '\"');
  auto const end = start_is_quote ? s[0] : ' ';
  auto i = start_is_quote ? 1 : 0;
  while (s && s[i] != end) {
    ++i;
  }
  auto region = s.substr(start_is_quote ? 1 : 0, utl::size(i - 1));
  return {region.str, region.len};
}

provider read_provider_names(utl::cstr line, int line_number) {
  int short_name = line.substr_offset(" K ");
  int long_name = line.substr_offset(" L ");
  int full_name = line.substr_offset(" V ");

  utl::verify(short_name != -1 && long_name != -1 && full_name != -1,
              "provider line format mismatch in line {}", line_number);

  return provider{.short_name_ = parse_name(line.substr(long_name + 3)),
                  .long_name_ = parse_name(line.substr(full_name + 3))};
}

provider_map_t parse_providers(config const& c, std::string_view file_content) {
  scoped_timer timer("parsing providers");

  provider_map_t providers;
  provider current_info;
  int previous_provider_number = 0;

  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, int line_number) {
        auto provider_number = utl::parse<int>(line.substr(c.track_.prov_nr_));
        if (line.length() > 6 && line[6] == 'K') {
          current_info = read_provider_names(line, line_number);
          previous_provider_number = provider_number;
        } else if (line.length() > 8) {
          utl::verify(previous_provider_number == provider_number,
                      "provider line format mismatch in line {}", line_number);
          for_each_token(line.substr(8), ' ', [&](utl::cstr token) {
            providers[token.to_str()] = current_info;
          });
        }
      });

  return providers;
}

}  // namespace nigiri::loader::hrd