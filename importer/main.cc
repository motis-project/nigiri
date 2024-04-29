#include <vector>
#include <regex>

#include "date/date.h"

#include "boost/program_options.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

std::vector<std::string> tokenize(std::string const& str,
                                  char delimiter,
                                  std::uint32_t n_tokens) {
  auto tokens = std::vector<std::string>{};
  tokens.reserve(n_tokens);
  auto start = 0U;
  for (auto i = 0U; i != n_tokens; ++i) {
    auto end = str.find(delimiter, start);
    if (end == std::string::npos && i != n_tokens - 1U) {
      break;
    }
    tokens.emplace_back(str.substr(start, end - start));
    start = end + 1U;
  }
  return tokens;
}

std::optional<date::sys_days> parse_date(std::string const& str) {

  auto const date_regex = std::regex{"^[0-9]{4}-[0|1][0-9]-[0-3][0-9]$"};
  if (!std::regex_match(begin(str), end(str), date_regex)) {
    return std::nullopt;
  }
  auto const tokens = tokenize(str, '-', 3U);
  return date::year_month_day{
      date::year{std::stoi(tokens[0])},
      date::month{static_cast<std::uint32_t>(std::stoul(tokens[1]))},
      date::day{static_cast<std::uint32_t>(std::stoul(tokens[2]))}};
}

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  auto const progress_tracker = utl::activate_progress_tracker("importer");
  auto const silencer = utl::global_progress_bars{true};

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "produce this help message")
    ("path,p", bpo::value<std::string>(),"path to the timetable, can be a directory or a zip file")
    ("zips,z", bpo::value<bool>()->default_value(false), "if a directory is provided as input path, loads all zip files within the directory")
    ("start_date,s", bpo::value<std::string>(), "start date of the timetable, format: YYYY-MM-DD")
    ("end_date,e", bpo::value<std::string>(), "end date of the timetable, format: YYYY-MM-DD")
    ("out_file,o",bpo::value<std::string>()->default_value("tt.bin"),"the name of the output file")
    ("link_stop_distance,l",bpo::value<std::uint32_t>()->default_value(100U),"the maximum distance at which stops in proximity will be linked")
    ("time_zone,t", bpo::value<std::string>()->default_value("Europe/Berlin"), "the default time zone to use")
  ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);  // sets default values

  auto const config = nigiri::loader::loader_config{vm["link_stop_distance"].as<std::uint32_t>(), vm["time_zone"].as<std::string>()};

  return 0;
}