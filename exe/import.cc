#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/program_options.hpp"

#include "date/date.h"

#include "utl/progress_tracker.h"

#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
using namespace nigiri::loader;
using namespace std::string_literals;

date::sys_days parse_date(std::string const& str) {
  if (str == "TODAY") {
    return std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
  }

  date::sys_days parsed;
  std::stringstream ss;
  ss << str;
  ss >> date::parse("%F", parsed);
  return parsed;
}

int main(int ac, char** av) {
  auto const progress_tracker = utl::activate_progress_tracker("importer");
  auto const silencer = utl::global_progress_bars{true};

  auto input_path = fs::path{};
  auto output_path = fs::path{"tt.bin"};
  auto start_date = "TODAY"s;
  auto n_days = 365U;
  auto link_stop_distance = 100U;
  auto tz = "Europe/Berlin"s;
  auto recursive = false;

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("in,i", bpo::value(&input_path),
       "input path (either a ZIP file or a directory containing ZIPs)")  //
      ("out,o", bpo::value(&output_path)->default_value(output_path),
       "output file path")  //
      ("start_date,s", bpo::value(&start_date)->default_value(start_date),
       "start date of the timetable, format: YYYY-MM-DD")  //
      ("num_days,n", bpo::value(&n_days)->default_value(n_days),
       "the length of the timetable in days")  //
      ("link_stop_distance,l",
       bpo::value(&link_stop_distance)->default_value(link_stop_distance),
       "the maximum distance at which stops in proximity will be linked")  //
      ("tz,t", bpo::value(&tz)->default_value(tz),
       "the default time zone to use");  //

  auto vm = bpo::variables_map{};
  bpo::store(bpo::command_line_parser(ac, av).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  auto input_files = std::vector<fs::path>{};
  if (is_directory(input_path) && recursive) {
    for (auto const& e : fs::directory_iterator(input_path)) {
      if (is_directory(e) /* unpacked zip file */ ||
          boost::algorithm::to_lower_copy(
              e.path().extension().generic_string()) == ".zip") {
        input_files.emplace_back(e.path());
      }
    }
  } else if (exists(input_path) && !recursive) {
    input_files.emplace_back(input_path);
  }

  if (input_files.empty()) {
    std::cerr << "no input file found\n";
    return 1;
  }

  auto const start = parse_date(start_date);
  load(input_files, {link_stop_distance, tz},
       {start, start + date::days{n_days}})
      .write(output_path);
}