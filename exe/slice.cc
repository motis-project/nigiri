#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/program_options.hpp"

#include "date/date.h"

#include "nigiri/common/parse_date.h"
#include "nigiri/slice.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
using namespace nigiri;

int main(int ac, char** av) {
  auto in = fs::path{"tt.bin"};
  auto out = fs::path{"slice.bin"};
  auto start_date = std::string{};
  auto n_days = 365U;

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("in,i", bpo::value(&in)->default_value(in), "input file path")  //
      ("out,o", bpo::value(&out)->default_value(out), "output file path")  //
      ("start_date,d", bpo::value(&start_date), "start date")  //
      ("n_days,n", bpo::value(&n_days), "number of days to extract to slice");

  auto vm = bpo::variables_map{};
  bpo::store(bpo::command_line_parser(ac, av).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const tt = timetable::read(in);
  auto const from = parse_date(start_date);
  auto const to = from + (n_days + 1U) * date::days{1U};
  slice(*tt, {from, to});
}