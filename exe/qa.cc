#include "boost/program_options.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
namespace bpo = boost::program_options;

int main(int ac, char** av) {
  auto in_r = fs::path{};
  auto in_c = fs::path{};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("reference,r", bpo::value(&in_r),
       "input path a, binary dump of vector<pareto_set<journey>>")  //
      ("compare,c", bpo::value(&in_c),
       "input path b, binary dump of vector<pareto_set<journey>>");
  auto const pos = bpo::positional_options_description{}.add("in", -1);

  auto vm = bpo::variables_map{};
  bpo::store(
      bpo::command_line_parser(ac, av).options(desc).positional(pos).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  if (vm.count("reference") != 1) {
    std::cout << "Error: please provide exactly one reference file\n";
    return 1;
  }

  if (vm.count("compare") != 1) {
    std::cout << "Error: please provide exactly one compare file\n";
    return 1;
  }
}