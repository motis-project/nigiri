#include "nigiri/qa/qa.h"

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

  auto const ref =
      std::make_unique<cista::wrapped<nigiri::qa::benchmark_criteria>>(
          nigiri::qa::benchmark_criteria::read(
              cista::memory_holder{cista::file{in_r.c_str(), "r"}.content()}));

  auto const cmp =
      std::make_unique<cista::wrapped<nigiri::qa::benchmark_criteria>>(
          nigiri::qa::benchmark_criteria::read(
              cista::memory_holder{cista::file{in_c.c_str(), "r"}.content()}));

  auto rating_timing =
      std::vector<std::pair<nigiri::qa::rating_t, std::chrono::milliseconds>>{};
  rating_timing.reserve((**ref).qc_.size());

  for (auto const& qc_ref : (**ref).qc_) {
    for (auto const& qc_cmp : (**cmp).qc_) {
      if (qc_ref.query_idx_ == qc_cmp.query_idx_) {
        auto const rating = nigiri::qa::rate(qc_ref.jc_, qc_cmp.jc_);
        auto const timing = qc_ref.query_time_ - qc_cmp.query_time_;
        rating_timing.emplace_back(rating, timing);
        break;
      }
    }
  }

  return 0;
}