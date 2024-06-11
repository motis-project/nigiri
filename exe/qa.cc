#include "nigiri/qa/qa.h"

#include "boost/program_options.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
namespace bpo = boost::program_options;

// needs sorted vector
template <typename T>
T quantile(std::vector<T> const& v, double q) {
  q = q < 0.0 ? 0.0 : q;
  q = 1.0 < q ? 1.0 : q;
  if (q == 1.0) {
    return v.back();
  }
  return v[static_cast<std::size_t>(v.size() * q)];
}

void print_result(std::vector<std::pair<nigiri::qa::rating_t,
                                        std::chrono::milliseconds>> const& var,
                  std::string const& var_name) {
  auto const pair_str = [](auto const& p) {
    auto ss = std::stringstream{};
    ss << "(" << p.first << ", " << p.second << ")";
    return ss.str();
  };

  std::cout << "\n--- " << var_name << " --- (n = " << var.size() << ")"
            << "\n  10%: " << pair_str(quantile(var, 0.1))
            << "\n  20%: " << pair_str(quantile(var, 0.2))
            << "\n  30%: " << pair_str(quantile(var, 0.3))
            << "\n  40%: " << pair_str(quantile(var, 0.4))
            << "\n  50%: " << pair_str(quantile(var, 0.5))
            << "\n  60%: " << pair_str(quantile(var, 0.6))
            << "\n  70%: " << pair_str(quantile(var, 0.7))
            << "\n  80%: " << pair_str(quantile(var, 0.8))
            << "\n  90%: " << pair_str(quantile(var, 0.9))
            << "\n  99%: " << pair_str(quantile(var, 0.99))
            << "\n99.9%: " << pair_str(quantile(var, 0.999))
            << "\n  max: " << pair_str(var.back())
            << "\n----------------------------------\n";
}

int main(int ac, char** av) {
  auto in_r = fs::path{};
  auto in_c = fs::path{};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("reference,r", bpo::value(&in_r),
       "path to binary dump of vector<pareto_set<journey>>")  //
      ("compare,c", bpo::value(&in_c),
       "path to binary dump of vector<pareto_set<journey>>");
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

  std::sort(begin(rating_timing), end(rating_timing),
            [](auto const& a, auto const& b) { return a.first < b.first; });
  print_result(rating_timing, "rating");

  std::sort(begin(rating_timing), end(rating_timing),
            [](auto const& a, auto const& b) { return a.second < b.second; });
  print_result(rating_timing, "timing");

  return 0;
}