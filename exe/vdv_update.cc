#include <filesystem>
#include <random>
#include <string>
#include <thread>

#include "boost/program_options.hpp"
#include "date/date.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/vdv/vdv_update.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
using namespace std::literals::chrono_literals;
using namespace std::string_literals;
using namespace nigiri;

int main(int argc, char* argv[]) {

  auto tt_path = std::filesystem::path{};
  auto vdv_update_path = std::filesystem::path{};

  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required())  //
      ("vdv_update_path,f", bpo::value(&vdv_update_path)->required());

  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);

  // process program options - begin
  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  bpo::notify(vm);

  auto tt = *nigiri::timetable::read(cista::memory_holder{
      cista::file{fs::path{"tt.bin"}.generic_string().c_str(), "r"}.content()});
  tt.locations_.resolve_timezones();

  auto rtt =
      rt::create_rt_timetable(tt, std::chrono::time_point_cast<date::days>(
                                      std::chrono::system_clock::now()));

  auto vdv_update_file = std::ifstream{vdv_update_path, std::ios_base::in};
  std::ostringstream ss;
  ss << vdv_update_file.rdbuf();
  auto doc = pugi::xml_document{};
  doc.load_string(ss.str().c_str());

  auto const stats = rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);
  std::cout << "Statistics:\n" << stats << "\n";
}
