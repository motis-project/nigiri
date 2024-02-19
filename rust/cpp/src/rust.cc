#include "nigiri/rust.h"

#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "nigiri/loader/load.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;
using namespace nigiri;
using namespace nigiri::loader;

std::string_view to_sv(rust::String const& s) { return {s.data(), s.size()}; }
std::string_view to_sv(rust::Str const& s) { return {s.data(), s.size()}; }

date::sys_days parse_date(std::string_view s) {
  auto sys_days = date::sys_days{};

  std::stringstream ss;
  ss << s;
  ss >> date::parse("%F", sys_days);

  return sys_days;
}

std::unique_ptr<timetable> load_timetable(rust::Vec<rust::String> const& paths,
                                          LoaderConfig const& config,
                                          rust::Str start_date,
                                          std::uint32_t const num_days) {
  std::cerr << "CREATNIG PROGRESS TRACKER\n";
  auto const progress_tracker = utl::activate_progress_tracker("nigiri");
  auto const silencer = utl::global_progress_bars{true};

  auto const start = parse_date(to_sv(start_date));
  auto const tt =
      load(utl::to_vec(paths, [](auto&& p) { return fs::path{to_sv(p)}; }),
           loader_config{.link_stop_distance_ = config.link_stop_distance,
                         .default_tz_ = to_sv(config.default_tz)},
           interval{start, start + std::chrono::days{num_days + 1}});

  std::cerr << "DELETING PROGRESS TRACKER\n";
  return std::make_unique<timetable>(tt);
}

void dump_timetable(timetable const& tt, rust::Str path) {
  tt.write(fs::path{to_sv(path)});
}