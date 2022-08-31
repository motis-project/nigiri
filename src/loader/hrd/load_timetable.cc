#include "nigiri/loader/hrd/load_timetable.h"

#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/hrd/service.h"
#include "nigiri/loader/hrd/stamm.h"

namespace nigiri::loader::hrd {

bool applicable(config const& c, dir const& d) {
  return utl::all_of(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        return alt.empty() || utl::any_of(alt, [&](std::string const& file) {
                 auto const exists = d.exists(c.core_data_ / file);
                 if (!exists) {
                   std::clog << "missing file for config " << c.version_.view()
                             << ": " << (c.core_data_ / file) << "\n";
                 }
                 return exists;
               });
      });
}

void load_timetable(source_idx_t const src,
                    config const& c,
                    dir const& d,
                    timetable& tt) {
  auto bars = utl::global_progress_bars{false};

  auto const st = stamm{c, tt, d};
  service_builder sb{st, tt};

  auto const service_files = utl::to_vec(
      d.list_files(c.fplan_), [&](auto&& path) { return d.get_file(path); });
  auto const byte_sum =
      utl::all(service_files) |
      utl::transform([](file const& s) { return s.data().size(); }) |
      utl::sum();

  utl::activate_progress_tracker("services");
  auto progress_tracker = utl::get_active_progress_tracker();

  progress_tracker->status("READ").out_bounds(0, 50).in_high(byte_sum);
  auto total_bytes_processed = std::uint64_t{0U};
  for (auto const& f : service_files) {
    log(log_lvl::info, "loader.hrd.services", "loading {}", f.filename());
    sb.add_services(
        c, f.filename(), f.data(), [&](std::size_t const bytes_processed) {
          progress_tracker->update(total_bytes_processed + bytes_processed);
        });
    total_bytes_processed += f.data().size();
  }
  progress_tracker->status("WRITE").out_bounds(51, 100).in_high(
      sb.route_services_.size());
  sb.write_services(src);
}

}  // namespace nigiri::loader::hrd