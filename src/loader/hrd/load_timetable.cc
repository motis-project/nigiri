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

  utl::activate_progress_tracker("services");
  auto progress_tracker = utl::get_active_progress_tracker();

  //  progress_tracker->status("READ").in_high(byte_sum);
  //  auto total_bytes_processed = std::uint64_t{0U};
  auto i = 0U;
  for (auto const& path : d.list_files(c.fplan_)) {
    if (i++ > 10) {
      break;
    }
    log(log_lvl::info, "loader.hrd.services", "loading {}", path);
    auto const file = d.get_file(path);
    sb.add_services(c, file.filename(), file.data(),
                    [&](std::size_t const bytes_processed) {
                      (void)bytes_processed;
                      //          progress_tracker->update(total_bytes_processed
                      //          + bytes_processed);
                    });
    sb.write_services(src);
    //    total_bytes_processed += f.data().size();
  }
}

}  // namespace nigiri::loader::hrd