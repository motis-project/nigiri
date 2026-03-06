#include "nigiri/export_gtfs.h"

#include <filesystem>
#include <fstream>

#include "nigiri/timetable.h"

namespace nigiri {

void export_gtfs(timetable const& tt, std::filesystem::path const& dir) {
  std::filesystem::create_directories(dir);

  std::ofstream stops(dir / "stops.txt");
  stops << "stop_id,stop_name,stop_lat,stop_lon\n";

  for (location_idx_t l{0}; l < tt.n_locations(); ++l) {
    auto const id = tt.locations_.ids_[l].view();
    auto const name = tt.get_default_name(l);
    auto const coord = tt.locations_.coordinates_[l];

    stops << id << ",\" << name << "\"," << coord.lat_ << "," << coord.lng_
          << "\n";
  }
}

}  // namespace nigiri
