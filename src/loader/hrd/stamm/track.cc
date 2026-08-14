#include "nigiri/loader/hrd/stamm/track.h"

#include "nigiri/loader/register.h"

#include "utl/get_or_create.h"

#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

tracks parse_track_rules(config const& c,
                         stamm& st,
                         timetable& tt,
                         std::string_view file_content) {
  auto const timer = scoped_timer{"loader.hrd.tracks"};
  auto ret = tracks{};
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned line_number) {
    if (line.len == 0 || line.starts_with("%")) {
      return;
    } else if (line.len < c.track_rul_.min_line_length_) {
      throw utl::fail("invalid track rules file (line={})", line_number);
    }

    auto const eva_num = parse_eva_number(line.substr(c.track_rul_.eva_num_));
    auto const train_num =
        parse<std::uint32_t>(line.substr(c.track_rul_.train_num_));
    auto const admin = st.resolve_provider(line.substr(c.track_rul_.admin_));
    auto const time =
        hhmm_to_min(parse<int>(line.substr(c.track_rul_.time_).trim(), -1));
    auto const bitfield =
        parse<unsigned>(line.substr(c.track_rul_.bitfield_).trim(), 0U);
    auto const track_name_str =
        iso_8859_1_to_utf8(line.substr(c.track_rul_.track_name_).trim().view());

    auto const parent = st.resolve_location(eva_num);
    if (parent == location_idx_t::invalid()) {
      return;
    }

    auto const track_location = utl::get_or_create(
        ret.rack_locations_,
        track_at_station{.parent_station_ = parent,
                         .track_name_ = track_name_str},
        [&]() {
          auto l = location{tt, parent};
          auto const id = fmt::format("T:{}:{}", l.id_, track_name_str);
          l.id_ = id;
          l.name_ = tt.register_translation(track_name_str);
          l.type_ = location_type::kGeneratedTrack;
          l.parent_ = parent;
          auto const child = register_location(tt, l);
          tt.locations_.children_[parent].emplace_back(child);
          return child;
        });

    ret.track_rules_[track_rule_key{parent, train_num, admin}].push_back(
        track_rule{bitfield, track_location, minutes_after_midnight_t{time}});
  });

  // Track locations have no position of their own and are not listed as
  // equivalent stations, so nothing derives their transfers: connect every
  // track with its station and with the other tracks of that station at the
  // station's transfer time.
  auto tracks_at = hash_map<location_idx_t, std::vector<location_idx_t>>{};
  for (auto const& [key, l] : ret.rack_locations_) {
    tracks_at[key.parent_station_].push_back(l);
  }
  for (auto const& [parent, children] : tracks_at) {
    auto const t = tt.locations_.transfer_time_[parent];
    for (auto const child : children) {
      tt.locations_.preprocessing_footpaths_out_[parent].emplace_back(child, t);
      tt.locations_.preprocessing_footpaths_out_[child].emplace_back(parent, t);
      for (auto const other : children) {
        if (other != child) {
          tt.locations_.preprocessing_footpaths_out_[child].emplace_back(other,
                                                                         t);
        }
      }
    }
  }

  return ret;
}

}  // namespace nigiri::loader::hrd
