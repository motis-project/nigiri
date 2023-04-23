#include "nigiri/loader/hrd/stamm/track.h"

#include "utl/get_or_create.h"

#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

void parse_track_rules(config const& c,
                       stamm& st,
                       timetable& tt,
                       std::string_view file_content,
                       track_rule_map_t& track_rules,
                       track_location_map_t& track_locations) {
  auto const timer = scoped_timer{"loader.hrd.tracks"};
  hash_map<std::string, track_name_idx_t> track_names;
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned line_number) {
    if (line.len == 0 || line.starts_with("%")) {
      return;
    } else if (line.len < c.track_rul_.min_line_length_) {
      throw utl::fail("invalid track rules file (line={})", line_number);
    }

    auto const eva_num = parse_eva_number(line.substr(c.track_rul_.eva_num_));
    auto const train_num = parse<int>(line.substr(c.track_rul_.train_num_));
    auto const admin = st.resolve_provider(line.substr(c.track_rul_.admin_));
    auto const time =
        hhmm_to_min(parse<int>(line.substr(c.track_rul_.time_).trim(), -1));
    auto const bitfield =
        parse<unsigned>(line.substr(c.track_rul_.bitfield_).trim(), 0U);

    auto track_name_str =
        iso_8859_1_to_utf8(line.substr(c.track_rul_.track_name_).trim().view());
    auto const track_name_idx =
        utl::get_or_create(track_names, track_name_str, [&]() {
          auto const idx = track_name_idx_t{tt.track_names_.size()};
          tt.track_names_.emplace_back(std::move(track_name_str));
          return idx;
        });

    auto const parent = st.resolve_location(eva_num);
    if (parent == location_idx_t::invalid()) {
      return;
    }

    auto const track_location = utl::get_or_create(
        track_locations,
        track_at_station{.parent_station_ = parent, .track_ = track_name_idx},
        [&]() {
          auto const l = location{tt, parent};
          auto const id = fmt::format("T:{}:{}", l.id_,
                                      tt.track_names_[track_name_idx].view());
          auto const name = string{tt.track_names_[track_name_idx].view()};
          auto const child = tt.locations_.register_location(
              location{id, name, l.pos_, l.src_, location_type::kGeneratedTrack,
                       l.osm_id_, parent, l.timezone_idx_, l.transfer_time_,
                       l.equivalences_, l.footpaths_in_, l.footpaths_out_});
          tt.locations_.children_[parent].emplace_back(child);
          return child;
        });

    track_rules[track_rule_key{parent, train_num, admin}].push_back(
        track_rule{bitfield, track_location, minutes_after_midnight_t{time}});
  });
}

}  // namespace nigiri::loader::hrd
