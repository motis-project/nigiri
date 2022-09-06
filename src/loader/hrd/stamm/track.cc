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
  scoped_timer timer("parsing track rules");
  hash_map<std::string, track_name_idx_t> track_names;
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned line_number) {
    if (line.len == 0 || line.starts_with("%")) {
      return;
    } else if (line.len < c.track_rul_.min_line_length_) {
      throw utl::fail("invalid track rules file (line={})", line_number);
    }

    auto const eva_num = parse_eva_number(line.substr(c.track_rul_.eva_num_));
    auto const train_num =
        parse<unsigned>(line.substr(c.track_rul_.train_num_));
    auto const train_admin =
        st.resolve_provider(line.substr(c.track_rul_.train_admin_));
    auto const time =
        hhmm_to_min(parse<int>(line.substr(c.track_rul_.time_).trim(), -1));
    auto const bitfield =
        parse<unsigned>(line.substr(c.track_rul_.bitfield_).trim(), 0U);
    auto const track_name_str =
        iso_8859_1_to_utf8(line.substr(c.track_rul_.track_name_).trim().view());

    auto const track_name_idx =
        utl::get_or_create(track_names, track_name_str, [&]() {
          auto const idx = track_name_idx_t{tt.track_names_.size()};
          tt.track_names_.emplace_back(track_name_str);
          return idx;
        });

    auto const parent = st.resolve_location(eva_num);
    auto const track_location = utl::get_or_create(
        track_locations,
        track_at_station{.parent_station_ = parent, .track_ = track_name_idx},
        [&]() {
          auto const l = location{tt, parent};
          auto const id = string{l.id_.str() + "-" + track_name_str};
          auto const name = string{track_name_str};
          return tt.locations_.register_location(
              location{id, name, l.pos_, l.src_, location_type::kTrack,
                       l.osm_id_, parent, l.timezone_idx_, l.equivalences_,
                       l.footpaths_in_, l.footpaths_out_});
        });

    track_rules[track_rule_key{parent, train_num, train_admin,
                               minutes_after_midnight_t{time}}]
        .push_back(track_rule{bitfield, track_location});
  });
}

}  // namespace nigiri::loader::hrd