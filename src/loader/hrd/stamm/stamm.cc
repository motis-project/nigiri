#include "nigiri/loader/hrd/stamm/stamm.h"

#include "fmt/ranges.h"

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/to_vec.h"

#include "nigiri/loader/hrd/stamm/basic_info.h"

namespace nigiri::loader::hrd {

std::vector<file> stamm::load_files(config const& c, dir const& d) {
  return utl::to_vec(c.required_files_,
                     [&](std::vector<std::string> const& alt) {
                       if (alt.empty()) {
                         return file{};
                       }
                       for (auto const& file : alt) {
                         try {
                           return d.get_file(c.prefix(d) / c.core_data_ / file);
                         } catch (...) {
                         }
                       }
                       throw utl::fail("no file available: {}", alt);
                     });
}

stamm::stamm(config const& c, timetable& tt, dir const& d) : tt_{tt} {
  auto const files = load_files(c, d);
  timezones_ = parse_timezones(c, tt, files.at(TIMEZONES).data());
  locations_ =
      parse_stations(c, source_idx_t{0U}, tt, *this, files.at(STATIONS).data(),
                     files.at(COORDINATES).data(), files.at(FOOTPATHS).data());
  bitfields_ = parse_bitfields(c, files.at(BITFIELDS).data());
  categories_ = parse_categories(c, files.at(CATEGORIES).data());
  providers_ = parse_providers(c, tt, files.at(PROVIDERS).data());
  attributes_ = parse_attributes(c, tt, files.at(ATTRIBUTES).data());
  directions_ = parse_directions(c, tt, files.at(DIRECTIONS).data());
  date_range_ = parse_interval(files.at(BASIC_DATA).data());
  tracks_ = parse_track_rules(c, *this, tt, files.at(TRACKS).data());
}

stamm::stamm(timetable& tt, timezone_map_t&& m)
    : timezones_{std::move(m)}, tt_{tt} {}

interval<std::chrono::sys_days> stamm::get_date_range() const {
  return date_range_;
}

location_idx_t stamm::resolve_location(eva_number eva) const {
  auto const it = locations_.find(eva);
  return it == end(locations_) ? location_idx_t::invalid() : it->second.idx_;
}

category const* stamm::resolve_category(utl::cstr s) const {
  auto const it = categories_.find(s.view());
  return it == end(categories_) ? nullptr : &it->second;
}

trip_direction_idx_t stamm::resolve_direction(direction_info_t const& info) {
  return info.apply(utl::overloaded{
      [&](utl::cstr const str) {
        return utl::get_or_create(string_directions_, str.view(), [&]() {
          auto const it = directions_.find(str.view());
          if (it == end(directions_)) {
            return trip_direction_idx_t::invalid();
          } else {
            auto const dir_idx =
                trip_direction_idx_t{tt_.trip_directions_.size()};
            tt_.trip_directions_.emplace_back(it->second);
            return dir_idx;
          }
        });
      },
      [&](eva_number const eva) {
        return utl::get_or_create(eva_directions_, eva, [&]() {
          auto const it = locations_.find(eva);
          if (it == end(locations_)) {
            return trip_direction_idx_t::invalid();
          } else {
            auto const dir_idx =
                trip_direction_idx_t{tt_.trip_directions_.size()};
            tt_.trip_directions_.emplace_back(it->second.idx_);
            return dir_idx;
          }
        });
      }});
}

bitfield stamm::resolve_bitfield(unsigned i) const {
  if (i == 0U) {
    return bitfield ::max();
  } else {
    auto const it = bitfields_.find(i);
    return it == end(bitfields_) ? bitfield::max() : it->second;
  }
}

provider_idx_t stamm::resolve_provider(utl::cstr s) {
  auto const it = providers_.find(s.view());
  if (it == end(providers_)) {
    log(log_lvl::error, "nigiri.loader.hrd.provider",
        "creating new provider for missing {}", s.view());
    auto const idx = provider_idx_t{tt_.providers_.size()};
    tt_.providers_.emplace_back(
        provider{.short_name_ = s.view(), .long_name_ = s.view()});
    providers_[s.to_str()] = idx;
    return idx;
  } else {
    return it->second;
  }
}

std::pair<timezone_idx_t, tz_offsets> const& stamm::get_tz(
    eva_number const eva_number) const {
  utl::verify(!timezones_.empty(), "no timezones");
  auto const it = timezones_.upper_bound(eva_number);
  utl::verify(it != end(timezones_) || std::prev(it)->first <= eva_number,
              "no timezone for eva number {}", eva_number);
  return std::prev(it)->second;
}

attribute_idx_t stamm::resolve_attribute(utl::cstr s) const {
  auto const it = attributes_.find(s.view());
  return it == end(attributes_) ? attribute_idx_t::invalid() : it->second;
}

location_idx_t stamm::resolve_track(track_rule_key const& k,
                                    minutes_after_midnight_t const mam,
                                    day_idx_t day_idx) const {
  auto it = tracks_.track_rules_.find(k);
  if (it == end(tracks_.track_rules_)) {
    return k.location_;
  } else {
    auto const track_rule_it =
        utl::find_if(it->second, [&](track_rule const& r) {
          return (r.mam_ == track_rule::kTimeNotSet || r.mam_ == mam) &&
                 resolve_bitfield(r.bitfield_num_).test(to_idx(day_idx));
        });
    if (track_rule_it != end(it->second)) {
      return track_rule_it->track_location_;
    }
    return k.location_;
  }
}

trip_line_idx_t stamm::resolve_line(std::string_view s) {
  return utl::get_or_create(lines_, s, [&]() {
    auto const idx = trip_line_idx_t{tt_.trip_lines_.size()};
    tt_.trip_lines_.emplace_back(s);
    return idx;
  });
}

}  // namespace nigiri::loader::hrd
