#include "nigiri/loader/init_finish.h"

#include <execution>

#include "utl/enumerate.h"

#include "geo/box.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/register.h"
#include "nigiri/flex.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void register_special_stations(timetable& tt) {
  tt.languages_.store("");  // enable kDefaultLang
  tt.register_translation(std::string_view{""});  // enable kEmptyTranslation
  for (auto const& name : special_stations_names) {
    auto const name_translation = tt.register_translation(name);
    register_location(tt, location{tt,
                                   source_idx_t::invalid(),
                                   name,
                                   name_translation,
                                   kEmptyTranslation,
                                   kEmptyTranslation,
                                   {0.0, 0.0},
                                   location_type::kStation,
                                   location_idx_t::invalid(),
                                   timezone_idx_t::invalid(),
                                   0_minutes});
  }
  tt.location_routes_.resize(tt.n_locations());
  tt.bitfields_.emplace_back(bitfield{});  // bitfield_idx 0 = 000...00 bitfield
  tt.attribute_combinations_.add_back_sized(0U);  // combination 0 = empty
}

void build_location_tree(timetable& tt) {
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto box = geo::box{};
    box.extend(tt.locations_.coordinates_[l]);
    tt.locations_.rtree_.insert(box.min_.lnglat_float(),
                                box.max_.lnglat_float(), l);
  }
}

void assign_stops_to_flex_areas(timetable& tt) {
  for (auto const [i, bbox] : utl::enumerate(tt.flex_area_bbox_)) {
    auto const flex_area = flex_area_idx_t{i};
    tt.flex_area_locations_.emplace_back(
        std::initializer_list<location_idx_t>{});
    tt.locations_.rtree_.search(
        bbox.min_.lnglat_float(), bbox.max_.lnglat_float(),
        [&](auto, auto, location_idx_t const l) {
          if (is_in_flex_area(tt, flex_area, tt.locations_.coordinates_[l])) {
            tt.flex_area_locations_.back().push_back(l);
          }
          return true;
        });
  }
}

void assign_importance(timetable& tt) {
  auto& importance = tt.locations_.location_importance_;
  importance.resize(tt.n_locations());

  for (auto i = 0U; i != tt.n_locations(); ++i) {
    auto const l = location_idx_t{i};

    auto transport_counts = std::array<unsigned, kNumClasses>{};
    for (auto const& r : tt.location_routes_[l]) {
      for (auto const tr : tt.route_transport_ranges_[r]) {
        auto const clasz = static_cast<std::underlying_type_t<nigiri::clasz>>(
            tt.route_section_clasz_[r][0]);
        transport_counts[clasz] +=
            tt.bitfields_[tt.transport_traffic_days_[tr]].count();
      }
    }

    constexpr auto const prio =
        std::array<float, kNumClasses>{/* Air */ 20,
                                       /* HighSpeed */ 20,
                                       /* LongDistance */ 20,
                                       /* Coach */ 20,
                                       /* Night */ 20,
                                       /* RegionalFast */ 16,
                                       /* Regional */ 15,
                                       /* Suburban */ 10,
                                       /* Subway */ 10,
                                       /* Tram */ 3,
                                       /* Bus  */ 2,
                                       /* Ship  */ 10,
                                       /* Other  */ 1};
    auto const p = tt.locations_.parents_[l];
    auto& x = importance[p == location_idx_t::invalid() ? l : p];
    for (auto const [clasz, t_count] : utl::enumerate(transport_counts)) {
      x += prio[clasz] * static_cast<float>(t_count);
    }
    tt.locations_.max_importance_ = std::max(tt.locations_.max_importance_, x);
  }
}

// Based on https://www.w3.org/TR/WCAG20/#relativeluminancedef
float luminance(color_t color) {
  constexpr auto max = static_cast<float>(std::numeric_limits<uint8_t>::max());
  auto const r = (color.v_ >> 16 & 0xFF) / max;
  auto const g = (color.v_ >> 8 & 0xFF) / max;
  auto const b = (color.v_ & 0xFF) / max;

  auto const color_lum = [](float channel) -> float {
    return channel <= 0.03928f ? channel / 12.92f
                               : std::pow((channel + 0.055f) / 1.055f, 2.4f);
  };
  auto const red_lum = color_lum(r);
  auto const green_lum = color_lum(g);
  auto const blue_lum = color_lum(b);

  return 0.2126f * red_lum + 0.7152f * green_lum + 0.0722f * blue_lum;
}

// Based on contrast ratio formula from https://www.w3.org/TR/WCAG20/
float contrast_ratio(color_t a, color_t b) {
  auto const a_lum = luminance(a);
  auto const b_lum = luminance(b);

  auto const [lighter, darker] =
      a_lum > b_lum ? std::tuple{a_lum, b_lum} : std::tuple{b_lum, a_lum};

  return (lighter + 0.05f) / (darker + 0.05f);
}

void correct_color_contrast(timetable& tt) {
  for (auto& ids : tt.route_ids_) {
    for (auto& colors : ids.route_id_colors_) {
      auto const ratio = contrast_ratio(colors.color_, colors.text_color_);

      if (ratio < 3.0f) {
        constexpr auto white = color_t(0xFFFFFFFF);
        constexpr auto black = color_t(0xFF000000);
        auto const better = contrast_ratio(colors.color_, black) >
                                    contrast_ratio(colors.color_, white)
                                ? black
                                : white;
        colors.text_color_ = better;
      }
    }
  }
}

void finalize(timetable& tt, finalize_options const opt) {
  tt.location_routes_.resize(tt.n_locations());

  {
    auto const timer = scoped_timer{"loader.sort_trip_ids"};
    std::sort(
#if __cpp_lib_execution
        std::execution::par_unseq,
#endif
        begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_),
        [&](pair<trip_id_idx_t, trip_idx_t> const& a,
            pair<trip_id_idx_t, trip_idx_t> const& b) {
          return std::tuple{tt.trip_id_src_[a.first],
                            tt.trip_id_strings_[a.first].view()} <
                 std::tuple{tt.trip_id_src_[b.first],
                            tt.trip_id_strings_[b.first].view()};
        });
  }
  {
    auto const timer = scoped_timer{"loader.sort_providers"};
    std::sort(
#if __cpp_lib_execution
        std::execution::par_unseq,
#endif
        begin(tt.provider_id_to_idx_), end(tt.provider_id_to_idx_),
        [&](provider_idx_t const a, provider_idx_t const b) {
          return std::tie(tt.providers_[a].src_, tt.providers_[a].id_) <
                 std::tie(tt.providers_[b].src_, tt.providers_[b].id_);
        });
  }
  build_footpaths(tt, opt);
  build_lb_graph<direction::kForward>(tt, kDefaultProfile);
  build_lb_graph<direction::kBackward>(tt, kDefaultProfile);
  build_location_tree(tt);
  assign_stops_to_flex_areas(tt);
  assign_importance(tt);
  correct_color_contrast(tt);

  log(log_lvl::info, "nigiri.loader.finalize",
      "{} locations ({}% of idx space used)", tt.n_locations(),
      static_cast<double>(tt.n_locations()) / footpath::kMaxTarget * 100.0);
}

void finalize(timetable& tt,
              bool const adjust_footpaths,
              bool const merge_dupes_intra_src,
              bool const merge_dupes_inter_src,
              std::uint16_t const max_footpath_length) {
  finalize(tt, {adjust_footpaths, merge_dupes_intra_src, merge_dupes_inter_src,
                max_footpath_length});
}

}  // namespace nigiri::loader
