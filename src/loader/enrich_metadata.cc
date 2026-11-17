#include "nigiri/loader/enrich_metadata.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/to_vec.h"

#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/logging.h"
#include "nigiri/translations_view.h"

namespace nigiri::loader {

namespace {

constexpr auto const kFlags =
    std::array{kBikesAllowed, kCarsAllowed, kWheelchairAccessible};

trip_idx_t first_trip(timetable const& tt, transport_idx_t const t) {
  return tt.merged_trips_[tt.transport_to_trip_section_[t].front()].front();
}

bool is_active(timetable const& tt, transport_idx_t const t) {
  return tt.transport_traffic_days_[t] != kEmptyBitfieldIdx;
}

// Sources of each target, by weight descending.
template <typename T>
std::vector<std::pair<T, std::vector<T>>> by_target(
    metadata_enricher::pairs<T> const& pairs) {
  auto sorted = std::vector<std::pair<std::pair<T, T>, unsigned>>{
      begin(pairs), end(pairs)};
  std::ranges::sort(sorted, [](auto const& a, auto const& b) {
    return std::tuple{a.first.first, b.second, a.first.second} <
           std::tuple{b.first.first, a.second, b.first.second};
  });
  auto groups = std::vector<std::pair<T, std::vector<T>>>{};
  for (auto const& [k, _] : sorted) {
    if (groups.empty() || groups.back().first != k.first) {
      groups.emplace_back(k.first, std::vector<T>{});
    }
    groups.back().second.push_back(k.second);
  }
  return groups;
}

template <typename T, typename Source, typename Get, typename IsSet>
bool fill(T& target,
          std::vector<Source> const& sources,
          Get&& get,
          IsSet&& is_set) {
  if (is_set(target)) {
    return false;
  }
  for (auto const& s : sources) {
    if (auto const v = get(s); is_set(v)) {
      target = v;
      return true;
    }
  }
  return false;
}

// "S 01" = "s1": lowercase, without separators and leading zeros.
struct line_name {
  explicit line_name(std::string_view const s) {
    static auto const kNoise = std::regex{R"([\s_-]+|(^|\D)0+(?=\d))"};
    static auto const kNumber = std::regex{R"(\d+)"};
    name_ = std::regex_replace(std::string{s}, kNoise, "$1");
    std::ranges::transform(name_, begin(name_), [](unsigned char const c) {
      return static_cast<char>(std::tolower(c));
    });
    for (auto it = std::sregex_iterator{begin(name_), end(name_), kNumber};
         it != std::sregex_iterator{}; ++it) {
      numbers_.push_back(it->str());
    }
  }

  // A pure number matches a number of the other name: "12" = "ICE 12".
  bool same_number(line_name const& o) const {
    auto const is_number = [](line_name const& x) {
      return x.numbers_.size() == 1U && x.numbers_.front() == x.name_;
    };
    return (is_number(*this) && utl::find(o.numbers_, name_) != end(o.numbers_)) ||
           (is_number(o) && utl::find(numbers_, o.name_) != end(numbers_));
  }

  std::string name_;
  std::vector<std::string> numbers_;
};

// Feeds classify light/rack railways differently: all rail-bound is one.
int mode(clasz const c) {
  switch (c) {
    case clasz::kAir: return 0;
    case clasz::kBus:
    case clasz::kCoach:
    case clasz::kODM:
    case clasz::kRideSharing: return 2;
    case clasz::kShip: return 3;
    case clasz::kAerialLift: return 4;
    case clasz::kOther: return -1;
    default: return 1;
  }
}

std::string_view short_name(timetable const& tt, trip_idx_t const t) {
  auto const r = tt.trip_route_id_[t];
  return r == route_id_idx_t::invalid()
             ? std::string_view{}
             : tt.get_default_translation(
                   tt.route_ids_[tt.trip_id_src_[t]].route_id_short_names_[r]);
}

}  // namespace

void metadata_enricher::add_transport(transport_idx_t const target,
                                      transport_idx_t const source) {
  auto const trip_route_id = [&](transport_idx_t const t) {
    auto const trip = first_trip(tt_, t);
    return route_id{tt_.trip_id_src_[trip], tt_.trip_route_id_[trip]};
  };
  auto const route_t = trip_route_id(target);
  auto const route_s = trip_route_id(source);
  if (route_t != route_s && route_t.second != route_id_idx_t::invalid() &&
      route_s.second != route_id_idx_t::invalid()) {
    ++route_ids_[{route_t, route_s}];
  }

  auto const provider_t = tt_.transport_section_providers_[target].front();
  auto const provider_s = tt_.transport_section_providers_[source].front();
  if (provider_t != provider_s && provider_t != provider_idx_t::invalid() &&
      provider_s != provider_idx_t::invalid()) {
    ++providers_[{provider_t, provider_s}];
  }

  auto mask = std::uint8_t{0U};
  for (auto const f : kFlags) {
    if (tt_.is_flag_set(f, tt_.transport_route_[source])) {
      mask |= static_cast<std::uint8_t>(1U << f);
    }
  }
  if (mask != 0U) {
    flags_[target] |= mask;
  }
}

void metadata_enricher::add_stop(location_idx_t const target,
                                 location_idx_t const source) {
  if (target != source) {
    ++stops_[{target, source}];
  }
  auto const target_root = tt_.locations_.get_root_idx(target);
  auto const source_root = tt_.locations_.get_root_idx(source);
  if (target_root != source_root &&
      (target_root != target || source_root != source)) {
    ++stops_[{target_root, source_root}];
  }
}

void metadata_enricher::apply() {
  auto const timer = scoped_timer{"loader.enrich_metadata"};

  auto const has_text = [&](translation_idx_t const t) {
    return t != translation_idx_t::invalid() &&
           !tt_.get_default_translation(t).empty();
  };
  auto const has_id = [](wikidata_id const x) { return x.valid(); };
  auto const has_color = [](route_color const c) {
    return c.color_ != color_t{0U};
  };

  // Adds the languages the target lacks.
  auto const add_translations = [&](translation_idx_t& target,
                                    auto const& sources, auto&& get) {
    if (!has_text(target)) {
      return false;
    }
    auto merged = std::vector<translation>{};
    auto languages = std::vector<language_idx_t>{};
    auto const add = [&](language_idx_t const lang, std::string_view text) {
      auto& x = merged.emplace_back();  // owning: registering reallocates
      x.set_language(tt_.languages_.get(lang));
      x.set_text(text);
      languages.push_back(lang);
    };
    for (auto const [lang, text] : get_translation_view(tt_, target)) {
      add(lang, text);
    }
    auto const n_before = merged.size();
    for (auto const& s : sources) {
      auto const t = get(s);
      if (!has_text(t) || t == target) {
        continue;
      }
      for (auto const [lang, text] : get_translation_view(tt_, t)) {
        if (lang != kDefaultLang && !text.empty() &&
            utl::find(languages, lang) == end(languages)) {
          add(lang, text);
        }
      }
    }
    if (merged.size() == n_before) {
      return false;
    }
    target = tt_.register_translation(merged);
    return true;
  };

  auto n_routes = 0U;
  for (auto const& [r, sources] : by_target(route_ids_)) {
    auto& ids = tt_.route_ids_[r.first];
    auto const fill_from = [&](auto const member, auto&& is_set) {
      return fill(
          (ids.*member)[r.second], sources,
          [&](route_id const s) {
            return (tt_.route_ids_[s.first].*member)[s.second];
          },
          is_set);
    };
    auto filled = fill_from(&timetable::route_ids::route_id_url_, has_text);
    filled = fill_from(&timetable::route_ids::route_id_colors_, has_color) ||
             filled;
    filled = fill_from(&timetable::route_ids::route_id_wikidata_, has_id) ||
             filled;
    n_routes += filled ? 1U : 0U;
  }

  auto n_agencies = 0U;
  for (auto const& [p, sources] : by_target(providers_)) {
    auto& x = tt_.providers_[p];
    auto const get = [&](auto const member) {
      return [&, member](provider_idx_t const s) {
        return tt_.providers_[s].*member;
      };
    };
    auto filled = fill(x.url_, sources, get(&provider::url_), has_text);
    filled =
        fill(x.fare_url_, sources, get(&provider::fare_url_), has_text) ||
        filled;
    filled =
        fill(x.wikidata_, sources, get(&provider::wikidata_), has_id) || filled;
    filled = add_translations(x.name_, sources, get(&provider::name_)) || filled;
    n_agencies += filled ? 1U : 0U;
  }

  auto n_stops = 0U;
  auto& locations = tt_.locations_;
  for (auto const& [l, sources] : by_target(stops_)) {
    auto filled = fill(
        locations.wikidata_[l], sources,
        [&](location_idx_t const s) { return locations.wikidata_[s]; }, has_id);
    filled = add_translations(
                 locations.names_[l], sources,
                 [&](location_idx_t const s) { return locations.names_[s]; }) ||
             filled;
    n_stops += filled ? 1U : 0U;
  }

  // Only set a route flag if all active transports have a partner with it.
  auto routes = utl::to_vec(
      flags_, [&](auto const& x) { return tt_.transport_route_[x.first]; });
  utl::erase_duplicates(routes);
  auto n_flags = 0U;
  for (auto const r : routes) {
    for (auto const f : kFlags) {
      if (tt_.is_flag_set(f, r)) {
        continue;
      }
      auto all = true;
      for (auto const t : tt_.route_transport_ranges_[r]) {
        auto const it = flags_.find(t);
        all = all && (!is_active(tt_, t) ||
                      (it != end(flags_) && (it->second & (1U << f)) != 0U));
      }
      if (all) {
        tt_.route_flags_[f].set(to_idx(r) * 2U, true);
        ++n_flags;
      }
    }
  }

  log(log_lvl::info, "loader.enrich_metadata",
      "enriched {} routes, {} agencies, {} stops, {} route flags", n_routes,
      n_agencies, n_stops, n_flags);
}

void match_metadata_routes(
    timetable& tt,
    import_context const& ctx,
    vecvec<location_idx_t, location_idx_t> const& neighbours,
    metadata_enricher& enricher) {
  if (ctx.metadata_only_sources_.empty()) {
    return;
  }
  auto const timer = scoped_timer{"loader.match_metadata_routes"};

  auto const root = [&](route_idx_t const r, std::size_t const i) {
    return tt.locations_.get_root_idx(
        stop{tt.route_location_seq_[r][i]}.location_idx());
  };

  // Trips with the same stop sequence share a route: one line per route id.
  struct line {
    route_idx_t route_;
    transport_idx_t transport_;
    std::vector<line_name> names_;
  };
  auto lines = std::vector<line>{};
  auto is_metadata = std::vector<bool>(tt.n_routes());
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    if (utl::find(ctx.metadata_only_sources_,
                  tt.locations_.src_[root(r, 0U)]) ==
        end(ctx.metadata_only_sources_)) {
      continue;
    }
    is_metadata[to_idx(r)] = true;
    auto by_route_id = hash_map<route_id_idx_t, std::size_t>{};
    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const trip = first_trip(tt, t);
      auto const [it, is_new] =
          by_route_id.emplace(tt.trip_route_id_[trip], lines.size());
      if (is_new) {
        lines.push_back(line{r, t, {}});
      }
      auto& names = lines[it->second].names_;
      for (auto const name :
           {short_name(tt, trip),
            tt.get_default_translation(tt.trip_short_names_[trip])}) {
        for (auto const part : std::views::split(name, ';')) {  // OSM refs
          names.emplace_back(std::string_view{part});
        }
      }
    }
  }

  auto route_names = std::vector<std::optional<line_name>>(tt.n_routes());
  utl::parallel_for_run(tt.n_routes(), [&](std::size_t const i) {
    for (auto const t : tt.route_transport_ranges_[route_idx_t{i}]) {
      if (!is_metadata[i] && is_active(tt, t)) {
        route_names[i].emplace(short_name(tt, first_trip(tt, t)));
        break;
      }
    }
  });

  struct match {
    std::size_t line_;
    bool number_only_;
    std::vector<std::pair<stop_idx_t, stop_idx_t>> stops_;
  };
  auto matches = std::vector<std::vector<std::pair<route_idx_t, match>>>(
      lines.size());
  utl::parallel_for_run(lines.size(), [&](std::size_t const i) {
    auto const& l = lines[i];
    auto const m = tt.route_location_seq_[l.route_].size();

    auto candidates = std::vector<route_idx_t>{};
    auto const add_routes = [&](location_idx_t const x) {
      auto const add = [&](location_idx_t const y) {
        candidates.insert(end(candidates), begin(tt.location_routes_[y]),
                          end(tt.location_routes_[y]));
      };
      add(x);
      for (auto const c : tt.locations_.children_[x]) {
        add(c);
      }
    };
    for (auto b = 0U; b != m; ++b) {
      add_routes(root(l.route_, b));
      for (auto const nb : neighbours[root(l.route_, b)]) {
        add_routes(nb);
      }
    }
    utl::erase_duplicates(candidates);

    auto dp = std::vector<unsigned>{};
    for (auto const r : candidates) {
      auto const& name = route_names[to_idx(r)];
      auto const mode_l = mode(tt.route_clasz_[l.route_]);
      auto const mode_r = mode(tt.route_clasz_[r]);
      if (!name.has_value() ||
          (mode_l != mode_r && mode_l != -1 && mode_r != -1)) {
        continue;
      }
      auto const exact = utl::any_of(
          l.names_, [&](line_name const& x) { return x.name_ == name->name_; });
      if (!exact && utl::none_of(l.names_, [&](line_name const& x) {
            return x.same_number(*name);
          })) {
        continue;
      }

      // Longest common subsequence.
      auto const n = tt.route_location_seq_[r].size();
      auto const at = [&](std::size_t const a,
                          std::size_t const b) -> unsigned& {
        return dp[a * (m + 1U) + b];
      };
      dp.assign((n + 1U) * (m + 1U), 0U);
      for (auto a = 1U; a <= n; ++a) {
        for (auto b = 1U; b <= m; ++b) {
          at(a, b) =
              roots_match(neighbours, root(r, a - 1U), root(l.route_, b - 1U))
                  ? at(a - 1U, b - 1U) + 1U
                  : std::max(at(a - 1U, b), at(a, b - 1U));
        }
      }
      auto const n_matched = static_cast<double>(at(n, m));
      if (n_matched < 2.0 || (n_matched < 0.6 * static_cast<double>(n) &&
                              n_matched < 0.7 * static_cast<double>(m))) {
        continue;
      }
      auto& x = matches[i].emplace_back(r, match{i, !exact, {}}).second;
      for (auto a = n, b = m; a != 0U && b != 0U;) {
        if (at(a, b) == at(a - 1U, b)) {
          --a;
        } else if (at(a, b) == at(a, b - 1U)) {
          --b;
        } else {
          x.stops_.emplace_back(a - 1U, b - 1U);
          --a;
          --b;
        }
      }
    }
  });

  // Each route takes the line matching most of its stops.
  auto best = std::vector<match const*>(tt.n_routes(), nullptr);
  for (auto const& [r, x] : matches | std::views::join) {
    auto& b = best[to_idx(r)];
    if (b == nullptr || std::tuple{!x.number_only_, x.stops_.size()} >
                            std::tuple{!b->number_only_, b->stops_.size()}) {
      b = &x;
    }
  }
  auto n_matched_routes = 0U;
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const* x = best[to_idx(r)];
    if (x == nullptr) {
      continue;
    }
    ++n_matched_routes;
    auto const& l = lines[x->line_];
    for (auto const t : tt.route_transport_ranges_[r]) {
      if (is_active(tt, t)) {
        enricher.add_transport(t, l.transport_);
      }
    }
    for (auto const& [a, b] : x->stops_) {
      enricher.add_stop(
          stop{tt.route_location_seq_[r][a]}.location_idx(),
          stop{tt.route_location_seq_[l.route_][b]}.location_idx());
    }
  }

  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    if (is_metadata[to_idx(r)]) {
      for (auto const t : tt.route_transport_ranges_[r]) {
        tt.transport_traffic_days_[t] = kEmptyBitfieldIdx;
      }
    }
  }

  log(log_lvl::info, "loader.match_metadata_routes",
      "{} of {} metadata lines matched {} routes",
      utl::count_if(matches, [](auto&& x) { return !x.empty(); }),
      lines.size(), n_matched_routes);
}

}  // namespace nigiri::loader
