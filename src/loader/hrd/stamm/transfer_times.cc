#include "nigiri/loader/hrd/stamm/transfer_times.h"

#include <algorithm>
#include <map>

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"
#include "utl/verify.h"

#include "nigiri/loader/register.h"

#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

// === file loading =========================================================

std::vector<file> load_transfer_time_files(config const& c, dir const& d) {
  auto const load = [&](std::vector<std::string> const& alt) {
    for (auto const& f : alt) {
      try {
        return d.get_file(c.prefix(d) / c.core_data_ / f);
      } catch (...) {
        continue;
      }
    }
    return file{};
  };
  auto ret = std::vector<file>{};
  ret.emplace_back(load(c.transfers_.station_));
  ret.emplace_back(load(c.transfers_.admin_));
  ret.emplace_back(load(c.transfers_.line_));
  ret.emplace_back(load(c.transfers_.trip_));
  return ret;
}

// === parsing ==============================================================

namespace {

constexpr auto const kDefaultEva = eva_number{9999999U};

// column layouts acc. to HRDF 5.40.41, ch. 8 (0-based)
namespace col {
// umsteigb: eva, time IC-IC, time other
constexpr auto const b_eva = utl::field{0, 7};
constexpr auto const b_time_other = utl::field{11, 2};
// umsteigv: eva|@@@@@@@, admin1, admin2, time
constexpr auto const v_eva = utl::field{0, 7};
constexpr auto const v_admin_1 = utl::field{8, 6};
constexpr auto const v_admin_2 = utl::field{15, 6};
constexpr auto const v_time = utl::field{22, 2};
// umsteigl: eva|@@@@@@@, (admin, category, line, dir) x2, time, '!'
constexpr auto const l_eva = utl::field{0, 7};
constexpr auto const l_admin_1 = utl::field{8, 6};
constexpr auto const l_cat_1 = utl::field{15, 3};
constexpr auto const l_line_1 = utl::field{19, 8};
constexpr auto const l_dir_1 = utl::field{28, 1};
constexpr auto const l_admin_2 = utl::field{30, 6};
constexpr auto const l_cat_2 = utl::field{37, 3};
constexpr auto const l_line_2 = utl::field{41, 8};
constexpr auto const l_dir_2 = utl::field{50, 1};
constexpr auto const l_time = utl::field{52, 3};
constexpr auto const l_guaranteed = utl::field{55, 1};
// umsteigz: eva, (trip number, admin) x2, time, '!', traffic day bitfield
constexpr auto const z_eva = utl::field{0, 7};
constexpr auto const z_nr_1 = utl::field{8, 6};
constexpr auto const z_admin_1 = utl::field{15, 6};
constexpr auto const z_nr_2 = utl::field{22, 6};
constexpr auto const z_admin_2 = utl::field{29, 6};
constexpr auto const z_time = utl::field{36, 3};
constexpr auto const z_guaranteed = utl::field{39, 1};
constexpr auto const z_bitfield = utl::field{41, 6};
}  // namespace col

bool is_comment(utl::cstr const line) {
  return line.len == 0U || line[0] == '%' || line[0] == '*';
}

u8_minutes parse_time(utl::cstr const s) {
  auto const t = utl::parse<int>(s.trim(), 0);
  return u8_minutes{static_cast<u8_minutes::rep>(std::clamp(t, 0, 255))};
}

char normalize_dir(char const c) {
  switch (c) {
    case 'H': return '1';
    case 'R': return '2';
    default: return c;
  }
}

std::array<char, 8U> to_line_array(utl::cstr const s) {
  auto ret = std::array<char, 8U>{};
  auto const trimmed = s.trim();
  std::copy_n(trimmed.begin(), std::min(trimmed.len, std::size_t{8U}),
              ret.begin());
  return ret;
}

}  // namespace

void parse_station_transfer_times(transfer_times& tt,
                                  std::string_view file_content) {
  utl::for_each_line(utl::cstr{file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 13U) {
      return;
    }
    auto const eva = parse_eva_number(line.substr(col::b_eva));
    auto const time = parse_time(line.substr(col::b_time_other));
    if (eva == kDefaultEva) {
      tt.default_ = time;
    } else {
      tt.station_[eva] = time;
    }
  });
}

void parse_transfer_time_rules(transfer_times& r,
                               stamm& st,
                               std::string_view admin_file_content,
                               std::string_view line_file_content,
                               std::string_view trip_file_content) {
  if (admin_file_content.empty() && line_file_content.empty() &&
      trip_file_content.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.hrd.transfer_times"};

  auto const is_global = [](utl::cstr const eva_field) {
    return eva_field.starts_with("@@@@@@@");
  };

  auto const add_global_admin = [&](utl::cstr const s,
                                    provider_idx_t const p) {
    r.global_admins_.emplace(p);
    r.global_admin_strings_.emplace(s.view());
    r.global_admin_strings_.emplace(s.trim().view());
  };

  // --- umsteigv ---
  utl::for_each_line(utl::cstr{admin_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 24U) {
      return;
    }
    auto const a1 = line.substr(col::v_admin_1);
    auto const a2 = line.substr(col::v_admin_2);
    auto const rule = admin_rule{.from_ = st.resolve_provider(a1),
                                 .to_ = st.resolve_provider(a2),
                                 .time_ = parse_time(line.substr(col::v_time))};
    if (is_global(line)) {
      r.admin_global_.emplace_back(rule);
      add_global_admin(a1, rule.from_);
      add_global_admin(a2, rule.to_);
    } else {
      auto const eva = parse_eva_number(line.substr(col::v_eva));
      r.admin_[eva].emplace_back(rule);
      r.rule_stations_.emplace(eva);
    }
  });

  // --- umsteigl ---
  auto const parse_side = [&](utl::cstr const admin, utl::cstr const cat,
                              utl::cstr const line_id, utl::cstr const dir,
                              bool const global,
                              std::uint8_t& n_wildcards) -> line_rule_side {
    auto side = line_rule_side{};
    if (admin.trim().view() == "*") {
      ++n_wildcards;
      if (global) {
        r.global_matches_any_admin_ = true;
      }
    } else {
      side.admin_ = st.resolve_provider(admin);
      if (global) {
        add_global_admin(admin, side.admin_);
      }
    }
    if (cat.trim().view() == "*") {
      ++n_wildcards;
    } else {
      side.cat_any_ = false;
      side.cat_ = st.resolve_category(cat);
    }
    if (line_id.trim().view() == "*") {
      ++n_wildcards;
    } else {
      side.line_any_ = false;
      side.line_ = to_line_array(line_id);
    }
    auto const d = normalize_dir(dir.len == 0U ? '*' : dir[0]);
    if (d == '*' || d == ' ') {
      ++n_wildcards;
    } else {
      side.dir_ = d;
    }
    return side;
  };

  utl::for_each_line(utl::cstr{line_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 55U) {
      return;
    }
    auto const global = is_global(line);
    auto rule = line_rule{};
    rule.from_ = parse_side(line.substr(col::l_admin_1),
                            line.substr(col::l_cat_1),
                            line.substr(col::l_line_1),
                            line.substr(col::l_dir_1), global,
                            rule.n_wildcards_);
    rule.to_ = parse_side(line.substr(col::l_admin_2),
                          line.substr(col::l_cat_2),
                          line.substr(col::l_line_2),
                          line.substr(col::l_dir_2), global,
                          rule.n_wildcards_);
    rule.time_ = parse_time(line.substr(col::l_time));
    rule.guaranteed_ =
        line.len > col::l_guaranteed.from && line[col::l_guaranteed.from] == '!';
    if (global) {
      r.line_global_.emplace_back(rule);
    } else {
      auto const eva = parse_eva_number(line.substr(col::l_eva));
      r.line_[eva].emplace_back(rule);
      r.rule_stations_.emplace(eva);
    }
  });

  auto const by_wildcards = [](line_rule const& a, line_rule const& b) {
    return a.n_wildcards_ < b.n_wildcards_;
  };
  std::stable_sort(begin(r.line_global_), end(r.line_global_), by_wildcards);
  for (auto& [eva, rules] : r.line_) {
    std::stable_sort(begin(rules), end(rules), by_wildcards);
  }

  // --- umsteigz ---
  utl::for_each_line(utl::cstr{trip_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 39U) {
      return;
    }
    auto const eva = parse_eva_number(line.substr(col::z_eva));
    auto rule = trip_rule{
        .from_nr_ =
            utl::parse<std::uint32_t>(line.substr(col::z_nr_1).trim(), 0U),
        .from_admin_ = st.resolve_provider(line.substr(col::z_admin_1)),
        .to_nr_ =
            utl::parse<std::uint32_t>(line.substr(col::z_nr_2).trim(), 0U),
        .to_admin_ = st.resolve_provider(line.substr(col::z_admin_2)),
        .time_ = parse_time(line.substr(col::z_time)),
        .guaranteed_ = line.len > col::z_guaranteed.from &&
                       line[col::z_guaranteed.from] == '!',
        .bitfield_num_ =
            line.len >= col::z_bitfield.from + col::z_bitfield.size
                ? utl::parse<unsigned>(line.substr(col::z_bitfield).trim(), 0U)
                : 0U};
    r.trip_[eva].emplace_back(rule);
    r.rule_stations_.emplace(eva);
  });

  log(log_lvl::info, "loader.hrd.transfer_times",
      "transfer time rules: {} station times (default={}), admin={}+{} "
      "global, line={}+{} global, trip={} @ {} stations",
      r.station_.size(),
      r.default_.has_value() ? r.default_->count() : -1,
      r.admin_.size(), r.admin_global_.size(), r.line_.size(),
      r.line_global_.size(), r.trip_.size(), r.rule_stations_.size());
}

// === stop event attributes ================================================

namespace {

event_attrs from_section(service const& s, std::size_t const sec_idx) {
  auto e = event_attrs{};
  e.valid_ = true;

  auto const& b2e = s.begin_to_end_info_;
  auto const* sec =
      sec_idx < s.sections_.size() ? &s.sections_[sec_idx] : nullptr;

  e.train_nr_ = b2e.train_num_.has_value()
                    ? *b2e.train_num_
                    : (sec != nullptr && sec->train_num_.has_value()
                           ? *sec->train_num_
                           : s.initial_train_num_);
  e.admin_ = b2e.admin_.has_value()
                 ? *b2e.admin_
                 : (sec != nullptr && sec->admin_.has_value()
                        ? *sec->admin_
                        : s.initial_admin_);
  e.cat_ = b2e.category_.has_value()
               ? *b2e.category_
               : (sec != nullptr && sec->category_.has_value() ? *sec->category_
                                                               : nullptr);
  auto const line = b2e.line_.has_value()
                        ? *b2e.line_
                        : (sec != nullptr && sec->line_.has_value()
                               ? *sec->line_
                               : utl::cstr{});
  if (line.valid()) {
    e.line_ = to_line_array(line);
  }
  e.dir_ = b2e.dir_flag_.has_value()
               ? *b2e.dir_flag_
               : (sec != nullptr && sec->dir_flag_.has_value() ? *sec->dir_flag_
                                                               : ' ');
  return e;
}

}  // namespace

stop_attrs get_stop_attrs(service const& s, std::size_t const stop_idx) {
  auto a = stop_attrs{};
  if (stop_idx > 0U) {
    a.arr_ = from_section(s, stop_idx - 1U);
  }
  if (stop_idx + 1U < s.stops_.size()) {
    a.dep_ = from_section(s, stop_idx);
  }
  return a;
}

// === phase 1: event scan ==================================================

void scan_transfer_events(config const& c,
                          stamm& st,
                          transfer_times const& r,
                          transfer_groups& groups,
                          std::string_view file_content) {
  groups.active_ = true;

  auto const has_global = !r.admin_global_.empty() || !r.line_global_.empty();

  auto const is_global_admin = [&](utl::cstr const s) {
    return r.global_admin_strings_.contains(std::string{s.view()});
  };

  auto const maybe_relevant = [&](specification const& spec) {
    for (auto const& stop_line : spec.stops_) {
      if (stop_line.len >= 7U &&
          r.rule_stations_.contains(
              parse_eva_number(stop_line.substr(utl::field{0, 7})))) {
        return true;
      }
    }
    if (has_global) {
      if (r.global_matches_any_admin_) {
        return true;
      }
      if (spec.internal_service_.len >= 15U &&
          is_global_admin(spec.internal_service_.substr(utl::field{9, 6}))) {
        return true;
      }
      for (auto const& stop_line : spec.stops_) {
        if (stop_line.len >= 55U &&
            is_global_admin(stop_line.substr(utl::field{49, 6}).trim())) {
          return true;
        }
      }
    }
    return false;
  };

  auto const process = [&](specification const& spec) {
    if (!spec.valid() || spec.ignore()) {
      return;
    }
    try {
      if (!maybe_relevant(spec)) {
        return;
      }
      auto const s = service{c, st, source_file_idx_t::invalid(), spec};
      for (auto i = std::size_t{0U}; i != s.stops_.size(); ++i) {
        auto const eva = s.stops_[i].eva_num_;
        auto const attrs = get_stop_attrs(s, i);
        auto const relevant =
            r.rule_stations_.contains(eva) ||
            (has_global && !r.station_.contains(eva) &&
             (r.global_matches_any_admin_ ||
              (attrs.arr_.valid_ &&
               r.global_admins_.contains(attrs.arr_.admin_)) ||
              (attrs.dep_.valid_ &&
               r.global_admins_.contains(attrs.dep_.admin_))));
        if (!relevant) {
          continue;
        }
        auto const l = st.resolve_location(eva);
        if (l == location_idx_t::invalid()) {
          continue;
        }
        auto& station = groups.stations_[l];
        station.eva_ = eva;
        station.tuples_.emplace(attrs);
      }
    } catch (std::exception const&) {
      // invalid services are logged during the real service pass
    }
  };

  auto spec = specification{};
  utl::for_each_line_numbered(
      utl::cstr{file_content}, [&](utl::cstr line, unsigned const line_number) {
        if (line.len == 0U || line[0] == '%') {
          return;
        }
        auto const is_finished = spec.read_line(line, "scan", line_number);
        if (!is_finished) {
          return;
        }
        process(spec);
        spec.reset();
        spec.read_line(line, "scan", line_number);
      });
  if (!spec.is_empty()) {
    process(spec);
  }
}

// === phase 1.5: group building ============================================

namespace {

enum class rule_tier : std::uint64_t {
  kTrip = 0U,
  kLineStation,
  kAdminStation,
  kLineGlobal,
  kAdminGlobal
};

constexpr std::uint64_t encode_side(rule_tier const t,
                                    std::size_t const idx,
                                    bool const from) {
  return (static_cast<std::uint64_t>(t) << 48U) |
         (static_cast<std::uint64_t>(idx) << 1U) | (from ? 1U : 0U);
}

constexpr rule_tier tier_of(std::uint64_t const v) {
  return static_cast<rule_tier>(v >> 48U);
}

constexpr std::size_t idx_of(std::uint64_t const v) {
  return static_cast<std::size_t>((v & ((std::uint64_t{1U} << 48U) - 1U)) >>
                                  1U);
}

constexpr bool is_from(std::uint64_t const v) { return (v & 1U) == 1U; }

bool matches(line_rule_side const& r, event_attrs const& e) {
  return e.valid_ &&
         (r.admin_ == provider_idx_t::invalid() || r.admin_ == e.admin_) &&
         (r.cat_any_ || r.cat_ == e.cat_) &&
         (r.line_any_ || r.line_ == e.line_) &&
         (r.dir_ == '*' || r.dir_ == e.dir_);
}

using sig_t = std::vector<std::uint64_t>;

// index of the global rule sides by specific admin (avoids scanning all
// global rules for every stop event)
struct global_side_index {
  global_side_index(std::vector<line_rule> const& line_global,
                    std::vector<admin_rule> const& admin_global) {
    for (auto i = std::size_t{0U}; i != line_global.size(); ++i) {
      auto const& r = line_global[i];
      auto const add = [&](line_rule_side const& s, bool const from) {
        auto const side = encode_side(rule_tier::kLineGlobal, i, from);
        if (s.admin_ == provider_idx_t::invalid()) {
          (from ? from_any_admin_ : to_any_admin_).emplace_back(side);
        } else {
          (from ? from_by_admin_ : to_by_admin_)[s.admin_].emplace_back(side);
        }
      };
      add(r.from_, true);
      add(r.to_, false);
    }
    for (auto i = std::size_t{0U}; i != admin_global.size(); ++i) {
      auto const& r = admin_global[i];
      from_by_admin_[r.from_].emplace_back(
          encode_side(rule_tier::kAdminGlobal, i, true));
      to_by_admin_[r.to_].emplace_back(
          encode_side(rule_tier::kAdminGlobal, i, false));
    }
  }

  hash_map<provider_idx_t, std::vector<std::uint64_t>> from_by_admin_;
  hash_map<provider_idx_t, std::vector<std::uint64_t>> to_by_admin_;
  std::vector<std::uint64_t> from_any_admin_;  // line rule sides, admin '*'
  std::vector<std::uint64_t> to_any_admin_;
};

struct station_rules {
  std::vector<trip_rule> const* trip_{nullptr};
  std::vector<line_rule> const* line_{nullptr};
  std::vector<admin_rule> const* admin_{nullptr};
  std::vector<line_rule> const* line_global_{nullptr};
  std::vector<admin_rule> const* admin_global_{nullptr};
  global_side_index const* global_index_{nullptr};
  bool has_station_time_{false};
  u8_minutes station_time_{};
};

struct tuple_sides {
  sig_t static_;
  std::vector<std::pair<std::uint64_t, unsigned>> dynamic_;  // side, bitfield
};

tuple_sides compute_sides(station_rules const& sr, stop_attrs const& t) {
  auto s = tuple_sides{};
  auto const add = [&](rule_tier const tier, std::size_t const i,
                       bool const from, unsigned const bitfield_num) {
    if (bitfield_num == 0U) {
      s.static_.emplace_back(encode_side(tier, i, from));
    } else {
      s.dynamic_.emplace_back(encode_side(tier, i, from), bitfield_num);
    }
  };

  if (sr.trip_ != nullptr) {
    for (auto i = std::size_t{0U}; i != sr.trip_->size(); ++i) {
      auto const& r = (*sr.trip_)[i];
      if (t.arr_.valid_ && r.from_nr_ == t.arr_.train_nr_ &&
          r.from_admin_ == t.arr_.admin_) {
        add(rule_tier::kTrip, i, true, r.bitfield_num_);
      }
      if (t.dep_.valid_ && r.to_nr_ == t.dep_.train_nr_ &&
          r.to_admin_ == t.dep_.admin_) {
        add(rule_tier::kTrip, i, false, r.bitfield_num_);
      }
    }
  }
  if (sr.line_ != nullptr) {
    for (auto i = std::size_t{0U}; i != sr.line_->size(); ++i) {
      auto const& r = (*sr.line_)[i];
      if (matches(r.from_, t.arr_)) {
        add(rule_tier::kLineStation, i, true, 0U);
      }
      if (matches(r.to_, t.dep_)) {
        add(rule_tier::kLineStation, i, false, 0U);
      }
    }
  }
  if (sr.admin_ != nullptr) {
    for (auto i = std::size_t{0U}; i != sr.admin_->size(); ++i) {
      auto const& r = (*sr.admin_)[i];
      if (t.arr_.valid_ && r.from_ == t.arr_.admin_) {
        add(rule_tier::kAdminStation, i, true, 0U);
      }
      if (t.dep_.valid_ && r.to_ == t.dep_.admin_) {
        add(rule_tier::kAdminStation, i, false, 0U);
      }
    }
  }
  // station transfer time shadows global rules (HRDF precedence 4 vs. 5/6)
  if (!sr.has_station_time_ && sr.global_index_ != nullptr) {
    auto const& gi = *sr.global_index_;
    auto const check = [&](std::uint64_t const side, event_attrs const& e) {
      auto const from = is_from(side);
      auto const i = idx_of(side);
      if (tier_of(side) == rule_tier::kAdminGlobal) {
        add(rule_tier::kAdminGlobal, i, from, 0U);  // admin matched via index
      } else {
        auto const& rs =
            from ? (*sr.line_global_)[i].from_ : (*sr.line_global_)[i].to_;
        if (matches(rs, e)) {
          add(rule_tier::kLineGlobal, i, from, 0U);
        }
      }
    };
    auto const check_all =
        [&](event_attrs const& e,
            hash_map<provider_idx_t, std::vector<std::uint64_t>> const&
                by_admin,
            std::vector<std::uint64_t> const& any_admin) {
          if (!e.valid_) {
            return;
          }
          if (auto const it = by_admin.find(e.admin_); it != end(by_admin)) {
            for (auto const side : it->second) {
              check(side, e);
            }
          }
          for (auto const side : any_admin) {
            check(side, e);
          }
        };
    check_all(t.arr_, gi.from_by_admin_, gi.from_any_admin_);
    check_all(t.dep_, gi.to_by_admin_, gi.to_any_admin_);
  }
  utl::sort(s.static_);
  return s;
}

struct resolved_transfer {
  u8_minutes time_{};
  bool preferred_{false};  // guaranteed transfer ("!")
};

resolved_transfer resolve_time(station_rules const& sr,
                               u8_minutes const default_time,
                               sig_t const& a,
                               sig_t const& b) {
  // first rule of tier t (rules are ordered by specificity within a tier)
  // with its from side matched by a and its to side matched by b
  auto const find = [&](rule_tier const t) -> std::optional<std::size_t> {
    auto const lo = encode_side(t, 0U, false);
    auto const hi =
        encode_side(static_cast<rule_tier>(static_cast<std::uint64_t>(t) + 1U),
                    0U, false);
    for (auto it = std::lower_bound(begin(a), end(a), lo);
         it != end(a) && *it < hi; ++it) {
      if (is_from(*it) &&
          std::binary_search(begin(b), end(b),
                             encode_side(t, idx_of(*it), false))) {
        return idx_of(*it);
      }
    }
    return std::nullopt;
  };

  if (auto const i = find(rule_tier::kTrip); i.has_value()) {
    auto const& r = (*sr.trip_)[*i];
    return {r.time_, r.guaranteed_};
  }
  if (auto const i = find(rule_tier::kLineStation); i.has_value()) {
    auto const& r = (*sr.line_)[*i];
    return {r.time_, r.guaranteed_};
  }
  if (auto const i = find(rule_tier::kAdminStation); i.has_value()) {
    return {(*sr.admin_)[*i].time_, false};
  }
  if (sr.has_station_time_) {
    return {sr.station_time_, false};
  }
  if (auto const i = find(rule_tier::kLineGlobal); i.has_value()) {
    auto const& r = (*sr.line_global_)[*i];
    return {r.time_, r.guaranteed_};
  }
  if (auto const i = find(rule_tier::kAdminGlobal); i.has_value()) {
    return {(*sr.admin_global_)[*i].time_, false};
  }
  return {default_time, false};
}

}  // namespace

void build_transfer_groups(stamm& st,
                           timetable& tt,
                           transfer_times const& r,
                           transfer_groups& groups) {
  auto const timer = scoped_timer{"loader.hrd.transfer_groups"};

  auto const n_days = std::min(
      std::size_t{kMaxDays},
      static_cast<std::size_t>(st.get_date_range().size().count()) + 8U);

  auto n_virt_locations = 0U;
  auto n_hubs = 0U;
  auto n_edges = 0U;
  auto n_stations = 0U;

  auto const global_index = global_side_index{r.line_global_, r.admin_global_};

  for (auto& [base, station] : groups.stations_) {
    auto const eva = station.eva_;
    auto sr = station_rules{};
    if (auto const it = r.trip_.find(eva); it != end(r.trip_)) {
      sr.trip_ = &it->second;
    }
    if (auto const it = r.line_.find(eva); it != end(r.line_)) {
      sr.line_ = &it->second;
    }
    if (auto const it = r.admin_.find(eva); it != end(r.admin_)) {
      sr.admin_ = &it->second;
    }
    sr.line_global_ = &r.line_global_;
    sr.admin_global_ = &r.admin_global_;
    sr.global_index_ = &global_index;
    sr.has_station_time_ = r.station_.contains(eva);
    sr.station_time_ = tt.locations_.transfer_time_[base];

    auto default_time = tt.locations_.transfer_time_[base];

    // partition tuples into groups by their matched rule sides
    auto sig_to_group = std::map<sig_t, unsigned>{};
    auto group_sigs = std::vector<sig_t>{};
    auto const get_group = [&](sig_t const& sig) -> std::optional<unsigned> {
      if (sig.empty()) {
        return std::nullopt;  // base station
      }
      auto const it = sig_to_group.find(sig);
      if (it != end(sig_to_group)) {
        return it->second;
      }
      auto const id = static_cast<unsigned>(group_sigs.size());
      sig_to_group.emplace(sig, id);
      group_sigs.emplace_back(sig);
      return id;
    };

    struct tuple_groups {
      stop_attrs attrs_;
      std::optional<unsigned> static_group_;
      std::vector<std::optional<unsigned>> by_day_;
    };
    auto tuples = std::vector<tuple_groups>{};

    for (auto const& attrs : station.tuples_) {
      auto const sides = compute_sides(sr, attrs);
      if (sides.static_.empty() && sides.dynamic_.empty()) {
        continue;
      }

      auto t = tuple_groups{.attrs_ = attrs,
                            .static_group_ = get_group(sides.static_),
                            .by_day_ = {}};
      if (!sides.dynamic_.empty()) {
        t.by_day_.resize(n_days);
        auto sig = sig_t{};
        for (auto day = std::size_t{0U}; day != n_days; ++day) {
          sig = sides.static_;
          for (auto const& [side, bitfield_num] : sides.dynamic_) {
            if (st.resolve_bitfield(bitfield_num).test(day)) {
              sig.emplace_back(side);
            }
          }
          utl::sort(sig);
          t.by_day_[day] = get_group(sig);
        }
      }
      tuples.emplace_back(std::move(t));
    }
    station.tuples_.clear();

    if (group_sigs.empty()) {
      continue;
    }

    // drop groups that behave exactly like the base station
    auto const base_sig = sig_t{};
    auto const time = [&](sig_t const& a, sig_t const& b) {
      return resolve_time(sr, default_time, a, b);
    };
    auto useful = std::vector<bool>{};
    auto node_of_group = std::vector<std::size_t>{};
    auto node_sig = std::vector<sig_t const*>{};
    struct cell {
      bool operator==(cell const&) const = default;
      u8_minutes time_{};
      bool preferred_{false};
    };
    auto matrix = std::vector<cell>{};
    auto n = std::size_t{0U};

    // drop groups that behave exactly like the base station, resolve the
    // matrix over the remaining ones (node 0 = base)
    auto const build_matrix = [&]() {
      useful.assign(group_sigs.size(), false);
      for (auto g = std::size_t{0U}; g != group_sigs.size(); ++g) {
        auto const& sig = group_sigs[g];
        auto const differs = [&](sig_t const& h) {
          return time(sig, h).time_ != time(base_sig, h).time_ ||
                 time(h, sig).time_ != time(h, base_sig).time_;
        };
        useful[g] = differs(base_sig) ||
                    time(sig, sig).time_ != time(base_sig, base_sig).time_ ||
                    utl::any_of(group_sigs, differs);
      }

      node_of_group.assign(group_sigs.size(), 0U);
      node_sig.clear();
      node_sig.push_back(&base_sig);
      for (auto g = std::size_t{0U}; g != group_sigs.size(); ++g) {
        if (useful[g]) {
          node_of_group[g] = node_sig.size();
          node_sig.push_back(&group_sigs[g]);
        }
      }
      n = node_sig.size();
      matrix.assign(n * n, cell{});
      for (auto i = std::size_t{0U}; i != n; ++i) {
        for (auto j = std::size_t{0U}; j != n; ++j) {
          auto const resolved = time(*node_sig[i], *node_sig[j]);
          matrix[i * n + j] = cell{resolved.time_, resolved.preferred_};
        }
      }
    };
    build_matrix();
    auto const at = [&](std::size_t const i, std::size_t const j) {
      return matrix[i * n + j];
    };

    // majority fold (as for GTFS transfers.txt): at stations WITHOUT an
    // explicit UMSTEIGB time, the most frequent rule value becomes the
    // station transfer time - the generic 5 min default is a guess while
    // the rule values are operator-validated for this station. Groups whose
    // rules state the folded value collapse into the base station; only
    // deviating rules keep their splits. (Trossingen: 3480 uniform 3 min
    // trip pairs -> transfer_time_ = 3, zero virtual locations.)
    if (!sr.has_station_time_ && n > 1U) {
      auto hist = std::map<u8_minutes, unsigned>{};
      auto total = 0U;
      for (auto i = std::size_t{0U}; i != n; ++i) {
        for (auto j = std::size_t{0U}; j != n; ++j) {
          auto const c = at(i, j);
          if (i != j && c.time_ != default_time && !c.preferred_) {
            ++hist[c.time_];
            ++total;
          }
        }
      }
      if (total != 0U) {
        auto const majority = std::max_element(
            begin(hist), end(hist), [](auto const& a, auto const& b) {
              return a.second < b.second;
            });
        if (2U * majority->second >= total) {
          default_time = majority->first;
          tt.locations_.transfer_time_[base] = default_time;
          build_matrix();
        }
      }
    }

    // merge groups that are behaviorally identical: same intra-group time and
    // identical relations to every behavior class (partition refinement, the
    // base station is pinned as class 0). Merging is only applied where it is
    // exact: one value per class pair, and pairs merged into one location must
    // behave like the location's transfer time (no preferred flag to lose).
    auto cls = std::vector<unsigned>(n, 1U);
    cls[0] = 0U;
    auto n_classes = n == 1U ? std::size_t{1U} : std::size_t{2U};
    while (n_classes < n) {
      // refine to fixpoint
      auto changed = true;
      while (changed) {
        changed = false;
        auto sig_to_class = std::map<std::vector<std::uint64_t>, unsigned>{};
        auto next = std::vector<unsigned>(n, 0U);
        for (auto i = std::size_t{1U}; i != n; ++i) {
          auto key = std::vector<std::uint64_t>{cls[i], at(i, i).time_.count()};
          auto rel = std::vector<std::uint64_t>{};
          for (auto j = std::size_t{0U}; j != n; ++j) {
            if (j != i) {
              auto const out = at(i, j);
              auto const in = at(j, i);
              rel.push_back((std::uint64_t{cls[j]} << 20U) |
                            (std::uint64_t{out.time_.count()} << 12U) |
                            (std::uint64_t{out.preferred_} << 11U) |
                            (std::uint64_t{in.time_.count()} << 1U) |
                            std::uint64_t{in.preferred_});
            }
          }
          utl::erase_duplicates(rel);
          key.insert(end(key), begin(rel), end(rel));
          next[i] = 1U + utl::get_or_create(sig_to_class, key, [&]() {
            return static_cast<unsigned>(sig_to_class.size());
          });
        }
        changed = next != cls;
        cls = std::move(next);
        n_classes = 1U + sig_to_class.size();
      }

      // verify exactness, split on violation and refine again
      auto rep = std::vector<std::size_t>(n_classes, 0U);
      for (auto i = n; i != 0U; --i) {
        rep[cls[i - 1U]] = i - 1U;
      }
      auto violation = false;
      for (auto i = std::size_t{1U}; i != n && !violation; ++i) {
        for (auto j = std::size_t{1U}; j != n; ++j) {
          if (i == j) {
            continue;
          }
          auto const bad =
              cls[i] == cls[j]
                  ? at(i, j).time_ != at(i, i).time_ || at(i, j).preferred_
                  : at(i, j) != at(rep[cls[i]], rep[cls[j]]);
          if (bad) {
            cls[j] = static_cast<unsigned>(n_classes++);
            violation = true;
            break;
          }
        }
      }
      if (!violation) {
        break;
      }
    }

    // create one virtual child location per behavior class
    auto class_locations = std::vector<location_idx_t>(n_classes, base);
    auto class_rep = std::vector<std::size_t>(n_classes, 0U);
    for (auto i = std::size_t{1U}; i != n; ++i) {
      class_rep[cls[i]] = i;
      if (class_locations[cls[i]] != base) {
        continue;
      }
      auto l = location{tt, base};
      l.id_ = {};  // virtual locations are not lookupable by id
      l.type_ = location_type::kVirt;
      l.parent_ = base;
      auto const child = register_location(tt, l);
      tt.locations_.children_[base].emplace_back(child);
      tt.locations_.transfer_time_[child] = at(i, i).time_;
      class_locations[cls[i]] = child;
      ++n_virt_locations;
    }

    auto virt_locations = std::vector<location_idx_t>(group_sigs.size(), base);
    for (auto g = std::size_t{0U}; g != group_sigs.size(); ++g) {
      if (useful[g]) {
        virt_locations[g] = class_locations[cls[node_of_group[g]]];
      }
    }

    if (n_classes <= 1U) {
      continue;
    }
    ++n_stations;

    // store tuple -> location assignment for phase 2
    auto const to_location = [&](std::optional<unsigned> const g) {
      return g.has_value() ? virt_locations[*g] : base;
    };
    for (auto& t : tuples) {
      auto a = transfer_groups::assignment{};
      a.static_group_ = to_location(t.static_group_);
      auto all_base = a.static_group_ == base;
      if (!t.by_day_.empty()) {
        a.by_day_.resize(t.by_day_.size());
        for (auto day = std::size_t{0U}; day != t.by_day_.size(); ++day) {
          a.by_day_[day] = to_location(t.by_day_[day]);
          all_base = all_base && a.by_day_[day] == base;
        }
      }
      if (!all_base) {
        station.assignment_.emplace(t.attrs_, std::move(a));
      }
    }

    // per-member door bits: row_clean = no elevated cell in the member's
    // ROW (its arrival may be broadcast to everyone at the default),
    // col_clean = no elevated cell in its COLUMN (it may collect the
    // pooled arrival at the default). a pair (a -> b) is derived iff
    // row_clean[a] || col_clean[b]; everything else is materialized.
    auto row_clean = std::vector<bool>(n_classes, true);
    auto col_clean = std::vector<bool>(n_classes, true);
    for (auto a = std::size_t{0U}; a != n_classes; ++a) {
      for (auto b = std::size_t{0U}; b != n_classes; ++b) {
        if (a != b && at(class_rep[a], class_rep[b]).time_ > default_time) {
          row_clean[a] = false;
          col_clean[b] = false;
        }
      }
    }

    // profile hubs: a member with elevated cells derives its remaining
    // default cells through one aggregation hub (location_type::kHub) per
    // distinct elevated profile. fwd side: row profile = the member's
    // elevated targets; the hub relaxes every member except them. bwd
    // side mirrors with column profiles (elevated sources). with these,
    // EVERY default cell is derivable in both search directions and only
    // deviating cells are materialized.
    auto const ensure = [](vector_map<location_idx_t, location_idx_t>& m,
                           location_idx_t const l) {
      if (m.size() <= to_idx(l)) {
        auto const old = m.size();
        m.resize(to_idx(l) + 1U);
        for (auto i = old; i != m.size(); ++i) {
          m[location_idx_t{i}] = location_idx_t::invalid();
        }
      }
    };
    auto const make_hubs = [&](bool const row_side,
                               vector_map<location_idx_t, location_idx_t>&
                                   assign) {
      auto by_profile =
          std::map<std::vector<std::size_t>, std::vector<std::size_t>>{};
      for (auto x = std::size_t{0U}; x != n_classes; ++x) {
        auto profile = std::vector<std::size_t>{};
        for (auto y = std::size_t{0U}; y != n_classes; ++y) {
          if (x != y && (row_side ? at(class_rep[x], class_rep[y]).time_
                                  : at(class_rep[y], class_rep[x]).time_) >
                            default_time) {
            profile.push_back(y);
          }
        }
        if (!profile.empty()) {
          by_profile[std::move(profile)].push_back(x);
        }
      }
      for (auto const& [profile, members] : by_profile) {
        auto l = location{tt, base};
        l.id_ = {};  // hubs are not lookupable by id
        l.type_ = location_type::kHub;
        l.parent_ = base;
        auto const hub = register_location(tt, l);
        tt.locations_.transfer_time_[hub] = default_time;
        auto excl = std::vector<location_idx_t>{};
        excl.reserve(profile.size());
        for (auto const y : profile) {
          excl.push_back(class_locations[y]);
        }
        std::sort(begin(excl), end(excl));
        for (auto const e : excl) {
          tt.locations_.hub_excl_[hub].emplace_back(e);
        }
        for (auto const x : members) {
          ensure(assign, class_locations[x]);
          assign[class_locations[x]] = hub;
        }
        ++n_hubs;
      }
    };
    make_hubs(true, tt.locations_.fwd_hub_);
    make_hubs(false, tt.locations_.bwd_hub_);

    // emit only the deviating (or preferred) cells
    for (auto a = std::size_t{0U}; a != n_classes; ++a) {
      for (auto b = std::size_t{0U}; b != n_classes; ++b) {
        if (a == b) {
          continue;
        }
        auto const c = at(class_rep[a], class_rep[b]);
        if (c.time_ == default_time && !c.preferred_) {
          continue;  // implicit
        }
        tt.locations_.transfer_rule_fps_[class_locations[a]].emplace_back(
            footpath{class_locations[b], c.time_});
        if (c.preferred_) {  // guaranteed transfer ("!")
          tt.locations_.preferred_transfers_[class_locations[a]].emplace_back(
              preferred_transfer{.to_ = class_locations[b]});
        }
        ++n_edges;
      }
    }
    auto const set_bit = [](bitvec& bv, location_idx_t const l) {
      if (bv.size() <= to_idx(l)) {
        bv.resize(to_idx(l) + 1U);
      }
      bv.set(to_idx(l), true);
    };
    for (auto x = std::size_t{0U}; x != n_classes; ++x) {
      if (row_clean[x]) {
        set_bit(tt.locations_.virt_group_out_, class_locations[x]);
      }
      if (col_clean[x]) {
        set_bit(tt.locations_.virt_group_in_, class_locations[x]);
      }
    }
  }

  log(log_lvl::info, "loader.hrd.transfer_groups",
      "created {} transfer group locations at {} stations, {} transfer edges, "
      "{} hubs",
      n_virt_locations, n_stations, n_edges, n_hubs);
}

}  // namespace nigiri::loader::hrd
