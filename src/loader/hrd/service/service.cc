#include "nigiri/loader/hrd/service/service.h"

#include <algorithm>
#include <numeric>
#include <tuple>

#include "fmt/format.h"

#include "utl/pairwise.h"
#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/clasz.h"
#include "nigiri/logging.h"

namespace nigiri::loader::hrd {

std::ostream& operator<<(std::ostream& out, parser_info const& pi) {
  return out << pi.filename_ << ":" << pi.dbg_.line_number_;
}

template <typename It, typename Predicate>
inline It find_nth(It begin, It end, std::size_t n, Predicate fun) {
  assert(n != 0);
  std::size_t num_elements_found = 0;
  auto it = begin;
  while (it != end && num_elements_found != n) {
    it = std::find_if(it, end, fun);
    ++num_elements_found;
    if (it != end && num_elements_found != n) {
      ++it;
    }
  }
  return it;
}

struct range {
  range() = default;
  range(std::vector<service::stop> const& stops,
        utl::cstr from_eva_or_idx,
        utl::cstr to_eva_or_idx,
        utl::cstr from_hhmm_or_idx,
        utl::cstr to_hhmm_or_idx) {
    if (from_eva_or_idx.trim().empty() && to_eva_or_idx.trim().empty() &&
        from_hhmm_or_idx.trim().empty() && to_hhmm_or_idx.trim().empty()) {
      from_idx_ = 0U;
      to_idx_ = stops.size() - 1U;
    } else {
      from_idx_ = get_index(stops, from_eva_or_idx, from_hhmm_or_idx, true);
      to_idx_ = get_index(stops, to_eva_or_idx, to_hhmm_or_idx, false);
    }
  }

  size_t from_idx() const { return from_idx_; }
  size_t to_idx() const { return to_idx_; }

private:
  bool is_index(utl::cstr s) { return s[0] == '#'; }

  size_t parse_index(utl::cstr s) { return parse_verify<size_t>(s.substr(1)); }

  size_t get_index(std::vector<service::stop> const& stops,
                   utl::cstr eva_or_idx,
                   utl::cstr hhmm_or_idx,
                   bool is_departure_event) {
    assert(!eva_or_idx.empty() && !hhmm_or_idx.empty());
    if (is_index(eva_or_idx)) {
      // eva_or_idx is an index which is already definite
      return parse_index(eva_or_idx);
    } else if (is_index(hhmm_or_idx) || hhmm_or_idx.trim().len == 0) {
      // eva_or_idx is not an index -> eva_or_idx is an eva number
      // hhmm_or_idx is empty -> search for first occurrence
      // hhmm_or_idx is an index -> search for nth occurrence
      const auto eva_num = parse<unsigned>(eva_or_idx);
      const auto n = is_index(hhmm_or_idx) ? parse_index(hhmm_or_idx) + 1 : 1;
      const auto it = find_nth(
          begin(stops), end(stops), n,
          [&](service::stop const& s) { return s.eva_num_ == eva_num; });
      utl::verify(it != end(stops),
                  "{}th occurrence of eva number {} not found", n, eva_num);
      return static_cast<size_t>(std::distance(begin(stops), it));
    } else {
      // hhmm_or_idx must be a time
      // -> return stop where eva number and time matches
      const auto eva_num = parse<unsigned>(eva_or_idx);
      const auto time = hhmm_to_min(parse<int>(hhmm_or_idx.substr(1)));
      const auto it =
          std::find_if(begin(stops), end(stops), [&](service::stop const& s) {
            return s.eva_num_ == eva_num &&
                   (is_departure_event ? s.dep_.time_ : s.arr_.time_) == time;
          });
      utl::verify(it != end(stops),
                  "event with time {} at eva number {} not found", time,
                  eva_num);
      return static_cast<size_t>(std::distance(begin(stops), it));
    }
  }

  size_t from_idx_{}, to_idx_{};
};

std::vector<std::pair<utl::cstr, range>> compute_ranges(
    std::vector<utl::cstr> const& spec_lines,
    std::vector<service::stop> const& stops,
    range_parse_information const& parse_info) {
  std::vector<std::pair<utl::cstr, range>> parsed(spec_lines.size());
  std::transform(
      begin(spec_lines), end(spec_lines), begin(parsed), [&](utl::cstr spec) {
        return std::make_pair(
            spec, range(stops, spec.substr(parse_info.from_eva_or_idx_),
                        spec.substr(parse_info.to_eva_or_idx_),
                        spec.substr(parse_info.from_hhmm_or_idx_),
                        spec.substr(parse_info.to_hhmm_or_idx_)));
      });
  return parsed;
}

template <typename TargetInformationType, typename TargetInformationParserFun>
void parse_range(std::vector<utl::cstr> const& spec_lines,
                 range_parse_information const& parse_info,
                 std::vector<service::stop> const& stops,
                 std::vector<service::section>& sections,
                 TargetInformationType service::section::*member,
                 TargetInformationParserFun parse_target_info) {
  for (auto const& r : compute_ranges(spec_lines, stops, parse_info)) {
    TargetInformationType target_info = parse_target_info(r.first, r.second);
    for (auto i = r.second.from_idx(); i < r.second.to_idx(); ++i) {
      (sections[i].*member) = target_info;
    }
  }
}

template <typename TargetInformationType, typename TargetInformationParserFun>
void parse_range(std::vector<utl::cstr> const& spec_lines,
                 range_parse_information const& parse_info,
                 std::vector<service::stop> const& stops,
                 std::vector<service::section>& sections,
                 std::vector<TargetInformationType> service::section::*member,
                 TargetInformationParserFun parse_target_info) {
  for (auto const& r : compute_ranges(spec_lines, stops, parse_info)) {
    TargetInformationType target_info = parse_target_info(r.first, r.second);
    for (auto i = r.second.from_idx(); i < r.second.to_idx(); ++i) {
      (sections[i].*member).push_back(target_info);
    }
  }
}

service::stop parse_stop(utl::cstr stop) {
  return {eva_number{parse_verify<unsigned>(stop.substr(0, utl::size(7)))},
          {hhmm_to_min(
               parse<int>(stop.substr(30, utl::size(5)), service::kTimeNotSet)),
           stop[29] != '-'},
          {hhmm_to_min(
               parse<int>(stop.substr(37, utl::size(5)), service::kTimeNotSet)),
           stop[36] != '-'}};
}

int initial_train_num(specification const& spec) {
  return parse_verify<int>(spec.internal_service_.substr(3, utl::size(5)));
}

utl::cstr initial_admin(specification const& spec) {
  return spec.internal_service_.substr(9, utl::size(6));
}

utl::cstr stop_train_num(utl::cstr const& stop) {
  return stop.substr(43, utl::size(5)).trim();
}

utl::cstr stop_admin(utl::cstr const& stop) {
  return stop.substr(49, utl::size(6)).trim();
}

service::section parse_initial_section(stamm& st, specification const& spec) {
  auto const first_stop = spec.stops_.front();
  auto const train_num = stop_train_num(first_stop);
  auto const admin = stop_admin(first_stop);
  return {train_num.empty() ? initial_train_num(spec)
                            : parse_verify<int>(train_num),
          st.resolve_provider(
              admin.empty() ? spec.internal_service_.substr(9, utl::size(6))
                            : admin)};
}

std::vector<service::section> parse_section(
    stamm& st, std::vector<service::section>&& sections, utl::cstr stop) {
  auto const train_num = stop_train_num(stop);
  auto const admin = stop_admin(stop);

  auto last_section = sections.back();
  sections.emplace_back(
      train_num.empty() ? last_section.train_num_
                        : parse_verify<int>(train_num),
      admin.empty() ? last_section.admin_ : st.resolve_provider(admin));

  return std::move(sections);
}

bool specification::is_empty() const { return !internal_service_; }

bool specification::valid() const {
  return ignore() || (!categories_.empty() && stops_.size() >= 2 &&
                      !traffic_days_.empty() && !is_empty());
}

bool specification::ignore() const {
  return !is_empty() && !internal_service_.starts_with("*Z");
}

void specification::reset() {
  internal_service_ = utl::cstr(nullptr, 0);
  traffic_days_.clear();
  categories_.clear();
  line_information_.clear();
  attributes_.clear();
  directions_.clear();
  stops_.clear();
}

bool specification::read_line(utl::cstr line,
                              char const* filename,
                              unsigned const line_number) {
  if (line.len == 0) {
    return false;
  }

  if (std::isdigit(line[0]) != 0) {
    stops_.push_back(line);
    return false;
  }

  if (line.len < 2 || line[0] != '*') {
    throw utl::fail("invalid line {}:{}: \"{}\"", filename, line_number,
                    line.view());
  }

  // ignore *I, *GR, *SH, *T, *KW, *KWZ
  bool potential_kurswagen = false;
  switch (line[1]) {
    case 'K': potential_kurswagen = true; [[fallthrough]];
    case 'Z': [[fallthrough]];
    case 'T':
      if (potential_kurswagen && line.len > 3 && line[3] == 'Z') {
        // ignore KWZ line
      } else if (is_empty()) {
        filename_ = filename;
        line_number_from_ = line_number;
        internal_service_ = line;
      } else {
        return true;
      }
      break;
    case 'A':
      if (line.starts_with("*A VE")) {
        traffic_days_.push_back(line);
      } else {  // *A based on HRD format version 5.00.8
        attributes_.push_back(line);
      }
      break;
    case 'G':
      if (!line.starts_with("*GR")) {
        categories_.push_back(line);
      }
      break;
    case 'L': line_information_.push_back(line); break;
    case 'R': directions_.push_back(line); break;
  }

  return false;
}

service::service(config const& c,
                 stamm& st,
                 source_file_idx_t const source_file_idx,
                 specification const& spec)
    : origin_{parser_info{spec.filename_,
                          {source_file_idx, spec.line_number_from_}}},
      num_repetitions_{
          parse<unsigned>(spec.internal_service_.substr(22, utl::size(3)))},
      interval_{
          parse<unsigned>(spec.internal_service_.substr(26, utl::size(3)))},
      stops_{utl::to_vec(spec.stops_, parse_stop)},
      sections_{std::accumulate(
          std::next(begin(spec.stops_)),
          std::next(begin(spec.stops_),
                    static_cast<long>(spec.stops_.size() - 1)),
          std::vector<section>({parse_initial_section(st, spec)}),
          [&](auto&& a, auto&& b) {
            return parse_section(st, std::forward<decltype(a)>(a),
                                 std::forward<decltype(b)>(b));
          })},
      initial_admin_{st.resolve_provider(initial_admin(spec))},
      initial_train_num_{initial_train_num(spec)} {
  parse_range(spec.attributes_, c.attribute_parse_info_, stops_, sections_,
              &section::attributes_, [&](utl::cstr line, range const&) {
                return attribute{
                    parse<unsigned>(line.substr(c.s_info_.traff_days_)),
                    st.resolve_attribute(line.substr(c.s_info_.att_code_))};
              });

  parse_range(spec.categories_, c.category_parse_info_, stops_, sections_,
              &section::category_, [&](utl::cstr line, range const&) {
                return st.resolve_category(line.substr(c.s_info_.cat_));
              });

  parse_range(spec.line_information_, c.line_parse_info_, stops_, sections_,
              &section::line_information_, [&](utl::cstr line, range const&) {
                return line.substr(c.s_info_.line_).trim();
              });

  parse_range(
      spec.traffic_days_, c.traffic_days_parse_info_, stops_, sections_,
      &section::traffic_days_, [&](utl::cstr line, range const&) {
        return parse_verify<unsigned>(line.substr(c.s_info_.traff_days_));
      });

  parse_range(spec.directions_, c.direction_parse_info_, stops_, sections_,
              &section::direction_, [&](utl::cstr line, range const& r) {
                if (isdigit(line[5]) != 0) {
                  return st.resolve_direction({line.substr(c.s_info_.dir_)});
                } else if (line[5] == ' ') {
                  return st.resolve_direction({stops_[r.to_idx()].eva_num_});
                } else {
                  return st.resolve_direction({line.substr(c.s_info_.dir_)});
                }
              });

  verify_service();
}

void service::verify_service() {
  utl::verify(stops_.size() >= 2, "service with less than 2 stops");
}

std::vector<std::pair<int, provider_idx_t>> service::get_ids() const {
  std::vector<std::pair<int, provider_idx_t>> ids;

  // Add first service id.
  auto const& first_section = sections_.front();
  ids.emplace_back(first_section.train_num_, first_section.admin_);

  // Add new service id if it changed.
  for (auto i = size_t{1U}; i < sections_.size(); ++i) {
    auto const id = std::pair{sections_[i].train_num_, sections_[i].admin_};
    if (id != ids.back()) {
      ids.emplace_back(id);
    }
  }

  return ids;
}

int service::find_first_stop_at(eva_number const eva_num) const {
  for (auto i = size_t{0U}; i < stops_.size(); ++i) {
    if (stops_[i].eva_num_ == eva_num) {
      return static_cast<int>(i);
    }
  }
  return kTimeNotSet;
}

int service::get_first_stop_index_at(eva_number const eva_num) const {
  auto const idx = find_first_stop_at(eva_num);
  utl::verify(idx != kTimeNotSet, "stop eva number {} not found", eva_num);
  return idx;
}

}  // namespace nigiri::loader::hrd
