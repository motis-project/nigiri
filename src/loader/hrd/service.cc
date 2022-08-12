#include "nigiri/loader/hrd/service.h"

#include <algorithm>
#include <numeric>
#include <tuple>

#include "fmt/format.h"

#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/hrd/timezone.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"

namespace nigiri::loader::hrd {

std::ostream& operator<<(std::ostream& out, parser_info const& pi) {
  return out << pi.filename_ << ":" << pi.line_number_from_ << ":"
             << pi.line_number_to_;
}

string parser_info::str() const {
  return fmt::format("{}:{}:{}", filename_, line_number_from_, line_number_to_);
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
      from_idx_ = 0;
      to_idx_ = stops.size() - 1;
    } else {
      from_idx_ = get_index(stops, from_eva_or_idx, from_hhmm_or_idx, true);
      to_idx_ = get_index(stops, to_eva_or_idx, to_hhmm_or_idx, false);
    }
  }

  size_t from_idx() const { return from_idx_; }
  size_t to_idx() const { return to_idx_; }

private:
  bool is_index(utl::cstr s) { return s[0] == '#'; }

  int parse_index(utl::cstr s) { return parse_verify<int>(s.substr(1)); }

  int get_index(std::vector<service::stop> const& stops,
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
      const auto eva_num = parse_verify<int>(eva_or_idx);
      const auto n = is_index(hhmm_or_idx) ? parse_index(hhmm_or_idx) + 1 : 1;
      const auto it = find_nth(
          begin(stops), end(stops), n,
          [&](service::stop const& s) { return s.eva_num_ == eva_num; });
      utl::verify(it != end(stops),
                  "{}th occurrence of eva number {} not found", n, eva_num);
      return static_cast<int>(std::distance(begin(stops), it));
    } else {
      // hhmm_or_idx must be a time
      // -> return stop where eva number and time matches
      const auto eva_num = parse_verify<int>(eva_or_idx);
      const auto time = hhmm_to_min(parse_verify<int>(hhmm_or_idx.substr(1)));
      const auto it =
          std::find_if(begin(stops), end(stops), [&](service::stop const& s) {
            return s.eva_num_ == eva_num &&
                   (is_departure_event ? s.dep_.time_ : s.arr_.time_) == time;
          });
      utl::verify(it != end(stops),
                  "event with time {} at eva number {} not found", time,
                  eva_num);
      return static_cast<int>(std::distance(begin(stops), it));
    }
  }

  int from_idx_{}, to_idx_{};
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
                 std::vector<TargetInformationType> service::section::*member,
                 TargetInformationParserFun parse_target_info) {
  for (auto const& r : compute_ranges(spec_lines, stops, parse_info)) {
    TargetInformationType target_info = parse_target_info(r.first, r.second);
    for (int i = r.second.from_idx(); i < r.second.to_idx(); ++i) {
      (sections[i].*member).push_back(target_info);
    }
  }
}

service::stop parse_stop(utl::cstr stop) {
  return {
      eva_number{parse_verify<unsigned>(stop.substr(0, utl::size(7)))},
      {hhmm_to_min(parse<int>(stop.substr(30, utl::size(5)), service::NOT_SET)),
       stop[29] != '-'},
      {hhmm_to_min(parse<int>(stop.substr(37, utl::size(5)), service::NOT_SET)),
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

service::section parse_initial_section(specification const& spec) {
  auto const first_stop = spec.stops_.front();
  auto const train_num = stop_train_num(first_stop);
  auto const admin = stop_admin(first_stop);
  return {
      train_num.empty() ? initial_train_num(spec)
                        : parse_verify<int>(train_num),
      admin.empty() ? spec.internal_service_.substr(9, utl::size(6)) : admin};
}

std::vector<service::section> parse_section(
    std::vector<service::section>&& sections, utl::cstr stop) {
  auto train_num = stop_train_num(stop);
  auto admin = stop_admin(stop);

  auto last_section = sections.back();
  sections.emplace_back(train_num.empty() ? last_section.train_num_
                                          : parse_verify<int>(train_num),
                        admin.empty() ? last_section.admin_ : admin);

  return sections;
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
                              int line_number) {
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
    case 'K': potential_kurswagen = true;
    /* no break */
    case 'Z':
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

service::service(parser_info origin,
                 int num_repetitions,
                 int interval,
                 std::vector<stop> stops,
                 std::vector<section> sections,
                 bitfield traffic_days,
                 utl::cstr initial_admin,
                 int initial_train_num)
    : origin_{origin},
      num_repetitions_{num_repetitions},
      interval_{interval},
      stops_{std::move(stops)},
      sections_{std::move(sections)},
      traffic_days_{traffic_days},
      initial_admin_{initial_admin},
      initial_train_num_{initial_train_num} {}

service::service(config const& c, specification const& spec)
    : origin_{parser_info{spec.filename_, spec.line_number_from_,
                          spec.line_number_to_}},
      num_repetitions_{
          parse<int>(spec.internal_service_.substr(22, utl::size(3)))},
      interval_{parse<int>(spec.internal_service_.substr(26, utl::size(3)))},
      stops_{utl::to_vec(spec.stops_, parse_stop)},
      sections_{
          std::accumulate(std::next(begin(spec.stops_)),
                          std::next(begin(spec.stops_), spec.stops_.size() - 1),
                          std::vector<section>({parse_initial_section(spec)}),
                          parse_section)},
      initial_admin_{initial_admin(spec)},
      initial_train_num_{initial_train_num(spec)} {
  parse_range(spec.attributes_, c.attribute_parse_info_, stops_, sections_,
              &section::attributes_, [&c](utl::cstr line, range const&) {
                return attribute{
                    parse_verify<int>(line.substr(c.s_info_.att_eva_)),
                    line.substr(c.s_info_.att_code_)};
              });

  parse_range(spec.categories_, c.category_parse_info_, stops_, sections_,
              &section::category_, [&c](utl::cstr line, range const&) {
                return line.substr(c.s_info_.cat_);
              });

  parse_range(spec.line_information_, c.line_parse_info_, stops_, sections_,
              &section::line_information_, [&c](utl::cstr line, range const&) {
                return line.substr(c.s_info_.line_).trim();
              });

  parse_range(spec.traffic_days_, c.traffic_days_parse_info_, stops_, sections_,
              &section::traffic_days_, [&c](utl::cstr line, range const&) {
                return parse_verify<int>(line.substr(c.s_info_.traff_days_));
              });

  parse_range(spec.directions_, c.direction_parse_info_, stops_, sections_,
              &section::directions_, [&](utl::cstr line, range const& r) {
                if (isdigit(line[5]) != 0) {
                  return direction_info{line.substr(c.s_info_.dir_)};
                } else if (line[5] == ' ') {
                  return direction_info{stops_[r.to_idx()].eva_num_};
                } else {
                  return direction_info{line.substr(c.s_info_.dir_)};
                }
              });

  verify_service();
}

void service::verify_service() {
  int section_index = 0;
  utl::verify(stops_.size() >= 2, "service with less than 2 stops");
  for (auto& section : sections_) {
    utl::verify(section.traffic_days_.size() == 1,
                "{}:{}:{}: section {} invalid: {} multiple traffic days",
                origin_.filename_, origin_.line_number_from_,
                origin_.line_number_to_, section_index,
                section.traffic_days_.size());
    utl::verify(section.line_information_.size() <= 1,
                "{}:{}:{}: section {} invalid: {} line information",
                origin_.filename_, origin_.line_number_from_,
                origin_.line_number_to_, section_index,
                section.line_information_.size());
    utl::verify(section.category_.size() == 1,
                "{}:{}:{}: section {} invalid: {} categories",
                origin_.filename_, origin_.line_number_from_,
                origin_.line_number_to_, section_index,
                section.category_.size());

    try {
      utl::verify(section.directions_.size() <= 1,
                  "{}:{}:{}: section {} invalid: {} direction information",
                  origin_.filename_, origin_.line_number_from_,
                  origin_.line_number_to_, section_index,
                  section.directions_.size());
    } catch (std::runtime_error const&) {
      log(log_lvl::error, "quick fixing direction info: {}:{}",
          origin_.filename_, origin_.line_number_from_);
      section.directions_.resize(1);
    }
    ++section_index;
  }
}

std::vector<std::pair<int, utl::cstr>> service::get_ids() const {
  std::vector<std::pair<int, utl::cstr>> ids;

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
  return NOT_SET;
}

int service::get_first_stop_index_at(eva_number const eva_num) const {
  auto const idx = find_first_stop_at(eva_num);
  utl::verify(idx != NOT_SET, "stop eva number {} not found", eva_num);
  return idx;
}

int service::event_time(int stop_index, event_type evt) const {
  return evt == event_type::DEP ? stops_[stop_index].dep_.time_
                                : stops_[stop_index].arr_.time_;
}

unsigned service::traffic_days_offset_at_stop(int stop_index,
                                              event_type evt) const {
  return static_cast<unsigned>(event_time(stop_index, evt) / 1440);
}

bitfield service::traffic_days_at_stop(int stop_index, event_type evt) const {
  return traffic_days_ << traffic_days_offset_at_stop(stop_index, evt);
}

vector<tz_offsets> service::get_stop_timezones(timezone_map_t const& tz) const {
  auto i = 0U;
  auto stop_times = vector<tz_offsets>(stops_.size() * 2 - 2);
  for (auto const& [from, to] : utl::pairwise(stops_)) {
    stop_times[i++] = get_tz(tz, from.eva_num_).second;
    stop_times[i++] = get_tz(tz, to.eva_num_).second;
  }
  return stop_times;
}

vector<duration_t> service::get_stop_times() const {
  auto i = 0U;
  auto stop_times = vector<duration_t>(stops_.size() * 2 - 2);
  for (auto const& [from, to] : utl::pairwise(stops_)) {
    stop_times[i++] = duration_t{from.dep_.time_};
    stop_times[i++] = duration_t{to.arr_.time_};
  }
  assert(i == stop_times.size());
  return stop_times;
}

void service::set_stop_times(vector<duration_t> const& stop_times) {
  utl::verify(stop_times.size() == stops_.size() * 2 - 2,
              "service::set_stop_times: invalid stop_times size");
  auto i = 0U;
  for (auto const& [from, to] : utl::pairwise(stops_)) {
    from.dep_.time_ = stop_times[i++].count();
    to.arr_.time_ = stop_times[i++].count();
  }
  assert(i == stop_times.size());
}

}  // namespace nigiri::loader::hrd
