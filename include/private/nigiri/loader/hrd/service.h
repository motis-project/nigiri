#pragma once

#include <cassert>
#include <utility>
#include <variant>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/pairwise.h"
#include "utl/parser/cstr.h"
#include "utl/to_vec.h"
#include "utl/zip.h"

#include "nigiri/loader/hrd/basic_info.h"
#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/timezone.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct parser_info {
  friend std::ostream& operator<<(std::ostream& out, parser_info const& pi);
  char const* filename_;
  int line_number_from_;
  int line_number_to_;
};

struct specification {
  bool is_empty() const;
  bool valid() const;
  bool ignore() const;
  void reset();
  bool read_line(utl::cstr line, char const* filename, int line_number);

  char const* filename_{"unknown file"};
  int line_number_from_{-1};
  int line_number_to_{-1};
  utl::cstr internal_service_;
  std::vector<utl::cstr> traffic_days_;
  std::vector<utl::cstr> categories_;
  std::vector<utl::cstr> line_information_;
  std::vector<utl::cstr> attributes_;
  std::vector<utl::cstr> directions_;
  std::vector<utl::cstr> stops_;
};

struct service {
  static const constexpr auto NOT_SET = -1;  // NOLINT

  struct event {
    int time_;
    bool in_out_allowed_;
  };

  struct stop {
    eva_number eva_num_;
    event arr_, dep_;
  };

  struct attribute {
    CISTA_COMPARABLE()

    attribute(int bitfield_num, utl::cstr code)
        : bitfield_num_{bitfield_num}, code_{code} {}

    int bitfield_num_;
    utl::cstr code_;
  };

  using direction_info =
      std::variant<utl::cstr /* custom string */, eva_number /* eva number */>;

  struct section {
    section() = default;
    section(int train_num, utl::cstr admin)
        : train_num_{train_num}, admin_{admin} {}

    int train_num_{0};
    utl::cstr admin_;
    std::vector<attribute> attributes_;
    std::vector<utl::cstr> category_;
    std::vector<utl::cstr> line_information_;
    std::vector<direction_info> directions_;
    std::vector<int> traffic_days_;
  };

  service(service const& s, vector<duration_t> const& times, bitfield const& b)
      : service{s} {
    traffic_days_ = b;
    set_stop_times(times);
  }

  service(parser_info origin,
          int num_repetitions,
          int interval,
          std::vector<stop> stops,
          std::vector<section> sections,
          bitfield traffic_days,
          int initial_train_num);

  service(config const&, specification const&);

  void verify_service();

  std::vector<std::pair<int, utl::cstr>> get_ids() const;

  int find_first_stop_at(eva_number) const;

  int get_first_stop_index_at(eva_number) const;

  int event_time(int stop_index, event_type evt) const;

  unsigned traffic_days_offset_at_stop(int stop_index, event_type) const;

  bitfield traffic_days_at_stop(int stop_index, event_type) const;

  vector<duration_t> get_stop_times() const;
  void set_stop_times(vector<duration_t> const&);
  vector<tz_offsets> get_stop_timezones(timezone_map_t const&) const;

  parser_info origin_{};
  int num_repetitions_{0};
  int interval_{0};
  std::vector<stop> stops_;
  std::vector<section> sections_;
  bitfield traffic_days_;
  int initial_train_num_{0};
};

template <typename Fn>
void expand_traffic_days(service const& s,
                         bitfield_map_t const& bitfields,
                         Fn&& consumer) {
  struct split_info {
    bitfield traffic_days_;
    unsigned from_section_idx_, to_section_idx_;
  };

  // Transform section bitfield indices into concrete bitfields.
  auto section_bitfields =
      utl::to_vec(s.sections_, [&](service::section const& section) {
        auto const it = bitfields.find(section.traffic_days_.at(0));
        utl::verify(it != end(bitfields), "bitfield {} not found",
                    section.traffic_days_.at(0));
        return it->second.first;
      });

  // Checks that all section bitfields are disjunctive and calls consumer.
  bitfield consumed_traffic_days;
  auto const check_and_consume = [&](service&& s) {
    utl::verify((s.traffic_days_ & consumed_traffic_days).none(),
                "traffic days of service {} are not disjunctive:\n"
                "    sub-sections: {}\n"
                "already consumed: {}",
                s.origin_, s.traffic_days_, consumed_traffic_days);
    consumed_traffic_days |= s.traffic_days_;
    consumer(std::move(s));
  };

  // Creates a service containing only the specified sections.
  auto const create_service_from_split = [](split_info const& split,
                                            service const& origin) -> service {
    auto const number_of_stops =
        split.to_section_idx_ - split.from_section_idx_ + 2;
    std::vector<service::stop> stops(number_of_stops);
    std::copy(std::next(begin(origin.stops_), split.from_section_idx_),
              std::next(begin(origin.stops_), split.to_section_idx_ + 2),
              begin(stops));

    auto const number_of_sections =
        split.to_section_idx_ - split.from_section_idx_ + 1;
    std::vector<service::section> sections(number_of_sections);
    std::copy(std::next(begin(origin.sections_), split.from_section_idx_),
              std::next(begin(origin.sections_), split.to_section_idx_ + 1),
              begin(sections));

    return service{origin.origin_,
                   origin.num_repetitions_,
                   origin.interval_,
                   stops,
                   sections,
                   split.traffic_days_,
                   origin.initial_train_num_};
  };

  // Removes the set bits from the section bitfields (for further iterations)
  // and writes a new services containing the specified sections [start, pos[.
  auto const consume_and_remove = [&](unsigned const start, unsigned const pos,
                                      bitfield const& current) {
    if (current.any()) {
      auto const not_current = ~current;
      for (unsigned i = start; i < pos; ++i) {
        section_bitfields[i] &= not_current;
      }
      assert(pos >= 1);
      check_and_consume(
          create_service_from_split(split_info{current, start, pos - 1}, s));
    }
  };

  // Recursive function splitting services with uniform traffic day bitfields.
  auto const split = [&](unsigned const start, unsigned const pos,
                         bitfield const& current, auto&& split) {
    if (pos == section_bitfields.size()) {
      consume_and_remove(start, pos, current);
      return;
    }

    auto const intersection = current & section_bitfields[pos];
    if (intersection.none()) {
      consume_and_remove(start, pos, current);
      return;
    }

    split(start, pos + 1, intersection, split);
    auto const diff = current & (~intersection);
    consume_and_remove(start, pos, diff);
  };

  for (auto i = size_t{0U}; i < section_bitfields.size(); ++i) {
    split(i, i, section_bitfields[i], split);
  }
}

template <typename Fn>
void expand_repetitions(service const& s, Fn&& consumer) {
  auto const update_event = [&](service::event const& origin, int interval,
                                int repetition) {
    auto const new_time = origin.time_ != service::NOT_SET
                              ? origin.time_ + (interval * repetition)
                              : service::NOT_SET;
    return service::event{new_time, origin.in_out_allowed_};
  };

  for (int rep = 1; rep <= std::max(1, s.num_repetitions_); ++rep) {
    consumer({s.origin_, 0, 0,
              utl::to_vec(begin(s.stops_), end(s.stops_),
                          [&](service::stop const& stop) {
                            return service::stop{
                                stop.eva_num_,
                                update_event(stop.arr_, s.interval_, rep),
                                update_event(stop.dep_, s.interval_, rep)};
                          }),
              s.sections_, s.traffic_days_, s.initial_train_num_});
  }
}

template <typename Fn>
void to_local_time(
    timezone_map_t const& timezones,
    std::pair<std::chrono::sys_days, std::chrono::sys_days> const& interval,
    service const& s,
    Fn&& consumer) {
  struct duration_hash {
    cista::hash_t operator()(vector<duration_t> const& v) {
      auto h = cista::BASE_HASH;
      for (auto const& el : v) {
        h = cista::hash_combine(h, el.count());
      }
      return h;
    }
  };

  auto utc_time_traffic_days =
      hash_map<vector<duration_t>, bitfield, duration_hash>{};
  auto const local_times = s.get_stop_times();
  auto const stop_timezones = s.get_stop_timezones(timezones);
  auto const& [first_day, last_day] = interval;
  auto utc_service_times = vector<duration_t>(s.stops_.size() * 2 - 2);
  for (auto day = first_day; day <= last_day; day += std::chrono::days{1}) {
    auto const day_idx = (day - first_day).count();
    if (!s.traffic_days_.test(day_idx)) {
      continue;
    }

    auto [_, first_dep_day_offset] =
        local_mam_to_utc_mam(stop_timezones.front(), day, local_times.front());

    auto i = 0;
    auto pred = duration_t{0};
    auto fail = false;
    for (auto const& [local_time, tz] : utl::zip(local_times, stop_timezones)) {
      auto const [utc_mam, day_shift] =
          local_mam_to_utc_mam(tz, day, local_time, first_dep_day_offset);
      if (day_shift != 0 || pred > utc_mam) {
        log(log_lvl::error, "nigiri.loader.hrd.service",
            "local to utc failed, ignoring: {}, day={}, day_shift={}, pred={}, "
            "utc_mam={}",
            s.origin_, day, day_shift, pred, utc_mam);
        fail = true;
        break;
      }

      utc_service_times[i++] = utc_mam;
      pred = utc_mam;
    }

    if (!fail) {
      utc_time_traffic_days[utc_service_times].set(day_idx);
    }
  }

  for (auto const& [times, traffic_days] : utc_time_traffic_days) {
    consumer(service{s, times, traffic_days});
  }
}

template <typename ConsumerFn, typename ProgressFn>
void parse_services(
    config const& c,
    char const* filename,
    std::pair<std::chrono::sys_days, std::chrono::sys_days> const& interval,
    bitfield_map_t const& bitfields,
    timezone_map_t const& timezones,
    std::string_view file_content,
    ProgressFn&& bytes_consumed,
    ConsumerFn&& consumer) {
  auto const expand_service = [&](service const& s) {
    expand_traffic_days(s, bitfields, [&](service&& s) {
      expand_repetitions(s, [&](service&& s) {
        to_local_time(timezones, interval, s, consumer);
      });
    });
  };

  specification spec;
  auto last_line = 0;
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, int line_number) {
        last_line = line_number;

        if (line_number % 1000 == 0) {
          bytes_consumed(line.c_str() - &file_content[0]);
        }

        if (line.len == 0 || line[0] == '%') {
          return;
        }

        auto const is_finished = spec.read_line(line, filename, line_number);

        if (!is_finished) {
          return;
        } else {
          spec.line_number_to_ = line_number - 1;
        }

        if (!spec.valid()) {
          log(log_lvl::error, "nigiri.loader.hrd.service",
              "skipping invalid service at {}:{}", filename, line_number);
        } else if (!spec.ignore()) {
          // Store if relevant.
          try {
            expand_service(service{c, spec});
          } catch (std::runtime_error const& e) {
            log(log_lvl::error, "unable to build service at {}:{}: {}",
                filename, line_number, e.what());
          }
        }

        // Next try! Re-read first line of next service.
        spec.reset();
        spec.read_line(line, filename, line_number);
      });

  if (!spec.is_empty() && spec.valid() && !spec.ignore()) {
    spec.line_number_to_ = last_line;
    expand_service(service{c, spec});
  }
}

template <typename ProgressFn>
void write_services(
    config const& c,
    char const* filename,
    std::pair<std::chrono::sys_days, std::chrono::sys_days> const& interval,
    bitfield_map_t const& bitfields,
    hash_map<eva_number, tz_offsets> const& timezones,
    hash_map<eva_number, location_idx_t> const& locations,
    std::string_view file_content,
    timetable& tt,
    ProgressFn&& bytes_consumed) {
  parse_services(c, filename, interval, bitfields, timezones, file_content,
                 std::forward<ProgressFn>(bytes_consumed),
                 [](service const& s) {

                 });
}

}  // namespace nigiri::loader::hrd
