#pragma once

#include "utl/parser/cstr.h"

#include "nigiri/loader/hrd/service/expand_local_to_utc.h"
#include "nigiri/loader/hrd/service/expand_repetitions.h"
#include "nigiri/loader/hrd/service/expand_traffic_days.h"
#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/service/service_store.h"
#include "nigiri/loader/hrd/stamm/stamm.h"

namespace nigiri::loader::hrd {

template <typename ConsumerFn, typename ProgressFn>
void parse_services(config const& c,
                    char const* filename,
                    interval<std::chrono::sys_days> const& interval,
                    service_store& store,
                    bitfield_map_t const& bitfields,
                    timezone_map_t const& timezones,
                    std::string_view file_content,
                    ProgressFn&& bytes_consumed,
                    ConsumerFn&& consumer) {
  auto const expand_service = [&](service_idx_t const s_idx) {
    expand_traffic_days(store, s_idx, bitfields, [&](ref_service const& a) {
      expand_repetitions(store, a, [&](ref_service const& b) {
        to_local_time(store, timezones, interval, b, consumer);
      });
    });
  };

  specification spec;
  auto last_line = 0U;
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, unsigned const line_number) {
        last_line = line_number;

        if (line_number % 1000 == 0) {
          bytes_consumed(static_cast<size_t>(line.c_str() - &file_content[0]));
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
          log(log_lvl::error, "loader.hrd.service",
              "skipping invalid service at {}:{}", filename, line_number);
        } else if (!spec.ignore()) {
          // Store if relevant.
          try {
            expand_service(store.add(service{c, spec}));
          } catch (std::exception const& e) {
            log(log_lvl::error, "loader.hrd.service.expand",
                "unable to build service at {}:{}: {}", filename, line_number,
                e.what());
          }
        }

        // Next try! Re-read first line of next service.
        spec.reset();
        spec.read_line(line, filename, line_number);
      });

  if (!spec.is_empty() && spec.valid() && !spec.ignore()) {
    spec.line_number_to_ = last_line;
    expand_service(store.add(service{c, spec}));
  }
}

}  // namespace nigiri::loader::hrd