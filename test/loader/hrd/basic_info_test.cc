#include "doctest/doctest.h"

#include "nigiri/loader/hrd/basic_info.h"
#include "nigiri/loader/hrd/parser_config.h"

using namespace nigiri::loader::hrd;

constexpr auto const file_content =
    "14.12.2014\n"
    "12.12.2015\n"
    "JF064 EVA_ABN~RIS Server~RIS OEV IMM~~J15~064_001 000000 END\n";

TEST_CASE("loader_hrd_basic_info, simple_interval") {
  auto const [a, b] = parse_interval(file_content);
  CHECK_EQ(1418515200, std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::sys_days{a}.time_since_epoch())
                           .count());
  CHECK_EQ(1449878400, std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::sys_days{b}.time_since_epoch())
                           .count());
}

TEST_CASE("loader_hrd_basic_info, schedule_name") {
  auto name = parse_schedule_name(file_content);
  CHECK_EQ("JF064 EVA_ABN~RIS Server~RIS OEV IMM~~J15~064_001 000000 END",
           name);
}
