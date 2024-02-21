#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "fmt/core.h"
#include "fmt/ostream.h"

namespace nigiri {

enum class log_lvl { debug, info, error };

constexpr char const* to_str(log_lvl const lvl) {
  switch (lvl) {
    case log_lvl::debug: return "debug";
    case log_lvl::info: return "info";
    case log_lvl::error: return "error";
  }
  return "";
}

static log_lvl s_verbosity;

inline std::string now() {
  using clock = std::chrono::system_clock;
  auto const now = clock::to_time_t(clock::now());
  struct tm tmp {};
#if _MSC_VER >= 1400
  gmtime_s(&tmp, &now);
#else
  gmtime_r(&now, &tmp);
#endif

  std::stringstream ss;
  ss << std::put_time(&tmp, "%FT%TZ");
  return ss.str();
}

#ifndef NIGIRI_LOG_HEADER
template <typename... Args>
void log(log_lvl const lvl,
         char const* ctx,
         fmt::format_string<Args...> fmt_str,
         Args&&... args) {
  if (lvl >= ::nigiri::s_verbosity) {
    fmt::print(
        std::clog, "{time} | [{lvl}][{ctx:30}] {msg}\n",
        fmt::arg("time", now()), fmt::arg("lvl", to_str(lvl)),
        fmt::arg("ctx", ctx),
        fmt::arg("msg", fmt::format(fmt_str, std::forward<Args>(args)...)));
  }
}
#else
#include NIGIRI_LOG_HEADER
#endif

struct scoped_timer final {
  explicit scoped_timer(std::string name);
  scoped_timer(scoped_timer const&) = delete;
  scoped_timer(scoped_timer&&) = delete;
  scoped_timer& operator=(scoped_timer const&) = delete;
  scoped_timer& operator=(scoped_timer&&) = delete;
  ~scoped_timer();

  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
};

}  // namespace nigiri
