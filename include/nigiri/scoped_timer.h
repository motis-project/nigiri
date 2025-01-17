#pragma once

#include <chrono>

namespace nigiri {

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