#include "nigiri/logging.h"

namespace nigiri {

log_lvl s_verbosity = log_lvl::debug;

scoped_timer::scoped_timer(std::string name)
    : name_{std::move(name)}, start_{std::chrono::steady_clock::now()} {
  log(log_lvl::info, name_.c_str(), "starting {}", std::string_view{name_});
}

scoped_timer::~scoped_timer() {
  using namespace std::chrono;
  auto const stop = steady_clock::now();
  auto const t =
      static_cast<double>(duration_cast<microseconds>(stop - start_).count()) /
      1000.0;
  log(log_lvl::info, name_.c_str(), "finished {} {}ms", std::string_view{name_},
      t);
}

}  // namespace nigiri
