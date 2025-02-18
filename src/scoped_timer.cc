#include "utl/logging.h"

#include "nigiri/scoped_timer.h"

namespace nigiri {

scoped_timer::scoped_timer(std::string name)
    : name_{std::move(name)}, start_{std::chrono::steady_clock::now()} {
  utl::log_info(name_.c_str(), "starting {}", std::string_view{name_});
}

scoped_timer::~scoped_timer() {
  using namespace std::chrono;
  auto const stop = steady_clock::now();
  auto const t =
      static_cast<double>(duration_cast<microseconds>(stop - start_).count()) /
      1000.0;
  utl::log_info(name_.c_str(), "finished {} {}ms", std::string_view{name_}, t);
}

}  // namespace nigiri
