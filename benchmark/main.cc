#include <iostream>

#include "utl/progress_tracker.h"

int main(int argc, char* argv[]) {
  auto const progress_tracker = utl::activate_progress_tracker("server");
  auto const silencer = utl::global_progress_bars{true};

  if (argc != 2) {
    std::cout << "usage: nigiri"
  }

  return 0;
}