#include <filesystem>

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "utl/progress_tracker.h"

#include "test_dir.h"

int main(int argc, char** argv) {
  utl::get_active_progress_tracker_or_activate("test");
  std::filesystem::current_path(NIGIRI_TEST_EXECUTION_DIR);
  return doctest::Context(argc, argv).run();
}
