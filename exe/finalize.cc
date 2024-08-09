#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

using namespace nigiri;

int main() {
  auto tt = timetable::read(
      cista::memory_holder{cista::file{"tt.bin", "r"}.content()});
  loader::finalize(*tt);
}