#include "nigiri/common/indent.h"

#include <ostream>

namespace nigiri {

void indent(std::ostream& out, unsigned const n) {
  for (auto i = 0U; i != n; ++i) {
    out << "  ";
  }
}

}  // namespace nigiri