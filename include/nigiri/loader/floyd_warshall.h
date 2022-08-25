#pragma once

#include "utl/verify.h"

#include "nigiri/types.h"

namespace nigiri {

template <typename T>
void floyd_warshall(matrix<T>& mat) {
  utl::verify(mat.entries_.size() == mat.n_columns_ * mat.n_columns_,
              "floyd_warshall: input is not a square matrix.");
  constexpr auto const kMaxDistance = std::numeric_limits<T>::max();

  for (auto k = 0U; k < mat.n_columns_; ++k) {
    for (auto i = 0U; i < mat.n_columns_; ++i) {
      for (auto j = 0U; j < mat.n_columns_; ++j) {
        auto const distance =
            static_cast<T>(std::min(kMaxDistance, mat(i, k) + mat(k, j)));
        if (mat(i, j) > distance) {
          mat(i, j) = distance;
        }
      }
    }
  }
}

}  // namespace nigiri
