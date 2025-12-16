#ifndef GESUMMV_HPP
#define GESUMMV_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define N 30
#elif defined(SMALL_DATASET)
# define N 90
#elif defined(MEDIUM_DATASET)
# define N 250
#elif defined(LARGE_DATASET)
# define N 1300
#elif defined(EXTRALARGE_DATASET)
# define N 2800
#endif

#endif // GESUMMV_HPP