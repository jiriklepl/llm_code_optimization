#ifndef CORRELATION_HPP
#define CORRELATION_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define M 28
# define N 32
#elif defined(SMALL_DATASET)
# define M 80
# define N 100
#elif defined(MEDIUM_DATASET)
# define M 240
# define N 260
#elif defined(LARGE_DATASET)
# define M 1200
# define N 1400
#elif defined(EXTRALARGE_DATASET)
# define M 2600
# define N 3000
#endif

#endif // CORRELATION_HPP
