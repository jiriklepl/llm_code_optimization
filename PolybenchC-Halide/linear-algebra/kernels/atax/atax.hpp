#ifndef ATAX_HPP
#define ATAX_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define M 38
# define N 42
#elif defined(SMALL_DATASET)
# define M 116
# define N 124
#elif defined(MEDIUM_DATASET)
# define M 390
# define N 410
#elif defined(LARGE_DATASET)
# define M 1900
# define N 2100
#elif defined(EXTRALARGE_DATASET)
# define M 1800
# define N 2200
#endif

#endif // ATAX_HPP
