#ifndef ATAX_HPP
#define ATAX_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define N 42
# define M 38
#elif defined(SMALL_DATASET)
# define N 124
# define M 116
#elif defined(MEDIUM_DATASET)
# define N 410
# define M 390
#elif defined(LARGE_DATASET)
# define N 2100
# define M 1900
#elif defined(EXTRALARGE_DATASET)
# define N 2200
# define M 1800
#endif

#endif // ATAX_HPP
