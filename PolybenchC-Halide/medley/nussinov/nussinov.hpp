#ifndef NUSSINOV_HPP
#define NUSSINOV_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define N 60
#elif defined(SMALL_DATASET)
# define N 180
#elif defined(MEDIUM_DATASET)
# define N 500
#elif defined(LARGE_DATASET)
# define N 2500
#elif defined(EXTRALARGE_DATASET)
# define N 5500
#endif

#endif // NUSSINOV_HPP
