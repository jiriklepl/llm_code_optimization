#ifndef TRMM_HPP
#define TRMM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define M 20
# define N 30
#elif defined(SMALL_DATASET)
# define M 60
# define N 80
#elif defined(MEDIUM_DATASET)
# define M 200
# define N 240
#elif defined(LARGE_DATASET)
# define M 1000
# define N 1200
#elif defined(EXTRALARGE_DATASET)
# define M 2000
# define N 2600
#endif

#endif // TRMM_HPP
