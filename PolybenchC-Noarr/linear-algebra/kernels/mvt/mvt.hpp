#ifndef MVT_HPP
#define MVT_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define N 40
#elif defined(SMALL_DATASET)
# define N 120
#elif defined(MEDIUM_DATASET)
# define N 400
#elif defined(LARGE_DATASET)
# define N 2000
#elif defined(EXTRALARGE_DATASET)
# define N 4000
#endif

#endif // MVT_HPP
