#ifndef ADI_HPP
#define ADI_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define TSTEPS 20
# define N 20
#elif defined(SMALL_DATASET)
# define TSTEPS 40
# define N 60
#elif defined(MEDIUM_DATASET)
# define TSTEPS 100
# define N 200
#elif defined(LARGE_DATASET)
# define TSTEPS 500
# define N 1000
#elif defined(EXTRALARGE_DATASET)
# define TSTEPS 1000
# define N 2000
#endif

#endif // ADI_HPP
