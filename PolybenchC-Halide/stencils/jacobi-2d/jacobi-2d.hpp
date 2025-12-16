#ifndef JAPOBI_2D_HPP
#define JAPOBI_2D_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define TSTEPS 20
# define N 30
#elif defined(SMALL_DATASET)
# define TSTEPS 40
# define N 90
#elif defined(MEDIUM_DATASET)
# define TSTEPS 100
# define N 250
#elif defined(LARGE_DATASET)
# define TSTEPS 500
# define N 1300
#elif defined(EXTRALARGE_DATASET)
# define TSTEPS 1000
# define N 2800
#endif

#endif // JAPOBI_2D_HPP
