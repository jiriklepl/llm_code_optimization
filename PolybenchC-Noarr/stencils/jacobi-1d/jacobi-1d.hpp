#ifndef JACOBI_1D_HPP
#define JACOBI_1D_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
# define TSTEPS 20
# define N 30
#elif defined(SMALL_DATASET)
# define TSTEPS 40
# define N 120
#elif defined(MEDIUM_DATASET)
# define TSTEPS 100
# define N 400
#elif defined(LARGE_DATASET)
# define TSTEPS 500
# define N 2000
#elif defined(EXTRALARGE_DATASET)
# define TSTEPS 1000
# define N 4000
#endif

#endif // JACOBI_1D_HPP
