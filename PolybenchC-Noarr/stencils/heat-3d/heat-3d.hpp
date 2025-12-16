#ifndef HEAT_3D_HPP_
#define HEAT_3D_HPP_

#include "defines.hpp"

#ifdef MINI_DATASET
# define TSTEPS 20
# define N 10
#elif defined(SMALL_DATASET)
# define TSTEPS 40
# define N 20
#elif defined(MEDIUM_DATASET)
# define TSTEPS 100
# define N 40
#elif defined(LARGE_DATASET)
# define TSTEPS 500
# define N 120
#elif defined(EXTRALARGE_DATASET)
# define TSTEPS 1000
# define N 200
#endif

#endif // HEAT_3D_HPP_
