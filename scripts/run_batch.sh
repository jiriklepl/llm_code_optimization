#!/bin/bash

#SBATCH --partition=mpi-homo-long
#SBATCH --output=batch-%j.out
#SBATCH --error=batch-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=0

"$@"
