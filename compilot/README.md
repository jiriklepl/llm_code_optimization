# Reproducing the Paper's Baseline

We have measured the Tiramisu baseline from the original paper. In our setup, we used an environment defined in [Dockerfile.compilot](./Dockerfile.compilot) and ran it using **Charliecloud** (a containerization platform that promises no runtime overhead, though it offers less isolation than alternatives).

We will first explain how to run and measure the baseline using standard Docker, and then discuss the differences when using Charliecloud.

Inside the container, we install the necessary dependencies, check out the specific version of Tiramisu used in the paper, and compile it. Once complete, you can connect to the container and run your experiments.

## Using Docker

### Build and Connect

First, build the Docker image using the provided `Dockerfile.compilot`. We will tag the image as `tiramisu_baseline` for easy reference. Run the following command from the root of the repository:

```bash
$> docker build -t tiramisu_baseline -f Dockerfile.compilot .
```

Once the build is complete, you can start the container and connect to it interactively. We use the `-it` flag to ensure we get an interactive terminal shell.

```bash
$> docker run -it --rm --name tiramisu_run tiramisu_baseline /bin/bash
```

You are now inside the container shell with the environment prepared.

### Instructions within the Container

Now, everything is ready for compiling and running selected benchmarks. The benchmarks we used are located in `/tiramisu/benchmarks/autoscheduler_benchmarks/polybench`. Variants are split first by size, and then by individual benchmark.

For example, to run the **MINI** version of GEMM, run the following:

```bash
$> cd /tiramisu/benchmarks
$> ./compile_and_run_benchmarks.sh autoscheduler_benchmarks/polybench/MINI/function_gemm_MINI function_gemm_MINI
# ... output messages ...
# ... runtime in ms will appear near the end ...
```

This will produce a series of measured rounds. By default, the algorithm runs 30 times. You can change the number of executions by setting an environment variable:

```bash
$> export NB_EXEC=<number_of_runs>
```

## Running on Charliecloud

On Charliecloud, you must first build the Dockerfile, transfer it to an image (a read-only directory), and then bind the writable folders when connecting. If you are using **Slurm** to manage your cluster, all of this must be done on a worker node.

```bash
$> ch-image build -f Dockerfile.compilot .
$> ch-convert -i ch-image -o dir compilot imgdir
```

This will build the image. In the current directory, you should now have a folder named `imgdir` containing the image's filesystem.

Since the image is read-only, you need a writable copy of the repository. Copy the repository from inside the container's `imgdir` to your working directory:

```bash
cp -r ./imgdir/tiramisu ./copied-tiramisu
```

Then, you can connect to the image as you would in Docker:

```bash
$> ch-run imgdir -b ./copied-tiramisu/:/tiramisu -- /bin/bash
```

You are now connected to the image, and your editable `copied-tiramisu` directory is mounted to `/tiramisu`.

**Important Note:** As Charliecloud is lightweight and only abstracts the filesystem, you must manually set environment variables to their expected values:

```bash
$> export TIRAMISU_ROOT=${TIRAMISU_DIR:-/tiramisu}
$> export LD_LIBRARY_PATH=/usr/local/lib:/tiramisu/build:/tiramisu/3rdParty/isl/lib:/tiramisu/3rdParty/llvm/lib
# Note: Check the Dockerfile to see if other environment variables need to be set.
```

Be aware that some variables from your host environment will leak into the container, which may cause unexpected errors. This is a common issue when working with Charliecloud.

After setting up the environment, you are ready to run the benchmarks. Remember that you will need to apply the fix to `compile_and_run_benchmarks.sh` within the `copied-tiramisu` folder (see the "Instructions within the Container" section above).

```bash
$> cd /tiramisu/benchmarks
$> ./compile_and_run_benchmarks.sh autoscheduler_benchmarks/polybench/MINI/function_gemm_MINI function_gemm_MINI
```

The execution process is the same as in the Docker setup.
