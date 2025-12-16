The algorithm is described using the following sections (some optional):
- `domain`: defines the dimensions and their sizes.
- `data`: defines the dimensionalities of the data structures, their scalar types and input/output roles, and their memory layouts specified via an affine expression. This can also include scalar data structures.
- `parameters`: defines any further scalar parameters used in the computation.
- `computation`: defines the computation steps
  - `for` directives express that a computation applies to a certain domain
    - `in seq(INTERVAL)` specifies sequential execution over an interval
    - `in INTERVAL` specifies unordered execution over an interval (implicit for domain dimensions)
  - `if CONDITION` directives express conditional execution
  - *assignments* express updates to data structures, using mathematical expressions over data structures and parameters; `sum(ITERATOR, EXPRESSION)` expresses reduction over an iterator where the iterator is specified similarly to `for` directives.
  - Each assignment and directive can be nested within another directive to express imperfectly nested loops and conditionals.
    - The nesting structure is specified via indentation.
  - If multiple assignments access the same data structure, sequential execution is implied.
  - Each directive and assignment can be prefixed by `LABEL:` to assign it a label `LABEL` that can be referenced elsewhere.
- `dependencies`: specifies any dependencies between computation steps.
  - `for i in [0, N), j in [0, M): assignment1(i) -> assignment2(i, j)` specifies that `assignment2` cannot execute for indices `i, j` until `assignment1` has executed for index `i`.
- `optimization`: suggests optimization strategies that can be applied to the computation.
  - Each optimization strategy is specified as a separate directive.
  - Most common optimization strategies include:
    - `block(DIMENSION, TILE_DIMENSION, TILE_SIZE)`: expresses blocking (tiling) of `DIMENSION` into tiles of size `TILE_SIZE` indexed by `TILE_DIMENSION`.
    - `interchange(DIMENSION1, DIMENSION2)`: expresses loop interchange between `DIMENSION1` and `DIMENSION2`.
    - `hoist(DIMENSION)`: expresses hoisting of `DIMENSION` to the outermost possible loop level.
    - `unroll(DIMENSION, FACTOR)`: expresses unrolling of `DIMENSION` by a factor of `FACTOR`.
    - `vectorize(DIMENSION, FACTOR)`: expresses vectorization of `DIMENSION` by a factor of `FACTOR`.
    - `parallelize(DIMENSION)`: expresses parallelization of `DIMENSION`.

The model description can be extended with more data structures, dimensions, parameters, and computation steps as needed to express optimizations. Further types of optimization strategies can be defined as needed, but they must be clearly explained via #-prefixed comments.
