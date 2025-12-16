@include ../optimization/halide_naive.prompt.md

# Model Translation Guidelines

Given a Halide C++ code snippet:

```cpp
@include  ../example/to_model/halide/gemm.cpp
```

With an unoptimized model description:

```
@include  ../example/to_model/gemm.model
```

And an optimized model description:

```
@include  ../example/from_model/gemm.model
```

The corresponding optimized Halide implementation might look like this:

```cpp
@include  ../example/from_model/halide/gemm.cpp
```

Be sure to apply the optimizations specified in the optimized model description while preserving the original computation semantics. If the specified optimizations prove to be inapplicable or incorrect, adjust the implementation accordingly to ensure correctness and performance improvements.
