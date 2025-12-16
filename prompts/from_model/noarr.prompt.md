@include ../optimization/noarr_naive.prompt.md

# Model Translation Guidelines

Given a Noarr code snippet:

```cpp
@include  ../example/to_model/noarr/gemm.cpp
```

With an unoptimized model description:

```
@include  ../example/to_model/gemm.model
```

And an optimized model description:

```
@include  ../example/from_model/gemm.model
```

The corresponding optimized Noarr implementation might look like this:

```cpp
@include  ../example/from_model/noarr/gemm.cpp
```

Be sure to apply the optimizations specified in the optimized model description while preserving the original computation semantics. If the specified optimizations prove to be inapplicable or incorrect, adjust the implementation accordingly to ensure correctness and performance improvements.
