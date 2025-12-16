@include ../optimization/exo_naive.prompt.md

# Model Translation Guidelines

Given a C code snippet with Exo constructs:

```c
@include  ../example/to_model/exo/gemm.c
```

With an unoptimized model description:

```
@include  ../example/to_model/gemm.model
```

And an optimized model description:

```
@include  ../example/from_model/gemm.model
```

The corresponding optimized Exo implementation might look like this:

```c
@include  ../example/from_model/exo/gemm.c
```

Be sure to apply the optimizations specified in the optimized model description while preserving the original computation semantics. If the specified optimizations prove to be inapplicable or incorrect, adjust the implementation accordingly to ensure correctness and performance improvements.
