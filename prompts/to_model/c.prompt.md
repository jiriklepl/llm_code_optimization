# Role
Your goal is to analyze the given C code and produce an optimized model description that captures its semantics and optimization opportunities.

@include _systeminfo.prompt.md

# Modeling Guidelines
Carefully examine the provided C source to understand its data layout, loop structure, and computation, then translate those semantics into a model representation that exposes optimization opportunities.

Given a C code snippet:

```c
@include  ../example/to_model/c/gemm.c
```

An unoptimized model description might look like this:

```
@include  ../example/to_model/gemm.model
```

@include _specification.prompt.md

# Task
Your task is to provide an optimized model description that improves performance. A model description that simply performs dimension blocking (tiling), changing the layout of one of the matrices, and performs loop interchange via hoisting inner loops to the outermost level might look like this:

```
@include  ../example/from_model/gemm.model
```

# Optimization Hints
Choose from the following optimization hints any that you deem appropriate for optimizing the given code. You may choose none, some, or all of them.

@include ../optimization/hints/_arithmetic_hint.prompt.md
@include ../optimization/hints/_cache_hint.prompt.md
@include ../optimization/hints/_structure_hint.prompt.md
@include ../optimization/hints/_parallelism_hint.prompt.md

# Verbosity and Reasoning
Write a model description that you are confident preserves the original functionality while optimizing performance. Before outputting the final model description, double-check its semantics against the original code and iteratively refine the draft until it meets the expected functionality and optimization goals. The resulting model description should be well-commented and explicit. Any unobvious optimization techniques should be accompanied by comments that clearly explain their semantics and purpose. Interpretability of the resulting model description is essential.
