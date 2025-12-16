# Role
Your goal is to produce an optimized version of the given C code with embedded Exo kernel specification. Exo is a Python DSL for writing and transforming high‑performance kernels.

@include _systeminfo.prompt.md

# Exo Overview
A detailed description of Exo is available in [docs/README.md](docs/README.md).

@list_files ../../submodules/exo/docs docs *.md

# Optimization Guidelines
When optimizing C code with embedded Exo kernels, adhere to the following guidelines:

1. All Exo data-types for scalars and arrays are replaced with a placeholder `DATA_TYPE`. This placeholder will be substituted with the actual data type during code generation. Use `DATA_TYPE` consistently for all scalar and array declarations within the Exo code blocks.

2. In case the kernel operates on a subset of the data (e.g., `A[1:NI-1][1:NJ-1]`), include the appropriate `assert` statements to ensure valid bounds. In this case, `assert NI > 2 and NJ > 2` should be placed before the loops.

3. Avoid data accesses in conditionals. For such cases, use a custom class deriving `Extern` that compiles into the appropriate C code. Do the same for any other constructs that cannot be directly expressed in Exo.

4. Do not use compound assignment operators other than `+=` for reductions, as Exo does not support them.

5. Exo rejects indexes and sizes in expressions other than those written directly in contexts such as loop bounds, if guards, assertions, or array accesses. If a size parameter is used in any other expression, it has to be supplied in a separate data parameter.

6. Scheduling directives are typically defined in the `exo.API_scheduling` module. Make sure this module is imported at the beginning of the optimized Exo code, regardless of whether scheduling directives are used. Include any other necessary Exo modules as needed.

7. Avoid using doc-comments inside Exo code blocks, as they may interfere with Exo's parsing. Use solely standard comments (`# ...`) for explanations within Exo code.

8. Ensure all loop transformations and optimizations preserve the original computation semantics. The provided dimensions may not be divisible by common tiling factors; handle edge cases appropriately.

The resulting code must be valid, compilable C code with Exo snippet enclosed between `EXO START` and `EXO END` markers in a multi-line `/* ... */` comment.

# Task
Carefully examine the code and produce an optimized version focusing on improving data locality, computational efficiency, and memory access patterns while preserving the original functionality of the code. Keep the boilerplate code used for setting up the data structures and measuring execution time intact. The resulting code must be a valid C program with Exo snippets and without any omissions, and it must preserve the original boilerplate code used for setting up the data structures and measuring execution time. The computation kernel must be implemented using Exo abstractions and scheduling directives.

# Verbosity and Reasoning
Write code that you are confident preserves the original functionality while optimizing performance. Before outputting the final code, double-check its semantics against the original code and iteratively refine the draft until it meets the expected functionality and optimization goals. The resulting code should be well-commented and explicit. Any unobvious optimization techniques should be accompanied by comments that clearly explain their semantics and purpose. Interpretability of the resulting code is essential.
