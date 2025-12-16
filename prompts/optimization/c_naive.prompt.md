# Role
Your goal is to produce an optimized version of the given C code.

@include _systeminfo.prompt.md

# Optimization Guidelines
When optimizing C code, adhere to the following guidelines:

1. Preserve the original include directives and add any necessary headers for optimization techniques used.

2. Ensure all loop transformations and optimizations preserve the original computation semantics. The provided dimensions may not be divisible by common tiling factors; handle edge cases appropriately.

The resulting code must be valid, compilable C code.

# Task
Carefully examine the code and produce an optimized version focusing on improving data locality, computational efficiency, and memory access patterns while preserving the original functionality of the code. Keep the boilerplate code used for setting up the data structures and measuring execution time intact. The resulting code must be a valid C program without any omissions, and it must preserve the original boilerplate code used for setting up the data structures and measuring execution time.

# Verbosity and Reasoning
Write code that you are confident preserves the original functionality while optimizing performance. Before outputting the final code, double-check its semantics against the original code and iteratively refine the draft until it meets the expected functionality and optimization goals. The resulting code should be well-commented and explicit. Any unobvious optimization techniques should be accompanied by comments that clearly explain their semantics and purpose. Interpretability of the resulting code is essential.
