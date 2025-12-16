# Role
Your goal is to produce an optimized version of the given C++ code with the Noarr library.

@include _systeminfo.prompt.md

# Noarr Overview
A detailed description of Noarr is available in [docs/README.md](docs/README.md).

@list_files ../../submodules/noarr-structures/docs docs *.md

# Optimization Guidelines
When optimizing C++ code using Noarr, adhere to the following guidelines:

1. All loop nests must be expressed using Noarr traversers with the `for_each([&](auto index_state){ ... })` or `for_dims<'DIM_NAME', ...>([&](auto inner_traverser){ ... })` constructs. Alternatively, you may use Noarr planners, which offer similar traversal functionalities but require explicit invocation through the `execute()` method.

2. Access elements using `A[index_state]` or `A[inner_traverser]`, depending on whether you are using an index state or an inner traverser. You can perform arithmetic on index states, such as `index_state + noarr::idx<'DIM_NAME'>(value)`.

3. Generally, Noarr bags are not copyable; however, they offer a `.get_ref()` method that returns a copyable observer bag. This applies to use cases where a bag is an argument to the `^` operator that constructs a new view on its data.

4. Do not use index state arithmetic operations on traversers directly; instead, retrieve their `.state()` and perform arithmetic on that.

5. Do not use multi-character dimension names; To extend the naming beyond single characters, use instances of `noarr::dim<auto Tag>` instead of character literals. Tags can be arbitrary objects of literal (semi-)regular types. `noarr::dim<Tag1> == noarr::dim<Tag2>` if and only if `Tag1` and `Tag2` are objects of the same type and `Tag1 == Tag2`. By using readable tag objects or naming the constexpr `dim` variables appropriately, you can achieve clarity similar to multi-character dimension names without violating Noarr's constraints.

6. Ensure all loop transformations and optimizations preserve the original computation semantics. The provided dimensions may not be divisible by common tiling factors; handle edge cases appropriately.

The resulting code must be valid, compilable C++ code with Noarr includes and constructs.

# Task
Carefully examine the code and produce an optimized version focusing on improving data locality, computational efficiency, and memory access patterns while preserving the original functionality of the code. Keep the boilerplate code used for setting up Noarr data structures and measuring execution time intact. The resulting code must be a valid C++ program with Noarr constructs and without any omissions, and it must preserve the original boilerplate code used for setting up the data structures and measuring execution time. The computation kernel must be implemented using Noarr abstractions such as traversers or planners.

# Verbosity and Reasoning
Write code that you are confident preserves the original functionality while optimizing performance. Before outputting the final code, double-check its semantics against the original code and iteratively refine the draft until it meets the expected functionality and optimization goals. The resulting code should be well-commented and explicit. Any unobvious optimization techniques should be accompanied by comments that clearly explain their semantics and purpose. Interpretability of the resulting code is essential.
