PCB (Penalized Component Blends)

## Overview

From the matrix factorization X=svd(X)≈UDV' where X is input data matrix with size m by n, svd(⋅) is a singular value decomposition and U, D and V' are truncated SVD triplet with component number k (so the size of U is m by k, V' is k by n and D is a diagonal matrix with size k by k), we set W = UM, H = NV' and want to find M and N matrices with sizes of k by p and p by k respectively which make the product M*N as close as D and both W and H as sparse as possible.


## Purpose & Motivation

What problem does this solve?
- keep only the top-p singular values to project high-dimensional data into a lower-dimensional space
- decomposes a input data matrix to find sparse hidden semantic structure.
How does it differ from existing methods?
- Different from conventional matrix factorization methods which optimize W and H with big size directly, this approach optimzes small size M and N matrices.


## Idea & Concept

Core Idea
- To achive the goal making the product MN as close as D and both W and H are as sparse as possible, we propose to minimize below objective function
  f(M,N) = norm(M*N-D,2)^2 + αₘ*rL1(U*M,σₘ)*norm(N,2) + αₙ*rL1(N*V',σₙ)*norm(M,2)
  where rL1(A,σ) = sum(sqrt.(A.^2 .+ σ^2)) is similar to L1 norm but different in that it relaxs the sharpeness of the kinked point of L1 norm with parameter σ, where A is a matrix σ is a scalar value and .^ and .+ are julia broadcast operations.
  norm(A,2) is a L2 norm of a matrix A.
  And, αₘ and αₙ are user specified control parameters to control sparsity amount.
  And, σₘ and σₙ are decreased from the user specified initial value σₘ₀ and σₙ₀ as iteration goes.

Pseudocode
- Specify αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol.
- Perform SVD to get triplet U, D, V from the input data X
- Calculate Dsq = sqrt.(D) where sqrt.(⋅) is a julia broadcasted sqrt function.
- Calculate W₀ = U*Dsq and H₀ = Dsq*V'
- Initialize σₘ with σ₀^2*var(W₀) and σₙ with σ₀^2*var(H₀) where var(A) is a function to calculate variance of A.
- Initialize M with M₀=U'W₀ and N with N₀ = H₀*V. Mprev = M and Nprev = N. fprev = f(M,N)
- i = 0; converged = false
- while !converged && i < maxiter
    i += 1
    <!-- optimize for M and N -->
    M, N, fval = optimize(f; M, N, U, V, D, αₘ, αₙ, σₘ, σₙ, inner_maxiter, inner_tol)
    <!-- check the convergence -->
    converged = check_reldiff(Mprev,Nprev,M,N; tol) <!-- check relative change of parameter -->
    converged |= check_reldiff(fprev,fval; tol) <!-- check relative change of objective value -->
    <!-- update σₘ and σₙ -->
    σₘ *= r^2
    σₙ *= r^2
    <!-- update Mpre, Npre and fprev -->
    Mprev, Nprev, fprev = copy(M), copy(N), f
  end
- return U*M, N*V', converged, i

## Julia Implementation

julia"""
    pcb(X, p; αₘ=1e-3, αₙ=1e-3, σ₀=1, r=0.3, tol=1e-6, maxiter=100, inner_tol=1e-6, inner_maxiter=10000)

    This penalizes component blending.

# Arguments
- `x`: Input data
- `p` : number of component
- `αₘ` : Sparsity parameter for W factor (default: 1e-3)
- `αₙ` : Sparsity paramter for H factor (default: 1e-3)
- `σ₀` : Control paramter for L1 relaxation (default: 1)
- `r`: Decreasing ratio of relaxation paramters σₘ and σₙ (default: 0.3)
- `tol`: Tolerance threshold (default: 1e-6)
- `maxiter`: Maximum iterations (default: 1000)
- `inner_tol`: Tolerance threshold of inner optimization step (default: 1e-6)
- `inner_maxiter`: Maximum iterations of inner optimization step (default: 1000)

# Returns
- `W` : W factor
- `H` : H factor 
- `converged` : Convergence result
- `i` : final number of iteration

# Examples
```julia
W, H, converged, iter = pcb(X, 15)
```
"""

## Test Plan

using Test
Wgt = rand(10,5); Hgt = rand(5,20); X = Wgt*Hgt 
@testset "pcb Tests" begin
    # data similarity test
    W, H, converged, iter = pcb(X,5)
    @test W*H ≈ X

    # sparsity test
    #= test W and H sparsity here =#
end

## Performance Analysis

Time Complexity

Best case: O(?)
Average case: O(?)
Worst case: O(?)

Benchmark (optional)
juliausing BenchmarkTools
@benchmark my_algorithm(sample_input)

## Key Directories

- `Project.toml` and `Manifest.toml` link to directories with the packages

## Standards

- it should be possible to play with the result at the Julia command line so
  that I can examine performance and run benchmarking
- extensive source code should be placed in files separate from "scripts" which
  contain the kind of commands that might be executed interactively

## Limitations & Caveats


# References

Papers
- Matthew Brand. Incremental singular value decomposition of uncertain data with missing values. In Computer Vision—ECCV 2002: 7th European Conference on Computer Vision Copenhagen, Denmark, May 28–31, 2002 Proceedings, Part I 7, pages 707–720. Springer, 2002.
- Chen F and Rohe K, “A New Basis for Sparse  Principal Component Analysis.”, Journal of Computational and Graphical Statistics, 2023.
- Mariano Tepper and Guillermo Sapiro. Compressed nonnegative matrix factorization is fast and accurate. IEEE Transactions on Signal Processing, 64(9):2269–2283, May 2016. doi: 10.1109/tsp.2016. 2516971. URL https://doi.org/10.1109%2Ftsp.2016.2516971.
- Andrzej Cichocki and Anh-Huy Phan. Fast local algorithms for large scale nonnegative matrix and tensor factorizations. IEICE transactions on fundamentals of electronics, communications and computer sciences, 92(3):708–721, 2009.
- Yangyang Xu, Wotao Yin, Zaiwen Wen, and Yin Zhang. An alternating direction algorithm for matrix completion with nonnegative factors. Frontiers of Mathematics in China, 7(2):365–384, apr 2012. doi: 10.1007/s11464-012-0194-5. URL https://doi.org/10.1007% 2Fs11464-012-0194-5.


# Notes & Future Ideas
