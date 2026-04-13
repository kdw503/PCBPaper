PCB (Penalized Component Blends)

# Overview

From the matrix factorization X=svd(X)≈UDV' where X is input data matrix with size m by n, svd(⋅) is a singular value decomposition and U, D and V' are truncated SVD triplet with component number r (so the size of U is m by r, V' is r by n and D is a diagonal matrix with size r by r), we set W = UM, H = NV' and want to find M and N matrices with sizes of r by p and p by r respectively which make the product MN as close as D and both W and H as sparse as possible.


# Purpose & Motivation

What problem does this solve?
- keep only the top-p singular values to project high-dimensional data into a lower-dimensional space
- decomposes a input data matrix to find sparse hidden semantic structure.
How does it differ from existing methods?
- Different from conventional matrix factorization methods which optimize W and H with big size directly, this approach optimzes small size M and N matrices.


# Idea & Concept
Core Idea
- To achive the goal making the product MN as close as D and both W and H are as sparse as possible, we propose to minimize below objective function
  f(M,N) = norm(M*N-D,2)^2 + αₘ*r(U*M,σₘ)*norm(N,2) + αₙ*r(N*V',σₙ)*norm(M,2)
  where r(A,σ) = sum(sqrt.(A.^2 .+ σ^2)) is similar to L1 norm but different in that it relaxs the sharpeness of the kinked point of L1 norm with parameter σ, where A is a matrix σ is a scalar value and .^ and .+ are julia broadcast operations.
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
- i = 0
- while !converged || i < maxiter
    i += 1
    # fix N and optimize for M
    M, _ = optimize(f; N, U, V, D, αₘ, αₙ, σₘ, σₙ, inner_maxiter, inner_tol)
    # fix M and optimize for N
    N, fval = optimize(f; M, U, V, D, αₘ, αₙ, σₘ, σₙ, inner_maxiter, inner_tol)
    # check the convergence
    converged = check_reldiff(Mprev,M,Nprev,N; tol) # check relative change of parameter
    converged |= check_reldiff(fprev,fval; tol) # check relative change of objective value
  end
- return M, N, i

# Test Plan
Test Cases
juliausing Test

@testset "MyAlgorithm Tests" begin
    # Basic case
    @test my_algorithm(input1) ≈ expected1

    # Boundary case
    @test my_algorithm(boundary_input) ≈ expected2

    # Edge case / error handling
    @test_throws ErrorType my_algorithm(invalid_input)
end

Test Case Table
CaseInputExpected OutputNotesBasic......Boundary......Error/Edge......Should throw error

# Performance Analysis
Time Complexity

Best case: O(?)
Average case: O(?)
Worst case: O(?)

Space Complexity

O(?)

Benchmark (optional)
juliausing BenchmarkTools
@benchmark my_algorithm(sample_input)

# Limitations & Caveats

Cases where the algorithm may not perform well
Numerical stability concerns
Known constraints or edge cases to watch out for


# References

Papers, blog posts, documentation, etc.


# Notes & Future Ideas







# Project Context

This is a proof-of-principle project. Reusability is not a driving goal, but getting something working quickly is.

## About This Project

The goal of this project is to implement a small neural network on a test
problem, MNIST digit classification, in which the "synaptic weight matrices" are
constrained to be of Gaussian form. Specifically, for a particular layer with
weight matrix `W`, we require

    W[i, j] = a[i] * Ω[i, j]

where

    Ω[i, j] = exp(logoverlap(ginput[i]::Gaussian, goutput[j]::Gaussian))

and `a[i]` is a real value. The chain rule converts gradients expressed with `W`
into gradients in the parameters of the Gaussians and the `a`s.

Use of any particular package is optional, but Lux and GaussianEmbeddings are
provided as plausible components. An advantage of using GaussianEmbeddings is
that it provides tools for optimizing the Gaussians in ways that guarantee
positive-definitenesss of the covariances and supports covariance regularization
through a parameter `τ`. Nevertheless if it is easier to implement from scratch,
do so.

## Key Directories

- `Project.toml` and `Manifest.toml` link to directories with the packages
- `GaussianEmbeddings/src` has code for GaussianEmbeddings:
    + `gaussians.jl` defines the `Gaussian` type
    + `core.jl` implements basic computations with Gaussians
    + `linalg.jl` defines the `ParamVector` type which gets optimized
    + `coembed.jl` contains the code for "co-embedding" which is the concept
      used for the synaptic weight matrices. In particular, `run_optimize`
      illustrates the optimization loop for fitting a fixed matrix
      with Gaussians (though use of `stochasticlm` is not recommended, use
      standard ML techniques for the optimization)

## Standards

- it should be possible to play with the result at the Julia command line so
  that I can examine performance and run benchmarking
- extensive source code should be placed in files separate from "scripts" which
  contain the kind of commands that might be executed interactively