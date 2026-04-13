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
  f(M,N) = norm(M*N-D,2)^2 + αₘ*norm(U*M,1)*norm(N,2) + αₙ*norm(N*V',1)*norm(M,2)
  norm(A,1) is a L1 norm and norm(A,2) is a L2 norm of a matrix A.
  And, αₘ and αₙ are user specified control parameters to control sparsity amount.

## Julia Implementation

julia"""
    pcb(X, p; pcb_method=:L1, αₘ=1e-3, αₙ=1e-3, tol=1e-6, maxiter=100)

    This penalizes component blending.

# Arguments
- `x`: Input data
- `p` : number of component
- `αₘ` : Sparsity parameter for W factor (default: 1e-3)
- `αₙ` : Sparsity paramter for H factor (default: 1e-3)
- `tol`: Tolerance threshold (default: 1e-6)
- `maxiter`: Maximum iterations (default: 1000)

# Returns
- `W` : W factor
- `H` : H factor 
- `converged` : Convergence result
- `i` : final number of iteration
"""

# References

Papers
- Matthew Brand. Incremental singular value decomposition of uncertain data with missing values. In Computer Vision—ECCV 2002: 7th European Conference on Computer Vision Copenhagen, Denmark, May 28–31, 2002 Proceedings, Part I 7, pages 707–720. Springer, 2002.
- Chen F and Rohe K, “A New Basis for Sparse  Principal Component Analysis.”, Journal of Computational and Graphical Statistics, 2023.
- Mariano Tepper and Guillermo Sapiro. Compressed nonnegative matrix factorization is fast and accurate. IEEE Transactions on Signal Processing, 64(9):2269–2283, May 2016. doi: 10.1109/tsp.2016. 2516971. URL https://doi.org/10.1109%2Ftsp.2016.2516971.
- Andrzej Cichocki and Anh-Huy Phan. Fast local algorithms for large scale nonnegative matrix and tensor factorizations. IEICE transactions on fundamentals of electronics, communications and computer sciences, 92(3):708–721, 2009.
- Yangyang Xu, Wotao Yin, Zaiwen Wen, and Yin Zhang. An alternating direction algorithm for matrix completion with nonnegative factors. Frontiers of Mathematics in China, 7(2):365–384, apr 2012. doi: 10.1007/s11464-012-0194-5. URL https://doi.org/10.1007% 2Fs11464-012-0194-5.


# Notes & Future Ideas
