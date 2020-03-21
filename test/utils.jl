using Distributions, PDMats
using Test, LinearAlgebra


# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0
@test extrema(r) == (1.5, 4.0)

@test partype(Gamma(1, 2)) == Float64
@test partype(Gamma(1.1, 2)) == Float64
@test partype(Normal(1//1, 2//1)) == Rational{Int}
@test partype(MvNormal(rand(Float32, 5), Matrix{Float32}(I, 5, 5))) == Float32

# special cases
@test partype(Kolmogorov()) == Float64
@test partype(Hypergeometric(2, 2, 2)) == Float64
@test partype(DiscreteUniform(0, 4)) == Float64

A = rand(1:10, 5, 5)
B = rand(Float32, 4)
C = 1//2
L = rand(Float32, 4, 4)
D = PDMats.PDMat(L * L')

# Ensure that utilities functions works with abstract arrays

@test isprobvec(GenericArray([1, 1, 1])) == false
@test isprobvec(GenericArray([1/3, 1/3, 1/3]))

# Positive definite matrix
M = GenericArray([1.0 0.0; 0.0 1.0])
# Non-invertible matrix
N = GenericArray([1.0 0.0; 1.0 0.0])

@test Distributions.isApproxSymmmetric(N) == false
@test Distributions.isApproxSymmmetric(M)

mats = [[1 2; 3 4], [5 6], [7], [8 9 10; 11 12 13; 14 15 16]]
bd_mat = [1  2  0  0  0  0  0  0;
          3  4  0  0  0  0  0  0;
          0  0  5  6  0  0  0  0;
          0  0  0  0  7  0  0  0;
          0  0  0  0  0  8  9 10;
          0  0  0  0  0 11 12 13;
          0  0  0  0  0 14 15 16;]

mixed_mats = [[1 2; 3 4], Float32[5 6], Float64[7], [8 9 10; 11 12 13; 14 15 16]]
bd_mix = Float64[1.0  2.0  0.0  0.0  0.0  0.0  0.0  0.0;
                 3.0  4.0  0.0  0.0  0.0  0.0  0.0  0.0;
                 0.0  0.0  5.0  6.0  0.0  0.0  0.0  0.0;
                 0.0  0.0  0.0  0.0  7.0  0.0  0.0  0.0;
                 0.0  0.0  0.0  0.0  0.0  8.0  9.0 10.0;
                 0.0  0.0  0.0  0.0  0.0 11.0 12.0 13.0;
                 0.0  0.0  0.0  0.0  0.0 14.0 15.0 16.0;]


empty_mats = [[1 2; 3 4], [5 6], [], [7], [8 9 10; 11 12 13; 14 15 16]]
bd_emp = [1  2  0  0  0  0  0  0;
          3  4  0  0  0  0  0  0;
          0  0  5  6  0  0  0  0;
          0  0  0  0  7  0  0  0;
          0  0  0  0  0  8  9 10;
          0  0  0  0  0 11 12 13;
          0  0  0  0  0 14 15 16;]

bad_3d_array = [rand(3), rand(3, 4), rand(3, 4, 5)]
bad_out_size_mat = [[1 2; 3 4], [5 6; 7 8]]
m = [1 2 0 0; 3 4 0 0; 0 0 5 6; 0 0 7 8]
mb = [1 2 0 ; 3 4 0 ; 0 0 5 ; 0 0 7 ]

@test Distributions.block_diagonal(mats) == bd_mat
@test Distributions.block_diagonal(mixed_mats) == bd_mix
@test Distributions.block_diagonal(empty_mats) == bd_emp
@test_throws ArgumentError Distributions.block_diagonal(bad_3d_array)
@test Distributions.block_diagonal!(bad_out_size_mat, m) == m
@test_throws ArgumentError  Distributions.block_diagonal!(bad_out_size_mat, mb)
