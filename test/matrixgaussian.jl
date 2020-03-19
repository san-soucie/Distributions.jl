using Distributions, Random
using Test, LinearAlgebra, PDMats

#  dimensions

m = 3
n = 6


M    = randn(m, n)
U    = rand(InverseWishart(m + 2, Matrix(1.0I, m, m)))
V    = rand(InverseWishart(n + 2, Matrix(1.0I, n, n)))
Σ    = Matrix(kron(V, U))
Σ32 = Matrix{Float32}(Σ)
ΣBF  = Matrix{BigFloat}(Σ)
PDΣ  = PDMat(Σ)
ml   = randn(n)
w    = randn(n)
μ    = randn()
σ    = 2.0rand()
σ²   = σ ^ 2


D = MatrixGaussian(M, Σ)  #  m x n
G = MatrixGaussian(n, m)     #  n x m
L = MatrixGaussian(reshape(ml,m, 1), U)  #  m x 1
H = MatrixNormal(reshape(w, 1, n), V)  #  1 x n
K = MatrixNormal(reshape([μ], 1, 1), reshape([σ], 1, 1))  #  1 x 1

d = vec(D) # MvNormal(vec(M), V ⊗ U)
g = MvNormal( Matrix(1.0I, m*n, m*n) )
l = MvNormal(m, U)
h = MvNormal(w, V)
k = Normal(μ, σ)

A = rand(D)
B = rand(G)
X = rand(L)
Y = rand(H)
Z = rand(K)

a = vec(A)
b = vec(B)
x = vec(X)
y = vec(Y)
z = Z[1, 1]

@testset "Check all MatrixGaussian constructors" begin
    @test MatrixGaussian(M, Σ) isa MatrixGaussian
    @test MatrixGaussian(M, PDΣ) isa MatrixGaussian
    @test MatrixGaussian(M, PDΣ.chol) isa MatrixGaussian
end
