using Distributions, Test, Random, LinearAlgebra
using Distributions: MatrixProduct

@testset "Testing normal matrix product distributions" begin
let
    rng = MersenneTwister(123456)
    M = 3
    N = 11
    # Construct independent distributions and `Product` distribution from these.
    μ = randn.([rng], fill(N, M))
    Σ = map(x -> x * x', randn.([rng], fill((N, N), M)))

    d1, d2 = collect(zip(size.(Σ)...))
    covariance = zeros(eltype(eltype(Σ)), sum(d1), sum(d2))
    i1 = 1
    i2 = 1
    for j = 1:length(a)
        covariance[i1:i1+d1[j]-1, i2:i2+d2[j]-1] = Σ[j]
        i1 += d1[j]
        i2 += d2[j]
    end
    ds = MvNormal.(μ, Σ)
    x = rand.(Ref(rng), ds)
    d_product = matrix_product_distribution(ds)
    @test d_product isa MvNormal
    # Check that methods for `Product` are consistent.
    @test length(d_product) == M
    @test size(d_product) == (N, M)
    @test logpdf(d_product, x) ≈ sum(logpdf.(ds, x))
    @test mean(d_product) == mean.(ds)
    @test var(d_product) == var.(ds)
    @test cov(d_product) == covariance
    @test entropy(d_product) ≈ sum(entropy.(ds))
end
end
