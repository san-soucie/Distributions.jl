using Distributions, Test, Random, LinearAlgebra
using Distributions: MatrixProduct


rng = MersenneTwister(123456)
M = 3
N = 11
k = N
# Construct independent distributions and `Product` distribution from these.
μ = randn.(Ref(rng), fill(N, M))
Σ = map(x -> x * x', randn.(Ref(rng), fill((N, N), M)))

d1, d2 = collect(zip(size.(Σ)...))
covariance = zeros(eltype(eltype(Σ)), sum(d1), sum(d2))
i1 = 1
i2 = 1
for j = 1:length(Σ)
    covariance[i1:i1+d1[j]-1, i2:i2+d2[j]-1] = Σ[j]
    i1 += d1[j]
    i2 += d2[j]
end
mvnormals = MvNormal.(μ, Σ)
dirichlets = Dirichlet.([k], rand(rng, M))
dirichlets_with_mode = Dirichlet.([k], rand(rng, M) .+ 1.0)
mixeds = [MvNormal(μ[1], Σ[1]), Dirichlet(N, rand(rng))]
mixeds_with_mode = [MvNormal(μ[1], Σ[1]), Dirichlet(N, rand(rng)+1.0)]

Σ_dirichlet = cov.(dirichlets)
d1, d2 = collect(zip(size.(Σ_dirichlet)...))
covariance_dirichlet = zeros(eltype(eltype(Σ_dirichlet)), sum(d1), sum(d2))
i1 = 1
i2 = 1
for j = 1:length(Σ_dirichlet)
    covariance_dirichlet[i1:i1+d1[j]-1, i2:i2+d2[j]-1] = Σ_dirichlet[j]
    i1 += d1[j]
    i2 += d2[j]
end

Σ_mixed = cov.(mixeds)
d1, d2 = collect(zip(size.(Σ_mixed)...))
covariance_mixed = zeros(eltype(eltype(Σ_mixed)), sum(d1), sum(d2))
i1 = 1
i2 = 1
for j = 1:length(Σ_mixed)
    covariance_mixed[i1:i1+d1[j]-1, i2:i2+d2[j]-1] = Σ_mixed[j]
    i1 += d1[j]
    i2 += d2[j]
end

x_norm = rand.(Ref(rng), mvnormals)
x_dir = rand.(Ref(rng), dirichlets)
x_mix = rand.(Ref(rng), mixed)

X_norm = hcat(x_norm...)'
X_dir = hcat(x_dir...)'
X_mix = hcat(x_mix...)'

mvnormal_product = matrix_product_distribution(mvnormals)
dirichlet_product = matrix_product_distribution(dirichlets)
dirichlet_product_with_mode = matrix_product_distribution(dirichlets_with_mode)
mixed_product = matrix_product_distribution(mixeds)
mixed_product_with_mode = matrix_product_distribution(mixeds_with_modes)
bad_distributions = [mvnormals[1], Dirichlet(k-1, rand(rng))]

mvnormal_product_draw = rand(rng, mvnormal_product)
dirichlet_product_draw = rand(rng, dirichlet_product)
mixed_product_draw = rand(rng, mixed_product)

not_in_mvnormal_support = [
    randn(rng, N+1, M),
    randn(rng, N*M, 1),
    mvnormal_product_draw + mvnormal_product_draw * im
]
not_in_dirichlet_support = [
    dirichlet_product_draw .* 0.9,
    hcat(dirichlet_product_draw, dirichlet_product_draw),
]
not_in_mixed_support = [
    rand(rng, N, 2) .- 2,
    randn(rng, N, 1),
]

@testset "MatrixProduct constructors" begin
    @test mvnormal_product isa MatrixGaussian
    @test dirichlet_product isa MatrixProduct
    @test mixed_product isa MatrixProduct
    @test_throws ArgumentError matrix_product_distribution(bad_distributions)
end

@testset "MatrixProduct length" begin
    @test length(mvnormal_product) == M
    @test length(dirichlet_product) == M
    @test length(mixed_product) == 2
end

@testset "MatrixProduct size" begin
    @test size(mvnormal_product) == (N, M)
    @test size(dirichlet_product) == (k, M)
    @test size(mixed_product) == (N, 2)
end

@testset "MatrixProduct rank" begin
    @test rank(mvnormal_product) == min(N, M)
    @test rank(dirichlet_product) == min(k, M)
    @test rank(mixed_product) == min(N, 2)
end

@testset "MatrixProduct insupport" begin
    @test insupport(mvnormal_product, mvnormal_product_draw)
    @test insupport(dirichlet_product, dirichlet_product_draw)
    @test insupport(mixed_product, mixed_product_draw)

    for x in not_in_mvnormal_support
        @test !insupport(mvnormal_product, x)
    end

    for x in not_in_dirichlet_support
        @test !insupport(dirichlet_product, x)
    end

    for x in not_in_mixed_support
        @test !insupport(mixed_product, x)
    end
end

@testset "MatrixProduct mean" begin
    @test mean(mvnormal_product) == hcat(mean.(mvnormals)...)'
    @test mean(dirichlet_product) == hcat(mean.(dirichlets)...)'
    @test mean(mixed_product) == hcat(mean.(mixeds)...)'
end

@testset "MatrixProduct mode" begin
    @test mode(mvnormal_product) == hcat(mode.(mvnormals)...)'
    @test_throws ErrorException mode(dirichlet_product)
    @test mode(dirichlet_product_with_mode) == hcat(mode.(dirichlets_with_mode)...)'
    @test_throws ErrorException mode(mixed_product)
    @test mode(mixed_product_with_mode) == hcat(mode.(mixeds_with_mode)...)'
end

@testset "MatrixProduct var" begin
    @test var(mvnormal_product) == hcat(var.(mvnormals)...)'
    @test var(dirichlet_product) == hcat(var.(dirichlets...)'
    @test var(mixed_product) == hcat(var.(mixeds...)'
end

@testset "MatrixProduct cov" begin
    @test cov(mvnormal_product) == covariance
    @test var(dirichlet_product) == covariance_dirichlet
    @test var(mixed_product) == covariance_mixed
end

@testset "MatrixProduct logpdf" begin
    @test logpdf(mvnormal_product, X_norm) ≈ sum(logpdf.(mvnormals, x_norm))
    @test logpdf(dirichlet_product, X_dir) ≈ sum(logpdf.(dirichlets, x_dir))
    @test logpdf(mixed_product, X_mix) ≈ sum(logpdf.(mixeds, x_mix))
end

@testset "MatrixProduct entropy" begin
    @test entropy(mvnormal_product) ≈ sum(entropy.(mvnormals))
    @test entropy(dirichlet_product) ≈ sum(entropy.(dirichlets))
    @test entropy(mixed_product) ≈ sum(entropy.(mixeds))
end

@testset "MatrixProduct consistency" begin
    for d in [mvnormal_product, dirichlet_product, mixed_product]
        @test insupport(d, mean(d))
        @test size(mean(d)) == size(var(d)) == size(d)
        @test size(cov(d)) == (sum(size(d))^2, sum(size(d))^2)
    end
end

@testset "MatrixProduct sample moments" begin
    for d in [mvnormal_product, dirichlet_product, mixed_product]
        @test isapprox(mean(rand(d, 100000)), mean(d) , atol = 0.1)
        @test isapprox(cov(hcat(vec.(rand(d, 100000))...)'), cov(d) , atol = 0.1)
    end
end
