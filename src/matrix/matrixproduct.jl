import Base: length, size
import Statistics: mean, var, cov
import StatsBase: entropy
import Distributions: insupport
import LinearAlgebra: rank



struct MatrixProduct{
    S<:ValueSupport,
    T<:MultivariateDistribution{S},
    V<:AbstractVector{T},
} <: MatrixDistribution{S}
    v::V
    function MatrixProduct(v::V) where
        V<:AbstractVector{T} where
        T<:MultivariateDistribution{S} where
        S<:ValueSupport
        all(y -> length(y) == length(v[1]), v) || throw(ArgumentError("All distributions should have equal length"))
        return new{S, T, V}(v)
    end
end

size(d::MatrixProduct) = (length(d.v[1]), length(d))
length(d::MatrixProduct) = length(d.v)
rank(d::MatrixProduct) = minimum(size(d))
mean(d::MatrixProduct) = hcat(mean.(d.v)...)
var(d::MatrixProduct)  = hcat(var.(d.v)...)
function cov(d::MatrixProduct)
    a = cov.(d.v)

    # creating a block diagonal matrix
    sizes = size.(a)
    d1, d2 = collect(zip(sizes...))
    out = zeros(eltype(eltype(a)), sum(d1), sum(d2))
    i1 = 1
    i2 = 1
    for j = 1:length(a)
        out[i1:i1+d1[j]-1, i2:i2+d2[j]-1] = a[j]
        i1 += d1[j]
        i2 += d2[j]
    end

    return out
end
_logpdf(d::MatrixProduct, x::AbstractVector{<:AbstractVector}) = sum(n->logpdf(d.v[n], x[n]), 1:length(d))
_logpdf(d::MatrixProduct, x::AbstractMatrix) = _logpdf(d, Array.(eachcol(x)))
entropy(d::MatrixProduct) = sum(entropy, d.v)


function _rand!(rng::AbstractRNG, d::MatrixProduct, x::AbstractMatrix)
    x = hcat(rand.(rng, d.v[i])...)
end

insupport(d::MatrixProduct, x::AbstractVector{<:AbstractVector}) = all(insupport.(d.v, x))
insupport(d::MatrixProduct, x::AbstractMatrix) = insupport(d, Array.(eachcol(x)))

function matrix_product_distribution(dists::AbstractVector{<:MultivariateDistribution})
    return MatrixProduct(dists)
end


function matrix_product_distribution(dists::AbstractVector{<:MvNormal})
    m = matrix_product_distribution(dists)
    return MatrixGaussian(mean(m), cov(m))
end
