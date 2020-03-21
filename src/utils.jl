## macro for argument checking

macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

##### Utility functions

isunitvec(v::AbstractVector) = (norm(v) - 1.0) < 1.0e-12

isprobvec(p::AbstractVector{<:Real}) =
    all(x -> x ≥ zero(x), p) && isapprox(sum(p), one(eltype(p)))

pnormalize!(v::AbstractVector{<:Real}) = (v ./= sum(v); v)

add!(x::AbstractArray, y::AbstractVector) = broadcast!(+, x, x, y)
add!(x::AbstractArray, y::Zeros) = x

multiply!(x::AbstractArray, c::Number) = (x .*= c; x)

exp!(x::AbstractArray) = (x .= exp.(x); x)

# get a type wide enough to represent all a distributions's parameters
# (if the distribution is parametric)
# if the distribution is not parametric, we need this to be a float so that
# inplace pdf calculations, etc. allocate storage correctly
@inline partype(::Distribution) = Float64

# for checking the input range of quantile functions
# comparison with NaN is always false, so no explicit check is required
macro checkquantile(p,ex)
    p, ex = esc(p), esc(ex)
    :(zero($p) <= $p <= one($p) ? $ex : NaN)
end

macro checkinvlogcdf(lp,ex)
    lp, ex = esc(lp), esc(ex)
    :($lp <= zero($lp) ? $ex : NaN)
end

# because X == X' keeps failing due to floating point nonsense
function isApproxSymmmetric(a::AbstractMatrix{Float64})
    tmp = true
    for j in 2:size(a, 1)
        for i in 1:(j - 1)
            tmp &= abs(a[i, j] - a[j, i]) < 1e-8
        end
    end
    return tmp
end

# because isposdef keeps giving the wrong answer for samples
# from Wishart and InverseWisharts
hasCholesky(a::Matrix{Float64}) = isa(trycholesky(a), Cholesky)

function trycholesky(a::Matrix{Float64})
    try cholesky(a)
    catch e
        return e
    end
end

# for MatrixProduct covariances
function block_diagonal(a::AbstractVector{<:AbstractArray})
    # creating a block diagonal matrix
    all(length.(size.(a)) .<= 2) || throw(ArgumentError(
        "All blocks should be matrices or vectors"))
    vec_or_mat = a[broadcast(x -> x != [], a)] # Drop empty arrays
    ensure_mat = map(
        x -> (length(size(x)) == 2) ? x : reshape(x, (length(x), 1)), vec_or_mat)
    return block_diagonal(ensure_mat)
end

function block_diagonal(a::AbstractVector{<:AbstractMatrix})
    # creating a block diagonal matrix
    sizes = size.(a)
    d1, d2 = collect.(collect(zip(sizes...)))
    out = zeros(eltype(eltype(a)), sum(d1), sum(d2))
    return block_diagonal!(a, out)
end

function block_diagonal!(
    a::AbstractVector{<:AbstractMatrix{T}}, out::AbstractMatrix{T}) where T

    # creating a block diagonal matrix
    size_check = (size(out) == tuple(sum.(collect(zip(size.(a)...)))...))
    size_check || throw(ArgumentError(
        "out matrix should be appropriate size for provided blocks"))
    out = out .* zero(T)
    sizes = size.(a)
    d1, d2 = collect.(collect(zip(sizes...)))
    i1_bounds = [1, (1 .+ cumsum(d1, dims=1))...]
    i1_lower_bounds = i1_bounds[1:end-1]
    i1_upper_bounds = i1_bounds[2:end] .- 1

    i2_bounds = [1, (1 .+ cumsum(d2, dims=1))...]
    i2_lower_bounds = i2_bounds[1:end-1]
    i2_upper_bounds = i2_bounds[2:end] .- 1

    bounds = zip(i1_lower_bounds, i1_upper_bounds, i2_lower_bounds, i2_upper_bounds)
    for (j, (i1_l, i1_u, i2_l, i2_u)) in enumerate(bounds)
        out[i1_l:i1_u, i2_l:i2_u] = a[j]
    end

    return out
end
