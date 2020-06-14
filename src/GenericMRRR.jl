module GenericMRRR

using LinearAlgebra
using LinearAlgebra: eigsortby, sorteig!, checksquare, reflector!
export ghessenberg!, geigen!, geigvals!

include("symtridiag.jl")
include("dqds.jl")

using LinearAlgebra: RealHermSymComplexHerm

"""
    ghessenberg!(Ah::RealHermSymComplexHerm) -> Hessenberg

reduce a real symmetric or complex Hermitian matrix
to real symmetric tridiagonal form by unitary similarity transformation
`Qᴴ A Q = T`
"""
function ghessenberg!(Ah::RealHermSymComplexHerm{TA,<:StridedMatrix}) where TA
# WARNING: this is naive and inefficient!
    A = Ah.data
    T = eltype(TA)
    RT = real(T)
    n = checksquare(A)
    D = zeros(RT,n)
    E = zeros(RT,n-1)
    τ = zeros(T,n-1)
    r = zeros(T,n)
    if Ah.uplo == 'U'
        for i in n-1:-1:1
            α = A[i,i+1]
            r[1] = α
            r[2:i] .= A[1:i-1,i+1]
            τi = reflector!(view(r,1:i))
            A[1:i-1,i+1] .= r[2:i]
            β = r[1]
            E[i] = real(β)
            if τi != 0
                A[i,i+1] = RT(1)
                # x = τ A v
                mul!(view(τ,1:i), view(Ah,1:i,1:i), view(A,1:i,i+1), τi, false)
                # w = x - (τ/2) * (xᴴ v) * v
                α = -(τi/2)* dot(view(τ,1:i), view(A,1:i,i+1))
                τ[1:i] .+= α * A[1:i, i+1]
                # TODO: apply transformation as rank-2 update
                # A <- A - v * wᴴ - w * vᴴ
                A[1:i,1:i] .-= triu(view(τ,1:i) .* adjoint(view(A,1:i,i+1)))
                A[1:i,1:i] .-=  triu(view(A,1:i,i+1) .* adjoint(view(τ,1:i)))
            else
                A[i,i] = real(A[i,i])
            end
            A[i,i+1] = E[i]
            D[i+1] = real(A[i+1,i+1])
            τ[i] = τi
        end
        D[1] = real(A[1,1])
    else
        A[1,1] = real(A[1,1])
        for i in 1:n-1
            τi = reflector!(view(A, i+1:n, i))
            α = A[i+1,i]
            E[i] = real(α)
            if τi != 0
                A[i+1,i] = RT(1)
                # x = τ A v
                mul!(view(τ,i:n-1), view(Ah,i+1:n,i+1:n), view(A,1+i:n,i), τi, false)
                # w = x - (τ/2) * (xᴴ v) * v
                α = -(τi/2)* dot(view(τ,i:n-1), view(A,i+1:n,i))
                τ[i:n-1] .+= α * A[i+1:n, i]
                # TODO: apply transformation as rank-2 update (zher2)
                # A <- A - v * wᴴ - w * vᴴ
                A[i+1:n,i+1:n] .-= tril(view(τ,i:n-1) .* adjoint(view(A,i+1:n,i)))
                A[i+1:n,i+1:n] .-=  tril(view(A,i+1:n,i) .* adjoint(view(τ,i:n-1)))
            else
                A[i+1,i+1] = real(A[i+1,i+1])
            end
            A[i+1,i] = E[i]
            D[i] = real(A[i,i])
            τ[i] = τi
        end
        D[n] = real(A[n,n])
    end
    return Hessenberg(A, τ, SymTridiagonal(D, E), Ah.uplo)
end


"""
    geigen!(A::SymTridiagonal{T},...) where T <: Real -> Eigen

Computes eigen-decomposition of a symmetric tridiagonal matrix.

Signatures and functionality are (intended to be!) identical to
`LinearAlgebra.eigen!(::SymTridiagonal,...)`, except that these handle more general element
types.
"""
function geigen!(A::SymTridiagonal{T}) where T <: Real
    λ, V = _st_schur!(A.dv, A.ev, wantV = true)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
function geigen!(A::SymTridiagonal{T}, irange::UnitRange) where T <: Real
    λ, V = _st_schur!(A.dv, A.ev, wantV = true,
                     select=IndexedEigvalSelector(first(irange),last(irange)))
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
function geigen!(A::SymTridiagonal{T}, vl::Real, vu::Real) where T <: Real
    λ, V = _st_schur!(A.dv, A.ev, wantV = true,
                     select=IntervalEigvalSelector(vl,vu))
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
"""
    geigvals!(A::SymTridiagonal{T},...) where T <: Real -> Vector{T}

Computes eigen-decomposition of a symmetric tridiagonal matrix.

Signatures and functionality are (intended to be!) identical to
`LinearAlgebra.eigvals!(::SymTridiagonal,...)`, except that these handle more general element
types.
"""
function geigvals!(A::SymTridiagonal{T}) where T <: Real
    λ, _ = _st_schur!(A.dv, A.ev, wantV = false)
    return λ
end
function geigvals!(A::SymTridiagonal{T}, irange::UnitRange) where T <: Real
    λ, _ = _st_schur!(A.dv, A.ev, wantV = false,
                     select=IndexedEigvalSelector(first(irange),last(irange)))
    return λ
end
function geigvals!(A::SymTridiagonal{T}, vl::Real, vu::Real) where T <: Real
    λ, _ = _st_schur!(A.dv, A.ev, wantV = false,
                     select=IntervalEigvalSelector(vl,vu))
    return λ
end

function geigen!(A::RealHermSymComplexHerm{T,<:StridedMatrix};
                sortby::Union{Function,Nothing}=eigsortby) where T
    S = ghessenberg!(A)
    λ, V = _st_schur!(S.H.dv, S.H.ev, wantV = true)
    if T <: Complex
        V = V .+ 0im
    end
    lmul!(S.Q, V)
    LinearAlgebra.Eigen(sorteig!(λ, V, sortby)...)
end

function geigvals!(A::RealHermSymComplexHerm{T,<:StridedMatrix};
                sortby::Union{Function,Nothing}=eigsortby) where T
    S = ghessenberg!(A)
    λ, _ = _st_schur!(S.H.dv, S.H.ev, wantV = false)
    sorteig!(λ, sortby)
    return λ
end

end # module
