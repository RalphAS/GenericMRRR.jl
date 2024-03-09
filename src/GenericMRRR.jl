module GenericMRRR

using LinearAlgebra
using LinearAlgebra: eigsortby, sorteig!, checksquare, reflector!
using GenericSchur
import GenericSchur: geigen!, geigvals!
export MRRR

include("symtridiag.jl")
include("dqds.jl")

struct AlgorithmFailure <: Exception
    msg::String
end
struct ConvergenceFailure <: Exception
    msg::String
end
struct RepresentationFailure <: Exception
    msg::String
end
"""
    MRRR <: LinearAlgebra.Algorithm

Specifier for the multiple relatively robust representation algorithm
for computing selected eigenvalues and eigenvectors of a real symmetric tridiagonal
matrix.
"""
struct MRRR <: LinearAlgebra.Algorithm end

"""
    geigen!(A::SymTridiagonal{T},...) where T <: AbstractFloat -> Eigen

Computes full or partial eigen-decomposition of a symmetric tridiagonal matrix.

Signatures and functionality are (intended to be!) identical to
`LinearAlgebra.eigen!(::SymTridiagonal,...)`, except that these handle more general element
types.
"""
function geigen!(A::SymTridiagonal{T}, alg::MRRR) where T <: AbstractFloat
    λ, V = _st_schur!(A.dv, A.ev, wantV = true)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
function geigen!(A::SymTridiagonal{T}, irange::UnitRange, alg::MRRR) where T <: AbstractFloat
    λ, V = _st_schur!(A.dv, A.ev, wantV = true,
                     select=IndexedEigvalSelector(first(irange),last(irange)))
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
function geigen!(A::SymTridiagonal{T}, vl::Real, vu::Real, alg::MRRR
                 ) where T <: AbstractFloat
    λ, V = _st_schur!(A.dv, A.ev, wantV = true,
                     select=IntervalEigvalSelector(vl,vu))
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
"""
    geigvals!(A::SymTridiagonal{T},...) where T <: AbstractFloat -> Vector{T}

Computes some or all eigenvalues of a symmetric tridiagonal matrix.

Signatures and functionality are (intended to be!) identical to
`LinearAlgebra.eigvals!(::SymTridiagonal,...)`, except that these handle more general element
types.
"""
function geigvals!(A::SymTridiagonal{T}, alg::MRRR) where T <: AbstractFloat
    λ, _ = _st_schur!(A.dv, A.ev, wantV = false)
    return λ
end
function geigvals!(A::SymTridiagonal{T}, irange::UnitRange, alg::MRRR
                   ) where T <: AbstractFloat
    λ, _ = _st_schur!(A.dv, A.ev, wantV = false,
                     select=IndexedEigvalSelector(first(irange),last(irange)))
    return λ
end
function geigvals!(A::SymTridiagonal{T}, vl::Real, vu::Real, alg::MRRR
                   ) where T <: AbstractFloat
    λ, _ = _st_schur!(A.dv, A.ev, wantV = false,
                     select=IntervalEigvalSelector(vl,vu))
    return λ
end

# this is the gateway for decomposed Hermitian matrices
function GenericSchur._gschur!(A::SymTridiagonal{T},
                  alg::MRRR,
                  Z::Union{Nothing, AbstractArray} = nothing;
                  maxiter=30*size(A,1)) where {T}
    wantV = (Z !== nothing)
    λ, Vtri = _st_schur!(A.dv, A.ev, wantV = wantV)
    A.dv .= λ
    if wantV
        Ztmp = copy(Z)
        mul!(Z, Ztmp, Vtri)
    end
end

using LinearAlgebra: RealHermSymComplexHerm

function geigen!(A::RealHermSymComplexHerm{T, <:StridedMatrix},
                 irange::UnitRange,
                 alg::MRRR; kwargs...
                 ) where T <: Union{AbstractFloat, Complex{AbstractFloat}}
    H = hessenberg!(A)
    Q = GenericSchur._materializeQ(H)
    λ, Vtri = _st_schur!(H.H.dv, H.H.ev, wantV = true,
                      select=IndexedEigvalSelector(first(irange),last(irange)))
    V = similar(Q, size(Vtri)...)
    mul!(V, Q, Vtri)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end

function geigen!(A::RealHermSymComplexHerm{T, <:StridedMatrix},
                 vl::Real, vu::Real,
                 alg::MRRR; kwargs...
                 ) where T <: Union{AbstractFloat, Complex{AbstractFloat}}
    H = hessenberg!(A)
    Q = GenericSchur._materializeQ(H)
    λ, Vtri = _st_schur!(H.H.dv, H.H.ev, wantV = true,
                     select=IntervalEigvalSelector(vl,vu))
    V = similar(Q, size(Vtri)...)
    mul!(V, Q, Vtri)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end
end # module
