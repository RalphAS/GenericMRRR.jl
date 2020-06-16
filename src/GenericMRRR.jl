module GenericMRRR

using LinearAlgebra
using LinearAlgebra: eigsortby, sorteig!, checksquare, reflector!
using GenericLinearAlgebra: symtri!
export geigen!, geigvals!

include("symtridiag.jl")
include("dqds.jl")

using LinearAlgebra: RealHermSymComplexHerm

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
    # S = ghessenberg!(A)
    # λ, V = _st_schur!(S.H.dv, S.H.ev, wantV = true)
    S = symtri!(A)
    λ, V = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = true)
    if T <: Complex
        V = V .+ 0im
    end
    lmul!(S.Q, V)
    LinearAlgebra.Eigen(sorteig!(λ, V, sortby)...)
end

function geigvals!(A::RealHermSymComplexHerm{T,<:StridedMatrix};
                sortby::Union{Function,Nothing}=eigsortby) where T
    # S = ghessenberg!(A)
    # λ, _ = _st_schur!(S.H.dv, S.H.ev, wantV = false)
    S = symtri!(A)
    λ, _ = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = false)
    sorteig!(λ, sortby)
    return λ
end

end # module
