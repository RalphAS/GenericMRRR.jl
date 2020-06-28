# extend methods to dense Hermitian

import .GenericLinearAlgebra
using .GenericLinearAlgebra: symtri!

using LinearAlgebra: RealHermSymComplexHerm

function geigen!(A::RealHermSymComplexHerm{T,<:StridedMatrix};
                 sortby::Union{Function,Nothing}=eigsortby
                 ) where T <: AbstractFloat
    S = symtri!(A)
    λ, V = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = true)
    if eltype(A) <: Complex
        V = V .+ 0im
    end
    lmul!(S.Q, V)
    LinearAlgebra.Eigen(sorteig!(λ, V, sortby)...)
end

function geigen!(A::RealHermSymComplexHerm{T,<:StridedMatrix}, irange::UnitRange
                 ) where T <: AbstractFloat
    S = symtri!(A)
    λ, V = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = true,
                     select=IndexedEigvalSelector(first(irange),last(irange)))
    if eltype(A) <: Complex
        V = V .+ 0im
    end
    lmul!(S.Q, V)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end

function geigen!(A::RealHermSymComplexHerm{T,<:StridedMatrix}, vl::Real, vu::Real
                 ) where T <: AbstractFloat
    S = symtri!(A)
    λ, V = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = true,
                     select=IntervalEigvalSelector(vl,vu))
    if eltype(A) <: Complex
        V = V .+ 0im
    end
    lmul!(S.Q, V)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end

function geigvals!(A::RealHermSymComplexHerm{T,<:StridedMatrix};
                sortby::Union{Function,Nothing}=eigsortby) where T <: AbstractFloat
    S = symtri!(A)
    λ, _ = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = false)
    sorteig!(λ, sortby)
    return λ
end

function geigvals!(A::RealHermSymComplexHerm{T,<:StridedMatrix}, irange::UnitRange
                   ) where T  <: AbstractFloat
    S = symtri!(A)
    λ, _ = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = false,
                     select=IndexedEigvalSelector(first(irange),last(irange)))
    sorteig!(λ, sortby)
    return λ
end

function geigvals!(A::RealHermSymComplexHerm{T,<:StridedMatrix}, vl::Real, vu::Real
                   ) where T  <: AbstractFloat
    S = symtri!(A)
    λ, _ = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = false,
                     select=IntervalEigvalSelector(vl,vu))
    sorteig!(λ, sortby)
    return λ
end
