# extend methods to dense Hermitian

import .GenericLinearAlgebra
using .GenericLinearAlgebra: symtri!

using LinearAlgebra: RealHermSymComplexHerm

function geigen!(A::RealHermSymComplexHerm{T,<:StridedMatrix};
                sortby::Union{Function,Nothing}=eigsortby) where T
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
    S = symtri!(A)
    λ, _ = _st_schur!(S.diagonals.dv, S.diagonals.ev, wantV = false)
    sorteig!(λ, sortby)
    return λ
end
