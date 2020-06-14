module TGH
# this is woefully incomplete

using LinearAlgebra
using LinearAlgebra: RealHermSymComplexHerm
using Test
using Quadmath
using GenericMRRR

# Note: we use Frobenius norms here

function runtest(A::RealHermSymComplexHerm{T,<:StridedMatrix}) where T
    λ, V = geigen!(A)
    n = size(A,1)
    m = size(V,2)
    ntol = min(m,n)
    myeps = eps(T)
    if m == n
        resnorm = norm(V*Diagonal(λ)*V' - A)
    else
        resnorm = norm(Diagonal(λ) - V' * A * V)
    end
    Tnorm = max(opnorm(A,1), floatmin(T))
    if resnorm < Tnorm
        d_err =  (resnorm / Tnorm) / (myeps * ntol)
    elseif Tnorm < 1
        d_err = (min(resnorm, Tnorm * ntol) / Tnorm) / (myeps * ntol)
    else
        d_err = min(resnorm / Tnorm, ntol) / (myeps * ntol)
    end
    if m == n
        v_err = norm(V * V' - I) / (myeps * ntol)
    else
        v_err = norm(V' * V - I) / (myeps * ntol)
    end
    d_err, v_err
end

for T in [Float64, ComplexF64, Float128, Complex{Float128}]
    @testset "dense $T" begin
        n = 16
        A = Hermitian(rand(T,n,n))
        de, ve = runtest(A)
        @test de < 30
        @test ve < 30
    end
end

end # module
