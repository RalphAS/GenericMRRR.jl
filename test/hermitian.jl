module TGH
# this is woefully incomplete

using LinearAlgebra
using LinearAlgebra: RealHermSymComplexHerm
using Test
using Quadmath
using GenericSchur: geigen!, geigvals!
using GenericMRRR

# Note: we use Frobenius norms here

function checkHermEig(A::RealHermSymComplexHerm{T}, λ, V, tol) where T
    n = size(A,1)
    m = size(V,2)
    ntol = min(m,n)
    myeps = eps(real(T))
    if m == n
        resnorm = norm(V*Diagonal(λ)*V' - A)
    else
        resnorm = norm(Diagonal(λ) - V' * A * V)
    end
    Tnorm = max(opnorm(A,1), floatmin(real(T)))
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
    @test d_err < 30
    @test v_err < 30
end

function runtest(A::RealHermSymComplexHerm{T,<:StridedMatrix}, tol) where T
    n = size(A,1)
    λ, V = geigen!(copy(A), MRRR())
    checkHermEig(A, λ, V, tol)
    @test issorted(λ)
    i1 = n >> 2
    i2 = i1 + (n >> 1)
    vl = (λ[i1-1] + λ[i1]) / 2
    vu = (λ[i2] + λ[i2+1]) / 2
    λ1, V1 = geigen!(copy(A), i1:i2, MRRR())
    checkHermEig(A, λ1, V1, tol)
    @test length(λ1) == i2-i1+1
    @test all(λ1 .> vl)
    @test all(λ1 .< vu)
    λ2, V2 = geigen!(copy(A), vl, vu, MRRR())
    checkHermEig(A, λ2, V2, tol)
    @test length(λ2) == i2-i1+1
    @test all(λ2 .> vl)
    @test all(λ2 .< vu)
end

for T in [Float128, Complex{Float128}, Float64, ComplexF64]
    @testset "dense Hermitian $T" begin
        n = 16
        A = Hermitian(rand(T,n,n))
        runtest(A, 30)
    end
end

end # module
