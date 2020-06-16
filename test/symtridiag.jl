module TGST

using LinearAlgebra
using Test
using Quadmath
using GenericMRRR
using GenericMRRR: RepresentationFailure

# used to build test matrices
using GenericLinearAlgebra: symtri!

const matdir = joinpath(@__DIR__,"stester","DATA")

function wilkinson(n,::Type{T}) where T
    if mod(n,2) == 1
        nh = (n-1) >>> 1
        D = T.(vcat(nh:-1:1,[0],1:nh))
    else
        D = abs.(T.(collect(-(n-1):2:(n-1)))/2)
    end
    E = fill(T(1),n-1)
    D,E
end

function clement(n,::Type{T}) where T
    D = zeros(T,n)
    E = [sqrt(T(i*(n+1-i))) for i in 1:n-1]
    D,E
end

function legendre(n,::Type{T}) where T
    D = zeros(T,n)
    E = [T(i)/sqrt(T((2i-1)*(2i)+1)) for i in 2:n]
    D,E
end

function laguerre(n,::Type{T}) where T
    D = T.(collect(3:2:2*n+1))
    E = T.(collect(2:n))
    D,E
end

function hermite(n,::Type{T}) where T
    D = zeros(T,n)
    E = [sqrt(T(i)) for i in 1:n-1]
    D,E
end

# LAPACK-style classes
function st_testmat1(n,itype,::Type{T}=Float64; dtype=:normal) where T
    dtype ∈ [:normal, :unif, :uniform] || throw(ArgumentError("incomprehensible dtype"))
    λ = zeros(T,n)
    ulp = eps(real(T))
    kpar = 1/ulp
    if itype == 1
        λ[1] = 1
        for j=2:n
            λ[j] = 1 / T(j)
        end
    elseif itype == 2
        λ[1:n-1] .= 1
        λ[n] = T(1) / n
    elseif itype == 3
        for j=1:n
            λ[j] = 1 / (kpar^((j-1)/(n-1)))
        end
    elseif itype == 4
        for j=1:n
            λ[j] = T(1) - (T(j-1)/T(n-1)) * (1-1/kpar)
        end
    elseif itype == 5
        λ .= exp.(-log(n) * rand(n))
    elseif itype == 6
        if dtype == :normal
            λ .= randn(n)
        else
            λ .= rand(dtype, n)
        end
    elseif itype == 7
        λ[1:n-1] .= (1:n-1)*ulp
        λ[n] = 1
    elseif itype == 8
        λ[1] = ulp
        for j = 2:n-1
            λ[j] = 1 + sqrt(ulp) * j
        end
        λ[n] = 2
    elseif itype == 9
        λ[1] = 1
        for j=2:n
            λ[j] = λ[j-1] + 100 * ulp
        end
    end
    Q,r = qr(randn(n,n))
    Q = T.(Matrix(Q)) # until LinearAlgebra is fixed
    # FIXME: sign adjustment
    A = Q' * diagm(λ) * Q
    if T <: Union{Float32, Float64}
        F = hessenberg!(A)
        S = F.H
    else
        # F = ghessenberg!(Hermitian(A))
        # S = F.H
        F = symtri!(Hermitian(A))
        S = F.diagonals
    end
    return S, λ
end
const itypes1 = 1:9

function st_testmat2(n,itype,::Type{T}=Float64; dtype=:normal) where T <: Real
    dtype ∈ [:normal, :unif, :uniform] || throw(ArgumentError("incomprehensible dtype"))
    λ = zeros(T,n)
    ulp = eps(real(T))
    if itype == 0
        S = SymTridiagonal(zeros(T,n),zeros(T,n-1))
        λ = zeros(T,n)
    elseif itype == 1
        S = SymTridiagonal(ones(T,n),zeros(T,n-1))
        λ = ones(T,n)
    elseif itype == 2
        S = SymTridiagonal(fill(T(2),n),ones(T,n-1))
        λ = nothing
    elseif itype == 3
        S = SymTridiagonal(wilkinson(n,T)...)
        λ = nothing
    elseif itype == 4
        S = SymTridiagonal(clement(n,T)...)
        λ = T.(collect(-n:2:n))
    elseif itype == 5
        S = SymTridiagonal(legendre(n,T)...)
        λ = nothing
    elseif itype == 6
        S = SymTridiagonal(laguerre(n,T)...)
        λ = nothing
    elseif itype == 7
        S = SymTridiagonal(hermite(n,T)...)
        λ = nothing
    else
        throw(ArgumentError("unimplemented itype"))
    end
    S,λ
end
const itypes2 = 1:7

function gluemats(mat1::SymTridiagonal{T}, mats...; gval=zero(T)) where T
    D = diag(mat1)
    E = diag(mat1,1)
    for mat in mats
        append!(D, diag(mat))
        push!(E, gval)
        append!(E, diag(mat,1))
    end
    SymTridiagonal(D, E)
end

function batch(n, ::Type{T}=Float64; thresh=50, quiet=true) where {T}
    dmax=0.0; vmax=0.0
    for itype in itypes1
        @testset "class 1 type $itype" begin
            A,_ = st_testmat1(n, itype, T)
            de, ve = runtest(diag(A), diag(A,1))
            dmax = max(dmax, de)
            vmax = max(vmax, ve)
            @test de < thresh
            @test ve < thresh
        end
    end
    quiet || println("peak errors: $dmax $vmax")
    dmax=0.0; vmax=0.0
    for itype in itypes2
        @testset "class 2 type $itype" begin
            A,_ = st_testmat2(n, itype, T)
            de, ve = runtest(diag(A), diag(A,1))
            dmax = max(dmax, de)
            vmax = max(vmax, ve)
            @test de < thresh
            @test ve < thresh
        end
    end
    quiet || println("peak errors: $dmax $vmax")
    nothing
end

maxnorm(A) = maximum(abs.(vec(A)))

function runtest(D::AbstractVector{T}, E; λ=nothing, V=nothing) where T <: Real
    if λ === nothing
        λ, V = geigen!(SymTridiagonal(copy(D), copy(E)))
    end
    n = length(D)
    m = size(V,2)
    ntol = min(m,n)
    myeps = eps(T)
    if m == n
        resnorm = maxnorm(V*Diagonal(λ)*V' - SymTridiagonal(D,E))
    else
        resnorm = maxnorm(Diagonal(λ) - V' * SymTridiagonal(D,E) * V)
    end
    Tnorm = max(opnorm(SymTridiagonal(D,E),1), floatmin(T))
    if resnorm < Tnorm
        d_err =  (resnorm / Tnorm) / (myeps * ntol)
    elseif Tnorm < 1
        d_err = (min(resnorm, Tnorm * ntol) / Tnorm) / (myeps * ntol)
    else
        d_err = min(resnorm / Tnorm, ntol) / (myeps * ntol)
    end
    if m == n
        v_err = maxnorm(V * V' - I) / (myeps * ntol)
    else
        v_err = maxnorm(V' * V - I) / (myeps * ntol)
    end
    d_err, v_err
end

function loadmat(fname)
    D = Vector{Float64}(undef,0)
    E = Vector{Float64}(undef,0)
    open(fname,"r") do f
        l = readline(f)
        n = parse(Int,l)
        for j in 1:n
            l = readline(f)
            sj,sd,se = split(l)
            push!(D, parse(Float64,sd))
            if j < n
                push!(E, parse(Float64,se))
            end
        end
    end
    D, E
end

function loadλs(fname)
    λ = Vector{Float64}(undef,0)
    open(fname,"r") do f
        l = readline(f)
        n = parse(Int,l)
        for j in 1:n
            l = readline(f)
            push!(λ, parse(Float64,l))
        end
    end
    λ
end

function checkλs(D, E, λ, fname)
    eigfile = fname[1:end-3] * "eig"
    if isfile(eigfile)
        λbase = loadλs(eigfile)
    else
        @info "Using LAPACK for \"true\" eigvals"
        λbase = eigvals(SymTridiagonal(D,E))
    end
    λs = sort(λ)
    maxerr = 0.0
    # a plausible value for scale to use for checking true zero
    zcrit = eps(eltype(D)) * max(maximum(abs.(D)),maximum(abs.(E)))
    jp = 0
    for j in 1:length(D)
        r = (λbase[j] == 0) ? abs(λ[j]) / zcrit : abs((λs[j] - λbase[j])/λbase[j])
        if r > maxerr
            maxerr = r
            jp = j
        end
    end
    if maxerr > 0
        println("worst λ: $(λs[jp]) cf. $(λbase[jp]), rel. error $maxerr")
    else
        println("perfect agreement.")
    end
end

function runfile(fname, wantV=true, checkev=!wantV)
    D, E = loadmat(fname)
    n = length(D)
    if wantV
        λ, V = geigen!(SymTridiagonal(copy(D), copy(E)))
    else
        λ = geigvals!(SymTridiagonal(copy(D), copy(E)))
        V = nothing
    end
    if wantV
        de, ve = runtest(D,E,λ=λ,V=V)
        println("errors: decomp $de orth $ve")
    end
    if checkev
        checkλs(D,E, λ, fname)
    end
    λ,V
end

const problemsF64 = Dict{String,Any}(
    "T_bug113.dat" => :representation,
    "T_bug113_38-47.dat" => :representation,
    "Julien_30.dat" => :representation,
    "B_bug255_bdsdc.dat"  => :representation,
    "B_glued_09a.dat" => :representation,
    "B_glued_09b.dat" => :representation,
    "B_glued_09c.dat" => :representation,
    "B_glued_09d.dat" => :representation,
    "B_gg_30_1D-3.dat" => :representation,
    "B_gg_30_1D-4.dat" => :representation,
    "B_gg_30_1D-5.dat" => :representation,
    "B_Kimura_429.dat" => :representation,
    "B_Kimura_981.dat" => :representation,
    "B_Kimura_985.dat" => :representation,
    "BNP_560a.dat"  => :inaccurate,
    "BNP_560b.dat"  => :inaccurate,
    "Golub-Kahan.dat" => :representation,
    "Lipshitz_1.dat"  => :representation,
    "T_bug126_U.dat" => :inaccurate,
    "Z_297.dat" => :representation,
    "Z_594.dat" => :representation,
)

const problemsF128 = Dict{String,Any}(
    "B_16.dat" => :representation,
    "B_bug316_gesdd.dat" => :representation,
    "Julien_30.dat" => :representation,
)

function runfiles(dir, nmax=100, wantV=true; promote=false, quiet=true, thresh=100)
    c = 0
    problems = promote ? problemsF128 : problemsF64
    for (root, dirs, files) in walkdir(dir)
        for fnam in files
            if fnam[end-3:end] == ".dat"
                D, E = loadmat(joinpath(root, fnam))
                if length(D) <= nmax
                    c += 1
                    @testset "$fnam vectors=$wantV" begin
                        if promote
                            Dx = Float128.(D)
                            Ex = Float128.(E)
                        else
                            Dx = copy(D)
                            Ex = copy(E)
                        end
                        local λ, V
                        if fnam ∈ keys(problems)
                            if wantV
                                if problems[fnam] == :representation
                                    @test_throws RepresentationFailure geigen!(SymTridiagonal(copy(Dx),copy(Ex)))
                                    @info "Accepting known failure for $fnam"
                                elseif problems[fnam] == :inaccurate
                                    λ, V = geigen!(SymTridiagonal(copy(Dx),copy(Ex)))
                                    de, ve = runtest(Dx,Ex,λ=λ,V=V)
                                    bad = de > thresh || ve > thresh
                                    (bad || !quiet) && println("  $fnam errors: decomp $de orth $ve")
                                    if bad
                                        @test_broken de < thresh
                                        @test_broken ve < thresh
                                        @info "Accepting known failure for $fnam"
                                    else
                                        @info "Residual test passed for $fnam"
                                        @test de < thresh
                                        @test ve < thresh
                                    end
                                else
                                    error("bad logic in test code.")
                                end
                            else
                                @test_throws problems[fnam] geigvals!(SymTridiagonal(copy(Dx),copy(Ex)))
                            end
                        else
                            if wantV
                                λ, V = geigen!(SymTridiagonal(copy(Dx),copy(Ex)))
                            else
                                λ = geigvals!(SymTridiagonal(copy(Dx),copy(Ex)))
                                V = nothing
                            end
                            if wantV
                                de, ve = runtest(Dx,Ex,λ=λ,V=V)
                                bad = de > thresh || ve > thresh
                                (bad || !quiet) && println("  errors: decomp $de orth $ve")
                                @test de < thresh
                                @test ve < thresh
                            else
                                checkλs(D, E, λ, joinpath(root,fnam))
                            end
                        end
                    end # testset
                end
            end
        end
    end
    quiet || println("done with $c files.")
end

for T in [Float64, Float128]
    @testset "Batch $T" begin
        batch(32, T)
    end
end


if isdir(matdir)
    @testset "STETester" begin
        # some of the larger cases are intermittent, so we skip them in CI
        runfiles(matdir, 100)
    end
    @testset "STETester-Float128" begin
        runfiles(matdir, promote=true)
    end
end

end # module
