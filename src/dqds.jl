# This file is part of GenericMRRR.jl, released under the MIT "Expat" license

# The methods in this file are translated from
# the dqds routines in LAPACK.
# LAPACK authors are (obviously) not responsible for translation errors.

# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.

module DQDS
using LinearAlgebra
using ..GenericMRRR: safemin

# Some of the dqds implementation here assumes over/underflow control and
# NaN handling as in IEEE-754.
# Here such code is gated with a trait, to allow for generalization.

abstract type FPModel end
struct IEEEFPModel <: FPModel end
struct UnknownFPModel <: FPModel end

# Ideally the default should be unknown, but that would make things annoying
# for Quadmath, ArbFloats, etc.
# FPModel(::Type{T}) where T = UnknownFPModel()
FPModel(::Type{T}) where T = IEEEFPModel()

# Open questions: DoubleFloats
FPModel(::Type{T}) where T <: Base.IEEEFloat = IEEEFPModel()
FPModel(::Type{BigFloat}) = IEEEFPModel()

# The bulk of the code works on a two-fold interleaving for data locality.
# We use (q,e) to update (qq,ee) and vice versa - this is the ping-pong scheme.
# This should be storage-equivalent to the array used in the LAPACK routines.
struct QDtet{T}
  q::T
  qq::T
  e::T
  ee::T
end

# We handle the ping-pong alternation with value type params.
# Values correspond to pp in LAPACK routines.

# The compiler might handle these cleanly.
@inline function update!(zv::AbstractVector{QDtet{T}}, j, ::Val{0}, qnew, enew) where T
    @inbounds zv[j] = QDtet(zv[j].q, qnew, zv[j].e, enew)
end
@inline function update!(zv::AbstractVector{QDtet{T}}, j, ::Val{1}, qnew, enew) where T
    @inbounds zv[j] = QDtet(qnew, zv[j].qq, enew, zv[j].ee)
end
@inline function update0!(zv::AbstractVector{QDtet{T}}, j, qnew, enew) where T
    @inbounds zv[j] = QDtet(zv[j].q, qnew, zv[j].e, enew)
end
@inline function update1!(zv::AbstractVector{QDtet{T}}, j, qnew, enew) where T
    @inbounds zv[j] = QDtet(qnew, zv[j].qq, enew, zv[j].ee)
end
@inline function _save_eig!(zv::AbstractVector{QDtet{T}}, j, w) where T
    @inbounds zv[j] = QDtet(w, zv[j].qq, zv[j].e, zv[j].ee)
end


mutable struct DQDS_State{T}
    nfail::Int
    iter::Int
    ndiv::Int
    ttype::Int
    dmin1::T
    dmin2::T
    dn::T
    dn1::T
    dn2::T
    δσ::T
    g::T
    τ::T
    function DQDS_State{T}(iter, ndiv) where T
        z0 = zero(T)
        new{T}(0,iter,ndiv,0,z0,z0,z0,z0,z0,z0,z0,z0)
    end
end

function _dqds_tol(n, ::Type{T}) where T
    log(T(n)) * 4 * eps(T)
end

# Use dqds algorithm to compute all eigenvalues of a tridiagonal SPD matrix A
# to high relative accuracy.
# Elements of the LU factorization of a tridiagonal similar to A must be
# interleaved into the qd vector Z on entry.
# Translation of DLASQ2 from LAPACK
#
# On algorithm failure DLASQ2 returns with a flag but I found no recovery logic,
# so here we just throw.
function _dqds_eigvals!(n, Z::AbstractVector{T}) where T
    z0 = zero(T)
    safmin = safemin(T)
    tol = 100 * eps(T)
    tol2 = tol^2

    simple_result(tr1, tr2) = (tr1, tr2, 0, 0.0, 0.0)

    if length(Z) < 2*n
        throw(ArgumentError("Z must be have at least 2n elements"))
    end
    if n == 0
        return simple_result(z0, z0)
    end
    if n == 1
        if Z[1] < 0
            throw(ArgumentError("Z must be PSD"))
        end
        return simple_result(Z[1], Z[1])
    end
    if n == 2
        if Z[2] < 0 || Z[3] < 0
            throw(ArgumentError("Z must be PSD"))
        end
        Ztrace = Z[1]+Z[2]+Z[3]
        w1, w2 = _qd_2x2(Z[1],Z[2],Z[3],tol2)
        Z[1] = w1
        Z[2] = w2
        esum = w1 + w2
        return simple_result(Ztrace, esum)
    end

    # check for negative data, compute sums of q's and e's
    # Note: paper says q,e must be positive but DLASQ2 allows zeros.
    Z[2n] = z0
    emin = Z[2]
    qmax = z0
    zmax = z0
    d = z0
    e = z0
    for k in 1:2:2*(n-1)
        if Z[k] < 0
            throw(ArgumentError("Z must be PSD: diag elt $k"))
        elseif Z[k+1] < 0
            throw(ArgumentError("Z must be PSD: off-diag elt $k"))
        end
        d += Z[k]
        e += Z[k+1]
        qmax = max(qmax, Z[k])
        emin = min(emin, Z[k+1])
        zmax = max(qmax, zmax, Z[k+1])
    end
    if Z[2*n-1] < 0
            throw(ArgumentError("Z must be PSD: diag elt 2n-1"))
    end
    d += Z[2*n-1]
    qmax = max(qmax, Z[2*n-1])
    zmax = max(qmax, zmax)

    Ztrace = d + e

    # check for diagonality
    if e == 0
        for k in 2:n
            Z[k] = Z[2*k-1]
        end
        sort!(view(Z,1:n), rev=true)
        Z[2*n-1] = d
        return simple_result(Ztrace, Ztrace)
    end

    if Ztrace == 0
        Z[2*n-1] = z0
        return simple_result(Ztrace, z0)
    end

    # rearrange data for locality
    # Flip the qd array if warranted.
    cbias = T(3/2)
    zv = Vector{QDtet{T}}(undef,n)
    if cbias * Z[1] < Z[2*n-1]
        for k in 1:n-1
            zv[k] = QDtet(Z[2(n+1-k)-1],z0,Z[2*(n-k)],z0)
        end
        zv[n] = QDtet(Z[1],z0,z0,z0)
    else
        for k in 1:n
            zv[k] = QDtet(Z[2k-1],z0,Z[2k],z0)
        end
    end

    i0 = 1
    n0 = n

    # initial split checking via dqd and Li's test
    for pp in 0:1
        qmax, emin = _split_check0!(zv, i0, n0, tol2, Val(pp))
    end

    # set up for and then loop over calls to _dqds_3 (good dqds steps)

    state = DQDS_State{T}(2,2*(n0-i0))

    array_done = false
    σ = z0
    for iwhila in 1:n+1
        if n0 < 1
            array_done = true
            break
        end
        # E[n0] holds the value of σ when submatrix i0:n0
        # splits from the rest, but is negated as a flag.
        state.δσ = z0 # reset for this iteration
        if n0 == n
            σ = z0
        else
            σ = -zv[n0].e
        end
        if σ < 0
            # This is a logic err, so panic is appropriate.
            throw(ErrorException("split marked by positive value in E"))
        end

        # find last unreduced submatrix's top index i0, qmax, and emin.
        # Find Geršgorin-type bound if Q's are much greater than E's.

        emax = z0
        if n0 > i0
            emin = abs(zv[n0-1].e)
        else
            emin = z0
        end
        qmin = zv[n0].q
        qmax = qmin
        early = false
        i4x = 0
        for i in n0:-1:2
            i4x = i
            if zv[i-1].e <= 0
                early = true
                break
            end
            if qmin >= 4*emax
                qmin = min(qmin, zv[i].q)
                emax = max(emax, zv[i-1].e)
            end
            qmax = max(qmax, zv[i-1].q + zv[i-1].e)
            emin = min(emin, zv[i-1].e)
        end
        if !early
            i4x = 1
        end
        i0 = i4x
        pp = 0
        if n0 - i0 > 1
            dee = zv[i0].q
            deemin = dee
            kmin = i0
            for i in i0:n0-1
                dee = zv[i+1].q * (dee / (dee + zv[i+1].e))
                if dee <= deemin
                    deemin = dee
                    kmin = i
                end
            end
            # flip if appropriate
            if ((kmin-i0)*2 < n0-kmin) && (deemin <= zv[n0].q * (1/T(2)))
                ipn4 = 4*(i0+n0)
                pp = 2
                ip = i0+n0
                for j in i0:(i0+n0-1) >> 1
                    zd0 = QDtet(zv[ip-j].q, zv[ip-j].qq, zv[ip-j-1].e, zv[ip-j-1].ee)
                    zv[ip-j-1] = QDtet(zv[ip-j-1].q, zv[ip-j-1].qq, zv[j].e, zv[j].ee)
                    zv[ip-j] = QDtet(zv[j].q, zv[j].qq, zv[ip-j].e, zv[ip-j].ee)
                    zv[j] = zd0
                end
            end
        end
        # put -(initial shift) into dmin
        # (use lower bound on Geršgorin if positive)
        dmin = -max(z0, qmin-2*sqrt(qmin)*sqrt(emax))

        # now i0:n0 is unreduced
        # use pp ∈ [0,1] for ping-pong; 2 indicates that Z was flipped
        # so tests for deflation in DLASQ3 should not be performed.

        nbig = 100 * (n0-i0+1)
        submatrix_done = false
        for iwhilb in 1:nbig
            if i0 > n0
                submatrix_done = true
                break
            end
            # While submatrix is unfinished take a good dqds step.
            n0,pp,dmin,σ = _dqds_3!(zv,i0,n0,pp,qmax,σ,dmin,state)

            pp = 1 - pp
            # When emin is very small check for splits
            if pp == 0 && (n0-i0 >= 3)
                if zv[n0].ee <= tol2 * qmax || zv[n0].e < tol2 * σ
                    qmax, emin, i0 = _split_check!(zv,i0,n0,tol2,σ)
                end

            end
        end # iwhilb
        if !submatrix_done
            # Iteration limit reached:
            # Restore shift σ and place new d's and e's in qd array.
            # This might need to be done for serveral blocks.
            i1 = i0
            n1 = n0
            blocks_remain = true
            while blocks_remain
                tempq = zv[i0].q
                qnew = zv[i0].q + σ
                zv[i0] = QDtet(qnew,zv[i0].qq,zv[i0].e,zv[i0].ee)
                for k in i0+1:n0
                    tempe = zv[k-1].e
                    enew = zv[k-1].e * (tempq / zv[k-1].q)
                    tempq = zv[k].q
                    qnew = tempq + σ + tempe - enew
                    zv[k-1] = QDtet(zv[k-1].q,zv[k-1].qq,enew,zv[k-1].ee)
                    zv[k] = QDtet(qnew,zv[k].qq,zv[k].e,zv[k].ee)
                end
                # prepare to repeat on previous block if any
                if i1 > 1
                    n1 = i1-1
                    while (i1 >= 2) && (zv[i1-1].e >= 0)
                        i1 -= 1
                    end
                    σ = -zv[n1].e
                else
                    blocks_remain = false
                end
            end
            # convergence failure; store what we have
            for k in 1:n
                Z[2k-1] = zv[k].q
                # Only block 1..n0 is unfinished.
                # Remainder of e's must be negligible, but junk might be
                # left in workspace.
                if k < n0
                    Z[2k] = zv[k].e
                else
                    Z[2k] = z0
                end
            end
            throw(ErrorException("inner loop failed to converge"))
            # info = 2
            return simple_result(Ztrace, z0)
        end
    end # iwhila
    if !array_done
        throw(ErrorException("outer loop failed"))
        # info = 3
        # return
    end

    # move q's to the front
    for k in 1:n
        Z[k] = zv[k].q
    end
    # sort and compute sum of eigenvalues
    sort!(view(Z,1:n),rev=true)
    esum = z0
    for k in n:-1:1
        esum += Z[k]
    end
    # return trace, sum, and performance monitors
    return Ztrace, esum, state.iter, state.ndiv / n^2, 100 * state.nfail / state.iter
end

for (pp,qf,ef,qqf,eef) in
     ((0, :q, :e, :qq, :ee),
      (1, :qq, :ee, :q, :e))
    @eval begin
        function _split_check0!(zv::AbstractVector{QDtet{T}}, i0, n0, tol2, ::Val{$pp}) where T
            z0 = T(0)
            safmin = safemin(T)
            d = zv[n0].$qf
            for i in (n0-1):-1:i0
                if zv[i].$ef <= tol2*d
                    d = zv[i].$qf
                    update!(zv,i,Val(1-$pp),d,-z0)
                else
                    d = zv[i].$qf * (d / (d + zv[i].$ef))
                end
            end

            # dqd maps Z to ZZ plus Li's test.
            emin = zv[i0+1].$qf # Z[4*i0+pp+1]
            d = zv[i0].$qf # Z[4*i0+pp-3]
            for i in i0:n0-1
                #for i4 in 4*i0+pp:4:4*(n0-1)+pp
                qnew = d + zv[i].$ef
                if zv[i].$ef < tol2*d
                    update!(zv,i,Val(1-$pp), zv[i].q, -z0)
                    qnew = d
                    enew = z0
                    d = zv[i+1].$qf
                elseif ((safmin * zv[i+1].$qf < qnew)
                        && (safmin * qnew < zv[i+1].$qf))
                    temp = zv[i+1].$qf / qnew
                    enew = zv[i].$ef * temp
                    d *= temp
                else
                    enew = zv[i+1].$qf * (zv[i].$ef / qnew)
                    d - zv[i+1].$qf * (d / qnew)
                end
                emin = min(emin, enew)
                update!(zv,i,Val($pp), qnew, enew)
            end
            update!(zv,n0,Val($pp), d, z0)
            # find qmax
            qmax = zv[i0].$qqf
            for i in i0+1:n0
                qmax = max(qmax, zv[i0].$qqf)
            end
            return qmax, emin
        end
    end # eval
end # pp loop

# compute eigenvalues of 2x2 sym. PSD matrix in qd form
@inline function _qd_2x2(q1::T, e1::T, q2::T, tol2) where T
    if q2 > q1
        q2, q1 = q1, q2
    end
    t = ((q1 - q2) + e1) * (1/T(2))
    if e1 > q2*tol2 && t != 0
        s = q2 * (e1 / t)
        if s <= t
            s = q2 * (e1 / (t * (1 + sqrt(1 + s/t))))
        else
            s = q2 * (e1 / (t + sqrt(t) * sqrt(t + s)))
        end
        t = q1 + (s + e1)
        q2 = q2 * (q1 / t)
        q1 = t
    end
    return q1, q2
end

function _split_check!(zv::AbstractArray{QDtet{T}},i0,n0,tol2,σ) where T
    z0 = T(0)
    splt = i0-1
    qmax = zv[i0].q
    emin = zv[i0].e
    old_emin = zv[i0].ee
    for i in i0:(n0-3)
        if zv[i].ee <= tol2 * zv[i].q || zv[i].e <= tol2 * σ
            #  zv[i].e = -σ
            zv[i] = QDtet(zv[i].q, zv[i].qq, -σ, zv[i].ee)
            splt = i
            qmax = z0
            emin = zv[i+1].e
            old_emin = zv[i+1].ee
        else
            qmax = max(qmax, zv[i+1].q)
            emin = min(emin, zv[i].e)
            old_emin = min(old_emin, zv[i].ee)
        end
    end
    zv[n0] = QDtet(zv[n0].q, zv[n0].qq, emin, old_emin)
    i0 = splt + 1
    return qmax, emin, i0
end

# n0,pp,dmin,σ = _dqds_3!(Z,i0,n0,pp,qmax,σ,dmin,state)
# Check for deflation, compute shift, and invoke dqds.
# In case of failure, change shifts and try again.
# translation of DLASQ3 from LAPACK
# Documentation for dlasq3 suggests it doesn't use old σ or dmin,
# but that's wrong.
# Documentation claims deflation tests should be skipped if pp==2;
# that's not in the code, but the logic is incoherent without that.
function _dqds_3!(zv::AbstractArray{QDtet{T}},i0,n0in,pp,qmax,σ,dmin,state) where T
    z0 = zero(T)
    myeps = eps(T)
    tol = 100myeps
    tol2 = tol^2
    cbias = T(3)/2

    n0 = n0in
    dbg_iter = 0
    while true
        dbg_iter += 1
        if dbg_iter > 1000
            throw(ErrorException("effed up."))
        end
        # check for deflation
        if n0 < i0
            return n0,pp,dmin,σ
        end

        if n0 == i0
            # deflate 1
            if pp==0
                w1 = zv[n0].q + σ
            elseif pp==1
                w1 = zv[n0].qq + σ
            else
                throw(ErrorException("simple deflation w/ pp=2: logic error"))
            end
            _save_eig!(zv, n0, w1)
            n0 -= 1
            continue
        end

        if n0 == i0+1
            deflate2 = true
        else
            if pp==0
                e1 = zv[n0-1].e
                ee1 = zv[n0-1].ee
                q1 = zv[n0-1].q
                w1 = σ + zv[n0].q
                e2 = zv[n0-2].e
                ee2 = zv[n0-2].ee
                q2 = zv[n0-2].q
            else
                e1 = zv[n0-1].ee
                ee1 = zv[n0-1].e
                q1 = zv[n0-1].qq
                w1 = σ + zv[n0].qq
                e2 = zv[n0-2].ee
                ee2 = zv[n0-2].e
                q2 = zv[n0-2].qq
            end
            if pp==2
                deflate2 = false
            else
                # if E[n0-1] is negligible, deflate 1
                if (e1 <= tol2 * w1) || (ee1 <= tol2 * q1)
                    _save_eig!(zv, n0, w1)
                    n0 -= 1
                    continue
                end
                deflate2 = (e2 <= tol2 * σ) || (ee2 <= tol2 * q2)
            end
        end
        if !deflate2
            break
        end

        if pp==0
            w1, w2 = _qd_2x2(zv[n0-1].q, zv[n0-1].e, zv[n0].q, tol2)
        else
            w1, w2 = _qd_2x2(zv[n0-1].qq, zv[n0-1].ee, zv[n0].qq, tol2)
        end
        w1 += σ
        w2 += σ
        _save_eig!(zv,n0,w1)
        _save_eig!(zv,n0-1,w2)
        n0 -= 2
        continue
    end # deflation loop

    if pp == 2
        pp = 0
    end
    # reverse qd array if warranted
    if dmin <= 0 || n0 < n0in
        flipped, xdmin2, xqmax = _maybe_flip!(zv, i0, n0, Val(pp))
        if flipped
            qmax = max( qmax, xqmax)
            state.dmin2 = min( state.dmin2, xdmin2)
            dmin = -z0
        end
    end # if reversing

    # choose a shift
    _dqds_4!(state, zv, i0, n0, pp, n0in, dmin)

    underflag = false
    while true
        if n0 > i0+1
            dmin = _dqds_5!(state,zv,i0,n0,pp,σ,myeps)
        end
        state.ndiv += n0-i0+2
        state.iter += 1
        if dmin >= 0 && state.dmin1 >= 0
            # success
            break
        end
        etest = (pp == 0) ? zv[n0-1].ee : zv[n0-1].e
        if   ((dmin < 0) && (state.dmin1 > 0)
              && (etest < tol * (σ + state.dn1))
              && (abs(state.dn) < tol*σ)
              )
            # convergence hidden by negative dn
            if pp==0
                zv[n0] = QDtet(zv[n0].q, z0, zv[n0].e, zv[n0].ee)
            else
                zv[n0] = QDtet(z0, zv[n0].qq, zv[n0].e, zv[n0].ee)
            end
            dmin = z0
            break
        elseif dmin < 0
            # τ too big; select another and try again
            state.nfail += 1
            if state.ttype < -22
                # failed twice; play it safe.
                state.τ = z0
            elseif state.dmin1 > 0
                # late failure: gives excellent shifts
                state.τ = (state.τ + dmin) * (1 - 2*myeps)
                state.ttype -= 11
            else
                # early failure
                state.τ /= 4
                state.ttype -= 12
            end
            continue
        elseif isnan(dmin)
            if state.τ == 0
                underflag = true
                break
            else
                state.τ = z0
                continue
            end
        else
            underflag = true
            break
        end
    end # dqds loop
    if underflag
        dmin = _dqd_6!(state, zv,i0,n0,Val(pp))
        state.ndiv += n0-i0+2
        state.iter += 1
        state.τ = z0
    end

    if state.τ < σ
        state.δσ += state.τ
        t = σ + state.δσ
        state.δσ -= (t - σ)
    else
        t = σ + state.τ
        state.δσ = σ - (t - state.τ) + state.δσ
    end
    σ = t
    return n0,pp,dmin,σ
end

function _maybe_flip!(zv::AbstractVector{QDtet{T}}, i0, n0, pp) where T
    z0 = T(0)
    cbias = T(3)/T(2)
    if pp==0
        flipping = cbias * zv[i0].q < zv[n0].q
    else
        flipping = cbias * zv[i0].qq < zv[n0].qq
    end
    if flipping
        ip = i0+n0
        for j in i0:(i0+n0-1) >> 1
            zd0 = QDtet(zv[ip-j].q, zv[ip-j].qq, zv[ip-j-1].e, zv[ip-j-1].ee)
            zv[ip-j-1] = QDtet(zv[ip-j-1].q, zv[ip-j-1].qq, zv[j].e, zv[j].ee)
            zv[ip-j] = QDtet(zv[j].q, zv[j].qq, zv[ip-j].e, zv[ip-j].ee)
            zv[j] = zd0
        end
        if n0-i0 <= 4
            zv[n0] = QDtet(zv[n0].q, zv[n0].qq, zv[i0].e, zv[i0].ee)
        end
        if pp==0
            dmin2 = zv[n0].e # Z[4*n0+pp-1]
            qmax = max( zv[i0].q, zv[i0+1].q)
        else
            dmin2 = zv[n0].ee # Z[4*n0+pp-1]
            qmax = max( zv[i0].qq, zv[i0+1].qq)
        end
        e1 = min( zv[n0].e, zv[i0].e, zv[i0+1].e)
        ee1 = min( zv[n0].ee, zv[i0].ee, zv[i0+1].ee)
        zv[n0] = QDtet(zv[n0].q, zv[n0].qq, e1, ee1)
    else
        dmin2, qmax = z0, z0
    end
    return flipping, dmin2, qmax
end

# compute an approximation to the smallest eigenvalue using values of d
# from previous transform.
# modifies `state`: sets τ, ttype;  updates g
# translation of DLASQ4 from LAPACK

# NOTE: Fernando (note, Feb 2020) claims that one should assign
#  state.τ = s
# before all of the unusual return statements.
# I haven't found the rationale for these unusual branches, so am puzzled.
function _dqds_4!(state, zv::AbstractArray{QDtet{T}}, i0, n0, pp, n0in, dmin) where T
    z0 = zero(T)
    const1 = 0.563
    const2 = 1.01
    const3 = 1.04
    if dmin <= 0
        state.τ = -dmin
        state.ttype = -1
        return
    end
    if pp == 0
        q0 =  zv[n0].q
        q1 = zv[n0-1].q
        q2 = zv[n0-2].q
        e1 = zv[n0-1].e
        e2 = zv[n0-2].e
        if n0-i0>2
            q3 = zv[n0-3].q
            e3 = zv[n0-3].e
        end
        ee1 = zv[n0-1].ee
        ee2 = zv[n0-2].ee
    else
        q0 =  zv[n0].qq
        q1 = zv[n0-1].qq
        q2 = zv[n0-2].qq
        e1 = zv[n0-1].ee
        e2 = zv[n0-2].ee
        if n0-i0>2
            q3 = zv[n0-3].qq
            e3 = zv[n0-3].ee
        end
        ee1 = zv[n0-1].e
        ee2 = zv[n0-2].e
    end

    if n0in == n0
        # no eigenvalues deflated
        if dmin == state.dn || dmin == state.dn1
            b1 = sqrt(q0) * sqrt(e1)
            b2 = sqrt(q1) * sqrt(e2)
            a2 = q1 + e1

            # cases 2 and 3
            if dmin == state.dn && state.dmin1 == state.dn1
                gap2 = state.dmin2 - a2 - state.dmin2 / 4
                if gap2 > 0 && gap2 > b2
                    gap1 = a2 - state.dn - (b2 / gap2) * b2
                else
                    gap1 = a2 - state.dn - (b1 + b2)
                end
                if gap1 > 0 && gap1 > b1
                    s = max(state.dn - (b1 / gap1) * b1, dmin * (1/T(2)))
                    state.ttype = -2
                else
                    s = z0
                    if state.dn > b1
                        s = state.dn - b1
                    end
                    if a2 > (b1 + b2)
                        s = min(s, a2 - (b1+b2))
                    end
                    s = max( s, dmin / 3)
                    state.ttype = -3
                end
            else
                # case 4
                state.ttype = -4
                s = dmin / 4
                if dmin == state.dn
                    γ = state.dn
                    a2 = z0
                    if e1 > q1
                        return
                    end
                    b2 = e1 / q1
                    npp = n0 - 2
                else
                    γ = state.dn1
                    if pp==0
                        if zv[n0-1].ee > zv[n0-1].qq
                            return
                        end
                        a2 = zv[n0-1].ee / zv[n0-1].qq
                        if e2 > q2
                            return
                        end
                    else
                        if zv[n0-1].e > zv[n0-1].q
                            return
                        end
                        a2 = zv[n0-1].e / zv[n0-1].q
                        if e2 > q2
                            return
                        end
                    end
                    b2 = e2 / q2
                    npp = n0-3
                end
                # approximate contribution to norm squared from i < nn-1
                a2 += b2
                if pp==0
                    @inbounds for i in npp:-1:i0
                        if b2 == 0
                            break
                        end
                        b1 = b2
                        if zv[i].ee > zv[i].qq
                            return
                        end
                        b2 *= zv[i].ee / zv[i].qq
                        a2 += b2
                        if 100*max(b1,b2) < a2 || const1 < a2
                            break
                        end
                    end
                else
                    @inbounds for i in npp:-1:i0
                        if b2 == 0
                            break
                        end
                        b1 = b2
                        if zv[i].e > zv[i].q
                            return
                        end
                        b2 *= zv[i].e / zv[i].q
                        a2 += b2
                        if 100*max(b1,b2) < a2 || const1 < a2
                            break
                        end
                    end
                end
                a2 *= const3
                # Rayleigh quotient residual bound
                if a2 < const1
                    s = γ * (1 - sqrt(a2)) / (1 + a2)
                end
            end
        elseif dmin == state.dn2
            # case 5
            state.ttype = -5
            s = dmin / 4
            # compute contribution to norm squared from i > nn-2
            if pp == 0
                b1 = zv[n0].qq
                b2 = zv[n0-1].qq
                γ = state.dn2
                if zv[n0-2].ee > b2 || zv[n0-1].ee > b1
                    return
                end
                a2 = (zv[n0-2].ee / b2) * (1 + zv[n0-1].ee / b1)
            else
                b1 = zv[n0].q
                b2 = zv[n0-1].q
                γ = state.dn2
                if zv[n0-2].e > b2 || zv[n0-1].e > b1
                    return
                end
                a2 = (zv[n0-2].e / b2) * (1 + zv[n0-1].e / b1)
            end
            # approximate contribution from i < nn-2
            if n0-i0>2
                b2 = e3 / q3
                a2 += b2
                if pp == 0
                  @inbounds for i in n0-4:-1:i0
                    if b2 == 0
                        break
                    end
                    b1 = b2
                    if zv[i].e > zv[i].q
                        return
                    end
                    b2 *= zv[i].e / zv[i].q
                    a2 += b2
                    if 100 * max(b2, b1) < a2 || const1 < a2
                        break
                    end
                  end
                else
                  @inbounds for i in n0-4:-1:i0
                    if b2 == 0
                        break
                    end
                    b1 = b2
                    if zv[i].ee > zv[i].qq
                        return
                    end
                    b2 *= zv[i].ee / zv[i].qq
                    a2 += b2
                    if 100 * max(b2, b1) < a2 || const1 < a2
                        break
                    end
                  end
                end
                a2 *= const3
            end
            if a2 < const1
                s = γ * (1 - sqrt(a2)) / (1 + a2)
            end
        else
            # case 6, no information
            if state.ttype == -6
                state.g += (1 - state.g) / T(3)
            elseif state.ttype == -18
                state.g = 1 / T(12)
            else
                state.g = 1 / T(4)
            end
            s = state.g * dmin
            state.ttype = -6
        end
    elseif n0in == n0+1
        # one eigval deflated; use dmin1, dn1
        if state.dmin1 == state.dn1 && state.dmin2 == state.dn2
            # cases 7 and 8
            state.ttype = -7
            s = state.dmin1 * (1/T(3))
            if e1 > q1
                return
            end
            b1 = e1 / q1
            b2 = b1
            if b2 != 0
                if pp == 0
                    @inbounds for i in (n0-2):-1:i0

                        a2 = b1
                        if zv[i].e > zv[i].q
                            return
                        end
                        b1 *= zv[i].e / zv[i].q
                        b2 += b1
                        if 100 * max(b1,a2) < b2
                            break
                        end
                    end
                else
                    @inbounds for i in (n0-2):-1:i0
                        a2 = b1
                        if zv[i].ee > zv[i].qq
                            return
                        end
                        b1 *= zv[i].ee / zv[i].qq
                        b2 += b1
                        if 100 * max(b1,a2) < b2
                            break
                        end
                    end
                end
            end
            b2 = sqrt(const3 * b2)
            a2 = state.dmin1 / (1 + b2^2)
            gap2 = state.dmin2 * (1/T(2)) - a2
            if gap2 > 0 && gap2 > b2*a2
                s = max(s, a2 * (1-const2*a2*(b2/gap2)*b2))
            else
                s = max(s, a2*(1-const2*b2))
                state.ttype = -8
            end
        else
            # case 9
            s = state.dmin1 / 4
            if state.dmin1 == state.dn1
                s = state.dmin1 * (1/T(2))
            end
            state.ttype = -9
        end
    elseif n0in == n0+2
        # two eigvales deflated; use dmin2, dn2
        if state.dmin1 == state.dn1 && state.dmin2 == state.dn2
            # cases 10 and 11
            state.ttype = -10
            s = state.dmin1 / 3
            if e1 > q1
                return
            end
            b1 = e1 / q1
            b2 = b1
            if b2 != 0
                if pp == 0
                  @inbounds for i in (n0-2):-1:i0
                    if zv[i].e > zv[i].q
                        return
                    end
                    b1 *= zv[i].e / zv[i].q
                    b2 += b1
                    if 100 * b1 < b2
                        break
                    end
                  end
                else
                  @inbounds for i in (n0-2):-1:i0
                    if zv[i].ee > zv[i].qq
                        return
                    end
                    b1 *= zv[i].ee / zv[i].qq
                    b2 += b1
                    if 100 * b1 < b2
                        break
                    end
                  end
                end
            end
            b2 = sqrt(const3 * b2)
            a2 = state.dmin2 / (1 + b2^2)
            gap2 = q1 + e2 - sqrt(q2) * sqrt(e2) - a2
            if gap2 > 0 && gap2 > b2*a2
                s = max(s, a2 * (1-const2*a2*(b2/gap2)*b2))
            else
                s = max(s, a2*(1-const2*b2))
            end
        else
            # case 11
            s = state.dmin2 / 4
            state.ttype = -11
        end
    elseif n0in > n0+2
        # case 12: more than 2 eigvals deflated; no information
        s = z0
        state.ttype = -12
    end
    state.τ = s
end

# compute one dqds transform in ping-pong form
# "translation" of DLASQ5 from LAPACK
function _dqds_5!(state,zv::AbstractArray{QDtet{T}},i0,n0,pp,σ,ϵ) where T
    __dqds_5!(state,zv,i0,n0,Val(pp),σ,ϵ,FPModel(T))
end

# I could repeat myself, or I could spend a few hours trying to optimize some metacode...
for (pp, qf, ef, update) in
    ((0, :q, :e, :update0!),
     (1, :qq, :ee, :update1!))
 @eval begin
  function __dqds_5!(state,zv::AbstractArray{QDtet{T}},i0,n0,::Val{$pp},σ,ϵ,::IEEEFPModel) where T
    z0 = zero(T)
    dthresh = ϵ * (σ + state.τ)
    negl_τ = state.τ < dthresh * (1/T(2))
    if negl_τ
        state.τ = z0
    end
    @inbounds begin
        # we set d's to 0 if they are small enough
        # if branches are costly, consider dispatching on negl_τ
        emin = zv[i0+1].$qf
        d = zv[i0].$qf - state.τ
        dmin = d
        state.dmin1 = -zv[i0].$qf
        @inbounds for j in i0:(n0-3)
            qnew = d + zv[j].$ef
            t = zv[j+1].$qf / qnew
            d = d * t - state.τ
            if negl_τ && (d < dthresh)
                d = z0
            end
            dmin = min( dmin, d)
            enew = zv[j].$ef * t
            emin = min( enew, emin)
            $update(zv,j,qnew,enew)
        end

        # unroll last two steps
        state.dn2 = d
        state.dmin2 = dmin
        j = (n0-2)
        qnew = state.dn2 + zv[j].$ef
        enew = zv[j+1].$qf * (zv[j].$ef / qnew)
        state.dn1 = zv[j+1].$qf * (state.dn2 / qnew) - state.τ
        $update(zv,j,qnew,enew)
        dmin = min(dmin, state.dn1)

        state.dmin1 = dmin
        j += 1
        qnew = state.dn1 + zv[j].$ef
        enew = zv[j+1].$qf * (zv[j].$ef / qnew)
        state.dn = zv[j+1].$qf * (state.dn1 / qnew) - state.τ
        $update(zv,j,qnew,enew)
        dmin = min(dmin, state.dn)

        update!(zv,n0,Val($pp),state.dn,emin)
    end
      return dmin
  end
# compute one dqd (zero shift) transform in ping-pong form,
# with over/underflow handling.
# Translation of DLASQ6 from LAPACK
  function _dqd_6!(state, zv::AbstractArray{QDtet{T}},i0,n0,::Val{$pp}) where T
    z0 = zero(T)
    safmin = safemin(T)
    emin = zv[i0+1].$qf
    d = zv[i0].$qf
    dmin = d

    for j in i0:(n0-3)
        qnew = d + zv[j].$ef
        if qnew == 0
            enew = z0
            d = zv[j+1].$qf
            dmin = d
            emin = z0
        elseif (safmin * zv[j+1].$qf < qnew) && (safmin * qnew < zv[j+1].$qf)
            t = zv[j+1].$qf / qnew
            d *= t
            enew = zv[j].$ef * t
        else
            enew = zv[j+1].$qf * (zv[j].$ef / qnew)
            d = zv[j+1].$qf * (d / qnew)
        end
        update!(zv,j,Val($pp),qnew,enew)
        dmin = min(dmin, d)
        emin = min(emin, enew)
    end

    # unroll last two steps
    state.dn2 = d
    state.dmin2 = dmin
    j = (n0-2)
    qnew = state.dn2 + zv[j].$ef
    if qnew == 0
        enew = z0
        state.dn1 = zv[j+1].$qf
        dmin = state.dn1
        emin = z0
    elseif (safmin * zv[j+1].$qf < qnew) && (safmin * qnew < zv[j+1].$qf)
        t = zv[j+1].$qf / qnew
        enew *= t
        state.dn1 = state.dn2 * t
    else
        enew = zv[j+1].$qf * (zv[j].$ef / qnew)
        state.dn1 = zv[j+1].$qf * (state.dn2 / qnew)
    end
    update!(zv,j,Val($pp),qnew,enew)
    dmin = min(dmin, state.dn1)

    state.dmin1 = dmin
    j += 1
    qnew = state.dn1 + zv[j].$ef
    if qnew == 0
        enew = z0
        state.dn = zv[j+1].$qf
        dmin = state.dn
        emin = z0
    elseif (safmin * zv[j+1].$qf < qnew) && (safmin * qnew < zv[j+1].$qf)
        t = zv[j+1].$qf / qnew
        enew = zv[j].$ef * t
        state.dn = state.dn1 * t
    else
        enew = zv[j+1].$qf * (zv[j].$ef / qnew)
        state.dn = zv[j+1].$qf * (state.dn1 / qnew)
    end
    update!(zv,j,Val($pp),qnew,enew)
    dmin = min(dmin, state.dn)

    update!(zv,n0,Val($pp),state.dn,emin)

    return dmin
  end # function
 end # eval block
end # pp loop

end # module
