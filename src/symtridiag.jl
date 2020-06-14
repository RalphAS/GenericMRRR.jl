# Schur decomposition for dense real symmetric tridiagonal matrices

# This file is part of GenericMRRR.jl, released under the MIT "Expat" license

# The methods in this file are translated from
# the Multiple Relatively Robust Represententation routines in LAPACK.
# LAPACK authors are (obviously) not responsible for translation errors.

# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.

# Original authors of the LAPACK Fortran version:
#  Beresford Parlett, University of California, Berkeley, USA
#  Jim Demmel, University of California, Berkeley, USA
#  Inderjit Dhillon, University of Texas, Austin, USA
#  Osni Marques, LBNL/NERSC, USA
#  Christof Vömel, University of California, Berkeley, USA

abstract type EigvalSelector end
struct AllEigvalSelector <: EigvalSelector
end
struct IndexedEigvalSelector <: EigvalSelector
    il::Int
    iu::Int
end
struct IntervalEigvalSelector{T} <: EigvalSelector
    vl::T
    vu::T
end

# consolidate for internal use
struct RealEigvalSelector{T}
    vl::T
    vu::T
    il::Int
    iu::Int
    style::Symbol
end

function scale_eigval_selector(sel::RealEigvalSelector, scale)
    RealEigvalSelector(scale*s.vl,scale*s.vu,s.il,s.iu,s.style)
end
_embed_selector(::AllEigvalSelector, ::Type{T}) where {T} =  RealEigvalSelector(
    T(0),T(0),0,0,:all)
_embed_selector(s::IntervalEigvalSelector, ::Type{T}) where {T} =  RealEigvalSelector(
    T(s.vl),T(s.vu),0,0,:val)
_embed_selector(s::IndexedEigvalSelector, ::Type{T}) where {T} =  RealEigvalSelector(
    T(0),T(0),s.il,s.iu,:idx)

# if this is true, don't use the dqds module
const force_bisect = Ref(false)

const force_RAC = Ref(false)

"""
a "safe" version of `floatmin`, such that `1/sfmin` does not overflow.
"""
function safemin(T)
    sfmin = floatmin(T)
    small = one(T) / floatmax(T)
    if small >= sfmin
        sfmin = small * (one(T) + eps(T))
    end
    sfmin
end

# _st_schur!(D, E; kwargs...)
# Compute eigenvalues and optionally eigenvectors of a real symmetric
# tridiagonal matrix SymTridiagonal(D,E).
# Order of results is complicated; sort in a wrapper if desired.
#
# Options:
#  `wantV::Bool` - whether to compute eigenvectors
#  `tryRAC` - whether to try for high relative accuracy (usu. true/false)
#
#  If `tryRAC == :force`, do the refinement step w/o question.
#   (This is mainly to insure that the code works for cases where
#   extra information is available, pending discovery of more comprehensive
#   practical tests.)
#
#  min_relgap is a kwarg here to facilitate charting the wild territory of
#   interesting precisions.
#
# translation of STEMR from LAPACK
function _st_schur!(D::AbstractVector{RT}, E::AbstractVector{RT};
                    wantV::Bool = false,
                    tryRAC::Union{Bool,Symbol} = true,
                    select::EigvalSelector = AllEigvalSelector(),
                    min_relgap = RT(1e-3)
                 ) where RT <: AbstractFloat

    n = length(D)

    select1 = _embed_selector(select, RT)
    if (select1.style != :all) && (n < 3)
        throw(ArgumentError("subselection is not implemented for n < 3"))
    end
    # TODO: other sanity checks on select

    if n == 0
        x = Vector{RT}(undef,0)
        return x, wantV ? Matrix{RT}(undef,0,0) : nothing
    elseif n == 1
        λ = [D[1]]
        if wantV
            vectors = fill(RT(1),1,1)
            return λ, vectors
        else
            return λ, nothing
        end
    elseif n == 2
        λ, vectors = _eig_sym2x2(D[1], E[1], D[2], wantV)
        return λ, vectors
    end

    myeps = eps(RT)
    safmin = safemin(RT)
    smallnum = safmin / eps(RT)
    bignum = 1 / smallnum
    rmin = sqrt(smallnum)
    rmax = min(sqrt(bignum), 1 / sqrt(sqrt(safmin)))


    # scale matrix to allowable range
    scale = one(RT)
    tnrm = max(maximum(abs.(D)),maximum(abs.(E[1:n-1])))
    if tnrm > 0 && tnrm < rmin
        scale = rmin / tnrm
    elseif tnrm > rmax
        scale = rmax / tnrm
    end
    if scale != 1
        lmul!(scale, D)
        lmul!(scale, view(E,1:n-1))
        tnrm *= scale
        if select1.style == :val
            select1 = scale_eigval_selector(select1, scale)
        end
    end

    # compute eigenvalues after splitting into smaller subblocks
    # if off-diagonals are small

    thresh = myeps # split tolerance
    # test whether expensive relative approach is warranted
    if tryRAC == true
        rac = _check_RAC(D, E)
        # FIXME: optionally report this; it's too noisy for normal use
    else
        rac = tryRAC != :force
    end

    Esq = E .* E

    if rac
        D_orig = copy(D) # needed to guarantee relative accuracy
        Esq_orig = copy(Esq)
    end

    if wantV
        # we will improve accuracy later
        rtol1 = sqrt(myeps)
        rtol2 = max(sqrt(myeps) / 256, 4 * myeps)
    else
        rtol2 = rtol1 = 4 * myeps
    end
    λ, w_err, w_gap, m, blocks, bounds = _st_vals!(D, E, Esq, rtol1, rtol2, thresh, rac, select1)
    wl, wu = bounds

    if wantV

        # stuff working wl, wu into args
        # Note: DLARRV claims it will work w/ an index subset, but callers always
        # seem to specify 1:m.
        select1 = RealEigvalSelector(wl,wu,1,m,select1.style)

        vectors, = _st_vectors!(D, E, blocks,
                                min_relgap, rtol1, rtol2, λ, w_err, w_gap,
                                select1)
    else
        # just apply shifts
        for j in 1:m
            i = blocks.iblock[j]
            λ[j] += E[blocks.isplit[i]]
        end
        vectors = nothing
    end
    if rac
        # refine computed eigenvalues to relative accuracy wrt original mtx
        ibegin = 1
        w_begin = 1
        for jblk in 1:blocks.iblock[m]
            iend = blocks.isplit[jblk]
            bsize = iend - ibegin + 1
            w_end = w_begin - 1
            # check if any eigvals in this block need refinement
            while w_end < m
                if blocks.iblock[w_end+1] == jblk
                    w_end += 1
                else
                    break
                end
            end
            if w_end < w_begin
                # jblk is done
                ibegin = iend + 1
                continue
            end
            ifirst = blocks.indexw[w_begin]
            offset = ifirst - 1
            ilast = blocks.indexw[w_end]
            rtol2 = 4myeps

            _st_refine!( bsize, view(D_orig,ibegin:iend),
                            view(Esq_orig,ibegin:iend-1),
                            ifirst, ilast, rtol2, offset,
                            view(λ,w_begin:w_end),
                            view(w_err,w_begin:w_end),
                            blocks.pivmin, tnrm)
            ibegin = iend + 1
            w_begin = w_end + 1
        end
    end
    if scale != 1
        λ .*= (1/scale)
    end

    return λ, vectors
end

# computes the eigenvectors of the tridiagonal matrix
# T = L D Lᵀ given L, D and APPROXIMATIONS to the eigenvalues of L D Lᵀ.
# The input eigenvalues should have been computed by _st_vals!()
# translation of DLARRV from LAPACK
# modifies D, L (many times!)
function _st_vectors!(D::AbstractVector{T}, L, blocks,
                       min_relgap, rtol1, rtol2, w, w_err, w_gap,
                       select) where T
    n = length(D)
    isplit = blocks.isplit
    iblock = blocks.iblock
    indexw = blocks.indexw
    gersl = blocks.gersl
    gersu = blocks.gersu
    pivmin = blocks.pivmin
    m = blocks.m
    vl = select.vl
    vu = select.vu
    do_l = select.il
    do_u = select.iu

    # apparently some blocks are not set later, so initializing to 0 is needed
    Z = zeros(T,n,m)
    supportZ = fill(-1,2,m) # provoke bound errors in case of misuse

    # intermediate L.*D and L.*D.*L are used many times, so we will save them
    LD = Vector{T}(undef,n)
    LLD = Vector{T}(undef,n)

    maxit = 10
    myeps = eps(T)
    rqtol = 2myeps
    z0 = zero(T)
    tryRQC = true

    w_temp = Vector{T}(undef,n)

    # Some of the following initial zeros are important.

    # Indices for factorization used to get Fernando-Parlett (FP) vector
    # 0 here means to get it from the representation (and then save it)
    twist_inds = fill(0,n)

    # Clusters of current layer
    new_cl = fill(0,2,n>>>1)
    # Clusters of layer above this one
    old_cl = fill(0,2,n>>>1)

    if do_l != 1 || do_u != m
        # if only selected eigenpairs are computed, remainder are not refined,
        # so bisection must compute to full accuracy.
        rtol1 = 4myeps
        rtol2 = 4myeps
    end

    # indices of desired eigvals are w_begin:w_end
    # support of nonzero eigvecs is ibegin:iend

    ndone = 0 # count of computed eigenvectors
    ibegin = 1
    w_begin = 1
    for jblk in 1:blocks.iblock[m]
        iend = blocks.isplit[jblk]
        σ = L[iend]
        bsize = iend - ibegin + 1
        w_end = w_begin - 1
        while w_end < m
            if blocks.iblock[w_end+1] == jblk
                w_end += 1
            else
                break
            end
        end

        # maybe jblk is done
        if w_end < w_begin
            ibegin = iend + 1
            continue
        elseif w_end < do_l || w_begin > do_u
            ibegin = iend + 1
            w_begin = w_end + 1
            continue
        end

        # find spectral diameter of block
        gl = minimum(view(gersl,ibegin:iend))
        gu = maximum(view(gersu,ibegin:iend))

        spdiam = gu - gl
        old_iend = ibegin - 1
        bsize = iend - ibegin + 1
        mblock = w_end - w_begin + 1 # IM in DLARRV
        # 1x1
        if bsize == 1
            ndone += 1
            Z[ibegin, w_begin] = 1
            supportZ[:,w_begin] .= ibegin
            w[w_begin] += σ
            w_temp[w_begin] = w[w_begin]
            ibegin += 1
            w_begin += 1
            continue
        end

        # Desired shifted eigvals are in w[w_begin:w_end], possibly approximated
        # w_err holds uncertainty intervals
        # Approximations will be refined when necessary as high relative
        # accuracy is required for eigenvector computation.
        w_temp[w_begin:w_end] .= view(w,w_begin:w_end)
        # Approximations w.r.t. original matrix

        w[w_begin:w_end] .+= σ

        # start at root: initialize descriptor
        ndepth = 0 # current depth of representation tree
        ncluster = 1 # count for next level of representation tree; 1 for root
        new_cl[1,1] = 1
        new_cl[2,1] = mblock

        workvecs = [Vector{T}(undef, bsize) for j in 1:4]
        idone = 0 # nr of eigvecs done in current block
        while idone < mblock
            # generate representation tree for current block; compute eigvecs
            if ndepth > m
                throw(ErrorException("algorithm failure: tree too deep"))
            end
            # breadth first processing of current level of tree
            old_ncluster = ncluster
            ncluster = 0
            old_cl, new_cl = new_cl, old_cl

            # process clusters on current level
            for i in 1:old_ncluster
                # cluster indices (org 1) are relative to w_begin for w etc.
                oldfirst = old_cl[1,i]
                oldlast = old_cl[2,i]
                if ndepth > 0
                    # RRR of cluster that was computed at previous level
                    # is stored in Z and overwritten once eigvecs have been
                    # computed or when cluster is refined

                    # usu. get representation from location of leftmost eigval
                    j = w_begin + oldfirst - 1
                    if do_l != 1 || do_u != m
                        if j < do_l
                            # use left end of Z array instead
                            j = do_l - 1
                        elseif j > do_u
                            # use right end of Z array instead
                            j = do_u
                        end
                    end
                    D[ibegin:iend] .= Z[ibegin:iend, j]
                    L[ibegin:iend-1] .= Z[ibegin:iend-1, j+1]
                    σ = Z[iend, j+1]
                    Z[ibegin:iend, j:j+1] .= z0
                end
                for j in ibegin:iend-1
                    t = D[j] * L[j]
                    LD[j] = t
                    LLD[j] = t * L[j]
                end
                if ndepth > 0
                    # index range of eigvals to compute within current block
                    p = indexw[w_begin - 1 + oldfirst]
                    q = indexw[w_begin - 1 + oldlast]
                    # offset for w etc.
                    offset = indexw[w_begin] - 1
                    # refine approx eigvals to needed precision
                    _bisect_limited!(bsize, view(D,ibegin:iend),
                                     view(LLD,ibegin:iend),
                                     p, q, rtol1, rtol2, offset,
                                     view(w_temp, w_begin:w_end),
                                     view(w_gap, w_begin:w_end),
                                     view(w_err, w_begin:w_end),
                                     pivmin, spdiam, bsize)
                    # recompute extremal gaps
                    if oldfirst > 1
                        itmp = w_begin + oldfirst - 2
                        w_gap[itmp] = max(w_gap[itmp], w[itmp+1] - w_err[itmp+1]
                                          - w[itmp] - w_err[itmp])
                    end
                    itmp = w_begin + oldlast - 1
                    if itmp < w_end
                        w_gap[itmp] = max(w_gap[itmp], w[itmp+1] - w_err[itmp+1]
                                          - w[itmp] - w_err[itmp])
                    end
                    # each time eigvals in w_temp get refined, store
                    # new approximants with shifts in w
                    for j in w_begin+oldfirst-1:w_begin+oldlast-1
                        w[j] = w_temp[j] + σ
                    end
                end # if ndepth > 0

                # process current node
                newfirst = oldfirst
                jbest = oldfirst
                bestrgap = z0
                j = oldfirst-1
                # for j in oldfirst:oldlast
                while j < oldlast
                    j += 1
                    if j == oldlast
                        # right end of cluster; boundary of child
                        newlast = j
                    elseif w_gap[w_begin + j-1] >= min_relgap * abs(w_temp[w_begin + j-1])
                        # right relative gap is big enough;
                        # child cluster newfirst:newlast is well separated
                        # from the follower
                        newlast = j
                    else
                        rgap = w_gap[w_begin + j-1] / abs(w_temp[w_begin + j-1])
                        if (j == oldfirst) || (rgap > bestrgap)
                            bestrgap = rgap
                            jbest = j
                        end
                        # inside child cluster; relative gap is not big enough
                        continue
                        # continue
                    end

                    # *RAS* experimentally adding next branch
                    have_repn = false
                    for ic=1:old_ncluster
                        if (old_cl[1,ic] == newfirst) && (old_cl[2,ic] == newlast)
                            have_repn = true
                        end
                    end
                    if have_repn
                        newlast = jbest
                        j = newlast
                    end

                    # compute size of child cluster
                    newsize = newlast - newfirst + 1
                    # set `newftt`, the column in Z where new RRR or new eigvec goes
                    if do_l == 1 && do_u == m
                        newftt = w_begin + newfirst - 1
                    else
                        if w_begin + newfirst - 1 < do_l
                            # use left end of Z
                            newftt = do_l - 1
                        elseif w_begin + newfirst-1 > do_u
                            # use right end of Z
                            newftt = do_u
                        else
                            newftt = w_begin + newfirst - 1
                        end
                    end
                    if newsize > 1
                        # current child is not a singleton but a cluster
                        # compute and store new representation of child

                        # compute left and right cluster gap
                        # LGAP and RGAP are not computed from W_TEMP because
                        # the eigenvalue approximations may stem from RRRs
                        # different shifts. However, W holds all eigenvalues
                        # of the unshifted matrix. Still, the entries in W_GAP
                        # have to be computed from W_TEMP since the entries
                        # in W might be of the same order so that gaps are not
                        # exhibited correctly for very close eigenvalues.

                        # println("w: ", w[w_begin:w_end])
                        # println("w_temp: ", w_temp[w_begin:w_end])
                        if newfirst == 1
                            lgap = max(z0, w[w_begin] - w_err[w_begin] - vl)
                        else
                            lgap = w_gap[w_begin + newfirst - 2]
                        end
                        rgap = w_gap[w_begin + newlast - 1]

                        # Compute left- and rightmost eigenvalue of child
                        # to high precision in order to shift as close
                        # as possible and obtain as large relative gaps
                        # as possible
                        for p in (indexw[w_begin-1+newfirst],
                                  indexw[w_begin-1+newlast])
                            offset = indexw[w_begin] - 1
                            _bisect_limited!(bsize, view(D,ibegin:iend),
                                             view(LLD,ibegin:iend),
                                             p, p, rqtol, rqtol, offset,
                                             view(w_temp, w_begin:w_end),
                                             view(w_gap, w_begin:w_end),
                                             view(w_err, w_begin:w_end),
                                             pivmin, spdiam, bsize, verbose=false)
                        end
                        if w_begin + newlast - 1 < do_l || w_begin + newfirst - 1 > do_u
                            # if the cluster contains no desired eigenvalues
                            # skip the computation of that branch of the rep. tree

                            # We could skip before the refinement of the extremal
                            # eigenvalues of the child, but then the representation
                            # tree could be different from the one when nothing is
                            # skipped. For this reason we skip at this place.
                            idone += newlast - newfirst + 1
                            # proceed to any remaining child nodes
                            newfirst = j + 1
                            continue # CHECKME: goes to next child (j)
                        end

                        # compute RRR of child cluster, storing in Z
                        τ, = _st_rrr!(bsize, view(D,ibegin:iend),
                                      view(L,ibegin:iend),
                                      view(LD,ibegin:iend),
                                      newfirst, newlast,
                                      view(w_temp,w_begin:w_end),
                                      view(w_gap,w_begin:w_end),
                                      view(w_err,w_begin:w_end),
                                      spdiam, lgap, rgap, pivmin,
                                      view(Z,ibegin:iend,newftt),
                                      view(Z,ibegin:iend,newftt+1)
                                      )
                        # Update shift and store it.
                        sσ = σ + τ
                        Z[iend, newftt+1] = sσ
                        # Store midpoints and semi-widths.
                        # Note that `w` is unchanged.
                        for k in newfirst:newlast
                            fudge = 3myeps * abs(w_temp[w_begin+k-1])
                            w_temp[w_begin+k-1] -= τ
                            fudge += 4myeps * abs(w_temp[w_begin+k-1])
                            w_err[w_begin+k-1] += fudge
                        end
                        ncluster += 1
                        new_cl[1,ncluster] = newfirst
                        new_cl[2,ncluster] = newlast
                    else
                        # compute eigenvector of singleton
                        iter = 0
                        tol = 4 * log(bsize) * myeps
                        k = newfirst
                        w_idx = w_begin + k - 1
                        w_idxmin = max(w_idx-1,1)
                        w_idxpl = min(w_idx+1,m)
                        λ = w_temp[w_idx]
                        ndone += 1
                        eskip =  w_idx < do_l || w_idx > do_u
                        if !eskip
                            left = w_temp[w_idx] - w_err[w_idx]
                            right = w_temp[w_idx] + w_err[w_idx]
                            indeig = indexw[w_idx]
                            # Since eigenpairs are computed together for a node,
                            # those approximations are all based on one shift.
                            # `w_temp` is used to compute the gaps; this
                            # is valuable if the ews cluster near the shift.
                            if k == 1
                                # force a small left gap
                                lgap = myeps * max(abs(left), abs(right))
                            else
                                lgap = w_gap[w_idxmin]
                            end
                            if k == mblock
                                # force a small right gap
                                rgap = myeps * max(abs(left), abs(right))
                            else
                                rgap = w_gap[w_idx]
                            end
                            gap = min(lgap, rgap)
                            if k == 1 || k == mblock
                                gaptol = z0
                            else
                                # DLARRV uses gap * ϵ
                                # *RAS* A more conservative value avoids some embarassment.
                                gaptol = gap * myeps / 8
                            end
                            isupmin = bsize
                            isupmax = 1
                            # Update w_gap to hold minimum to left or right.
                            # This is crucial in case bisection is used to
                            # ensure that eigval is refined to required precision.
                            save_gap = w_gap[w_idx]
                            w_gap[w_idx] = gap

                            # Prefer Rayleigh quotient correction because of quadratic
                            # convergence, but use bisection in case of sign error.
                            usedBS = false
                            usedRQ = false
                            # initially use bisection only if forced to
                            needBS = !tryRQC
                            vdone = false
                            local bestres, w_best, nrminv
                            while !vdone # 120
                                # check if bisection should be used to refine eigval
                                if needBS
                                    # take bisection as new iterate
                                    usedBS = true
                                    twist = twist_inds[w_idx]
                                    offset = indexw[w_begin] - 1

                                    _bisect_limited!(bsize, view(D,ibegin:iend),
                                                     view(LLD,ibegin:iend),
                                                     indeig, indeig, z0, 2myeps, offset,
                                                     view(w_temp, w_begin:w_end),
                                                     view(w_gap, w_begin:w_end),
                                                     view(w_err, w_begin:w_end),
                                                     pivmin, spdiam, twist,
                                                     verbose=false
                                                     )
                                    λ = w_temp[w_idx]
                                    # force computation of true minγ
                                    twist_inds[w_idx] = 0
                                end
                                # given λ, compute an eigenvector
                                negcnt, ztz, minγ, r, isuppz, convstuff = _st_get1ev(
                                    bsize, 1, bsize, λ, view(D,ibegin:iend),
                                       view(L,ibegin:iend-1),
                                       view(LD,ibegin:iend-1),view(LLD,ibegin:iend-1),
                                       pivmin, gaptol, view(Z,ibegin:iend,w_idx),
                                       !usedBS,
                                       twist_inds[w_idx], workvecs)
                                nrminv, resid, rqcorr = convstuff
                                twist_inds[w_idx] = r
                                supportZ[1:2,w_idx] .= isuppz

                                if iter == 0
                                    bestres = resid
                                    w_best = λ
                                elseif resid < bestres
                                    bestres = resid
                                    w_best = λ
                                end
                                isupmin = min(isupmin,supportZ[1,w_idx])
                                isupmax = max(isupmax,supportZ[2,w_idx])
                                iter += 1

                                # sin α <= |resid| / gap

                                if ((resid > tol * gap)
                                    && (abs(rqcorr) > rqtol * abs(λ))
                                    && !usedBS)
                                    # check that RQCORR update doesn't move eigval
                                    # away from desired one
                                    if indeig <= negcnt
                                        # wanted eigval lies to the left
                                        sgndef = -1
                                    else
                                        sgndef = 1
                                    end
                                    # only use RQCORR if it improves the iterate
                                    if ((rqcorr * sgndef >= 0)
                                        && (λ + rqcorr <= right)
                                        && (λ + rqcorr >= left))
                                        usedRQ = true
                                        # store new midpoint of bisection interval
                                        # assumes error estimate is correct;
                                        # see discussion in DLARRV
                                        if sgndef == 1
                                            left = λ
                                        else
                                            right = λ
                                        end
                                        w_temp[w_idx] = (right + left) / 2
                                        λ += rqcorr
                                        w_err[w_idx] = (right - left) / 2
                                    else
                                        needBS = true
                                    end
                                    if right - left < rqtol * abs(λ)
                                        # the eigval is computed to bisection accuracy
                                        # proceed to final eigenvector update
                                        usedBS = true
                                    elseif iter < maxit
                                        # continue
                                    elseif iter == maxit
                                        needBS = true
                                    else
                                        throw(ErrorException("RQ convergence failure"))
                                    end
                                else
                                    # do we need another step of iterative improvement?
                                    stp2ii = false
                                    if (usedRQ && usedBS) && (bestres <= resid)
                                        λ = w_best
                                        stp2ii = true
                                    end
                                    if stp2ii
                                        # improve error angle
                                        negcnt, ztz, minγ, r, isuppz, convstuff =
                                            _st_get1ev(bsize, 1, bsize, λ, view(D,ibegin:iend),
                                                   view(L,ibegin:iend-1),
                                                   view(LD,ibegin:iend-1),view(LLD,ibegin:iend-1),
                                                   pivmin, gaptol, view(Z,ibegin:iend,w_idx),
                                                   !usedBS,
                                                   twist_inds[w_idx], workvecs)
                                        nrminv, resid, rqcorr = convstuff
                                        twist_inds[w_idx] = r
                                        supportZ[1:2,w_idx] .= isuppz

                                    end
                                    w_temp[w_idx] = λ
                                    vdone = true
                                end # RQ check or BS update branches
                            end # while !vdone

                            # compute FP-vector support wrt whole matrix
                            supportZ[1:2,w_idx] .+= old_iend
                            zfrom,zto = supportZ[:,w_idx]
                            isupmin += old_iend
                            isupmax += old_iend
                            # ensure vector is ok if support in RQI has changed
                            Z[isupmin:zfrom-1,w_idx] .= 0
                            Z[zto+1:isupmax,w_idx] .= 0
                            Z[zfrom:zto,w_idx] .*= nrminv
                        end # if !eskip 125
                        # update this eigenvalue
                        w[w_idx] = λ + σ
                        # recompute gaps, but only allow expansion
                        if !eskip
                            if k > 1
                                w_gap[w_idxmin] = max(w_gap[w_idxmin],
                                                      w[w_idx] - w_err[w_idx]
                                                      - w[w_idxmin] - w_err[w_idxmin])
                            end
                            if w_idx < w_end
                                w_gap[w_idx] = max(save_gap, w[w_idxpl] - w_err[w_idxpl]
                                                   - w[w_idx] - w_err[w_idx])
                            end
                        end
                        idone += 1
                    end # cluster/singleton branches
                    # end of code for current child
                    # proceed to any remaining child nodes
                    newfirst = j + 1
                end # j loop (child inside node) 140
            end # i loop (clusters on current level) 150
            ndepth += 1
        end # while idone < mblock
        ibegin = iend + 1
        w_begin = w_end + 1
    end # jblk loop 170

    return Z, supportZ
end

# Find a new relatively robust representation s.t. at least one eigenvalue
# is relatively isolated
# translation of DLARRF from LAPACK
# updates w_gap
function _st_rrr!(n,D::AbstractVector{T}, L, LD, cl_start, cl_end,
                  w, w_gap, w_err, spdiam, clgapl, clgapr, pivmin,
                  d₊, l₊
                  ) where T
    # d₊ = Vector{T}(undef,n)
    # l₊ = Vector{T}(undef,n-1)
    td₊ = similar(d₊)
    tl₊ = similar(l₊)
    σ = T(0)

    ktrymax = 1 # was 1, then 3
    sleft = 1
    sright = 2
    maxgrowth1 = T(8)
    maxgrowth2 = T(8)

    fact = T(2) # EXP was T(2)^ktrymax
    myeps = eps(T)
    shift = 0
    force_r = false
    nofail = false

    # compute avg gap length
    cl_width = abs(w[cl_end] - w[cl_start]) + w_err[cl_end] + w_err[cl_start]
    avg_gap = cl_width / (cl_end - cl_start)
    mingap = min(clgapl, clgapr)
    # initial values for shifts to both ends
    lσ = min(w[cl_start], w[cl_end]) - w_err[cl_start]
    rσ = max(w[cl_start], w[cl_end]) + w_err[cl_end]
    # use small fudge to make sure we really shift to outside
    lσ -= abs(lσ) * 4myeps
    rσ += abs(rσ) * 4myeps
    # upper bounds for how much to back off initial shifts
    ldmax = mingap / 4 + 2pivmin
    rdmax = mingap / 4 + 2pivmin
    lδ = max(avg_gap, w_gap[cl_start]) / fact
    rδ = max(avg_gap, w_gap[cl_end-1]) / fact

    s = safemin(T)
    small_growth = 1 / s
    fail = (n-1) * mingap / (spdiam * myeps)
    fail2 = (n-1) * mingap / (spdiam * sqrt(myeps))
    # initialize record of best representation found
    best_shift = lσ
    growth_bound = maxgrowth1 * spdiam
    ktry = 0
    while ktry <= ktrymax
        norefine1 = false
        norefine2 = false
        # ensure we don't back off too much from initial shifts
        lδ = min(ldmax, lδ)
        rδ = min(rdmax, rδ)
        # compute element growth when shifting to both ends of cluster
        # accept shift if there is no element growth at one end

        # left
        s = -lσ
        d₊[1] = D[1] + s
        if abs(d₊[1]) < pivmin
            d₊[1] = -pivmin
            # refined RRR test should not be used in this case
            norefine1 = true
        end
        max1 = abs(d₊[1])
        for i in 1:n-1
            l₊[i] = LD[i] / d₊[i]
            s = s * l₊[i] * L[i] - lσ
            d₊[i+1] = D[i+1] + s
            if abs(d₊[i+1]) < pivmin
                d₊[i+1] = -pivmin
                norefine1 = true
            end
            max1 = max(max1, abs(d₊[i+1]))
        end
        norefine1 = norefine1 || isnan(max1)
        if force_r || ((max1 <= growth_bound) && !norefine1)
            σ = lσ
            shift = sleft
            break
        end

        # right end
        s = -rσ
        td₊[1] = D[1] + s
        if abs(td₊[1]) < pivmin
            td₊[1] = -pivmin
            # refined RRR test should not be used in this case
            norefine2 = true
        end
        max2 = abs(td₊[1])
        for i in 1:n-1
            tl₊[i] = LD[i] / td₊[i]
            s = s * tl₊[i] * L[i] - rσ
            td₊[i+1] = D[i+1] + s
            if abs(td₊[i+1]) < pivmin
                td₊[i+1] = -pivmin
                norefine2 = true
            end
            max2 = max(max2, abs(td₊[i+1]))
        end
        norefine2 = norefine2 || isnan(max2)
        if force_r || ((max2 <= growth_bound) && !norefine2)
            σ = rσ
            shift = sright
            break
        end

        # Reach here if both shifts led to excessive element growth.
        if !(norefine1 && norefine2)
            # Record the better of the two (provided refinement is allowed)
            if !norefine1
                indx = 1
                if (max1 <= small_growth)
                    small_growth = max1
                    best_shift = lσ
                end
            end
            if !norefine2
                if norefine1 || (max2 <= max1)
                    indx = 2
                end
                if max2 <= small_growth
                    small_growth = max2
                    best_shift = rσ
                end
            end

            # If we reach here, both left and right shifts led to growth.
            # If element growth is moderate, we accept the representation
            # if it passes a refined test for RRR.
            # This supposes no NaN occurred above.
            # Only use the refined test for isolated clusters.
            do_rrr1 = ((cl_width < mingap / 128)
                       && (min(max1, max2) < fail2)
                       && !norefine1 && !norefine2)
            if do_rrr1
                if indx == 1
                    t = abs(d₊[n])
                    oldp = prod = znm2 = T(1)
                    for i in n-1:-1:1
                        if prod <= myeps
                            prod = (d₊[i+1] * tl₊[i+1]) / (d₊[i] * tl₊[i]) * oldp
                        else
                            prod *= abs(tl₊[i])
                        end
                        oldp = prod
                        znm2 += prod^2
                        t = max(t, abs(d₊[i] * prod))
                    end
                    rrr1 = t / (spdiam * sqrt(znm2))
                    if rrr1 <= maxgrowth2
                        σ = lσ
                        shift = sleft
                        break
                    end
                elseif indx == 2
                    t = abs(td₊[n])
                    znm2 = prod = oldp = T(1)
                    for i in n-1:-1:1
                        if prod <= myeps
                            prod = (td₊[i+1] * l₊[i+1]) / (td₊[i] * l₊[i]) * oldp
                        else
                            prod *= abs(l₊[i])
                        end
                        oldp = prod
                        znm2 += prod^2
                        t = max(t, abs(td₊[i] * prod))
                    end
                    rrr2 = t / (spdiam * sqrt(znm2))
                    if rrr2 <= maxgrowth2
                        σ = rσ
                        shift = sright
                        break
                    end
                end # indx branches
            else
                nothing
            end # if do_rrr1
        else
            nothing
        end # if refined test was allowed

        if ktry < ktrymax
            # if we fall through to here, both shifts failed the RRR test
            # so back off to the outside
            lσ = max(lσ - lδ, lσ - ldmax)
            rσ = min(rσ + rδ, rσ + rdmax)
            if ktry < 3
                lδ *= 2
                rδ *= 2
            else
                # *RAS* experiment: break out of conventional thinking
                lδ *= 8
                rδ *= 8
            end
            ktry += 1
        else
            # None of the representations investigated satisfied our criteria.
            # Take the best one we found.
            if (small_growth < fail) || nofail
                lσ = best_shift
                rσ = best_shift
                force_r = true
            else
                throw(ErrorException("failed to find adequate representation"))
            end
        end
    end
    if shift != sleft
        # store new L and D
        d₊ .= td₊
        l₊ .= tl₊
    end
    return σ, d₊, l₊
end

# Compute the scaled r-th column of the inverse of the submatrix in rows b1:bn
# of tridiagonal matrix L D Lᵀ - λ I.
# D&P show in LAWN 154 that this has good orthogonality property.
# translation of LAR1V from LAPACK
function _st_get1ev(n, b1, bn, λ, D::AbstractVector{T}, L, LD, LLD, pivmin, gaptol, Z,
                    wantnc, r, workvecs) where T
    myeps = eps(T)

    z0 = zero(T)
    if r == 0
        r1 = b1
        r2 = bn
    else
        r1 = r2 = r
    end
    L₊ = workvecs[1]
    U₋ = workvecs[2]
    S =  workvecs[3]
    P =  workvecs[4]
    if b1 == 1
        S[1] = z0
    else
        S[b1] = LLD[b1-1]
    end

    # Step 1. Compute stationary transform using the differential form up to index r2
    sawnan1 = false
    neg1 = 0
    s = S[b1] - λ
    @inbounds for i in b1:r1-1
        d₊ = D[i] + s
        L₊[i] = LD[i] / d₊
        if d₊ < 0
            neg1 += 1
        end
        S[i+1] = s * L₊[i] * L[i]
        s = S[i+1] - λ
    end
    sawnan1 = isnan(s)
    if !sawnan1
        @inbounds for i in r1:r2-1
            d₊ = D[i] + s
            L₊[i] = LD[i] / d₊
            S[i+1] = s * L₊[i] * L[i]
            s = S[i+1] - λ
        end
        sawnan1 = isnan(s)
    end
    if sawnan1
        # slower version of step 1 with checks
        neg1 = 0
        s = S[b1] - λ
        @inbounds for i in b1:r1-1
            d₊ = D[i] + s
            if abs(d₊) < pivmin
                d₊ = -pivmin
            end
            L₊[i] = LD[i] / d₊
            if d₊ < 0
                neg1 += 1
            end
            if L₊[i] == 0
                S[i+1] = LLD[i]
            else
                S[i+1] = s * L₊[i] * L[i]
            end
            s = S[i+1] - λ
        end
        @inbounds for i in r1:r2-1
            d₊ = D[i] + s
            if abs(d₊) < pivmin
                d₊ = -pivmin
            end
            L₊[i] = LD[i] / d₊
            if L₊[i] == 0
                S[i+1] = LLD[i]
            else
                S[i+1] = s * L₊[i] * L[i]
            end
            s = S[i+1] - λ
        end
    end

    # Step 2. Compute stationary transform using the differential form down to index r1
    sawnan1 = false
    P[bn] = D[bn] - λ
    neg2 = 0
    @inbounds for i in bn-1:-1:r1
        d₋ = LLD[i] + P[i+1]
        t = D[i] / d₋
        if d₋ < 0
            neg2 += 1
        end
        U₋[i] = L[i] * t
        P[i] = P[i+1] * t - λ
    end
    t = P[r1]
    sawnan2 = isnan(t)
    if sawnan2
        neg2 = 0
        @inbounds for i in bn-1:-1:r1
            d₋ = LLD[i] + P[i+1]
            if abs(d₋) < pivmin
                d₋ = -pivmin
            end
            t = D[i] / d₋
            if d₋ < 0
                neg2 += 1
            end
            U₋[i] = L[i] * t
            P[i] = P[i+1] * t - λ
            if t == 0
                P[i] = D[i] - λ
            end
        end
    end

    # Step 3. find index from r1:r2 of largest magnitude diagonal element of inverse
    minγ = S[r1] + P[r1]
    if minγ < 0
        neg1 += 1
    end
    if wantnc
        negcnt = neg1 + neg2
    else
        negcnt = -1
    end
    if abs(minγ) == 0
        minγ = myeps * S[r1]
    end
    r = r1
    @inbounds for i in r1:r2-1
        t = S[i+1] + P[i+1]
        if t == 0
            t = myeps * S[i+1]
        end
        if abs(t) <= abs(minγ)
            minγ = t
            r = i+1
        end
    end

    # Step 4. Compute FP vector: solve Nᵀ v = eᵣ
    # first work downwards from r
    isuppz = [b1,bn]
    Z[r] = T(1)
    ztz = T(1)
    if !sawnan1 && !sawnan2
        @inbounds for i in r-1:-1:b1
            Z[i] = -(L₊[i] * Z[i+1])
            if ((abs(Z[i]) + abs(Z[i+1])) * abs(LD[i])) < gaptol
                isuppz[1] = i+1
                Z[i] = z0
                break
            end
            ztz += Z[i] * Z[i]
        end
    else
        @inbounds for i in r-1:-1:b1
            if Z[i+1] == 0
                Z[i] = -(LD[i+1] / LD[i]) * Z[i+2]
            else
                Z[i] = -(L₊[i] * Z[i+1])
            end
            if ((abs(Z[i]) + abs(Z[i+1])) * abs(LD[i])) < gaptol
                isuppz[1] = i+1
                Z[i] = z0
                isuppz[1] = i+1
                break
            end
            ztz += Z[i] * Z[i]
        end
    end
    # then work upwards from r in blocks
    if  !sawnan1 && !sawnan2
        @inbounds for i in r:bn-1
            Z[i+1] = -(U₋[i] * Z[i])
            if ((abs(Z[i]) + abs(Z[i+1]))) * abs(LD[i]) < gaptol
                Z[i+1] = z0
                isuppz[2] = i
                break
            end
            ztz += Z[i+1] * Z[i+1]
        end
    else
        @inbounds for i in r:bn-1
            if Z[i] == 0
                Z[i+1] = -(LD[i-1] / LD[i]) * Z[i-1]
            else
                Z[i+1] = -(U₋[i] * Z[i])
            end
            if ((abs(Z[i]) + abs(Z[i+1]))) * abs(LD[i]) < gaptol
                Z[i+1] = z0
                isuppz[2] = i
                break
            end
            ztz += Z[i+1] * Z[i+1]
        end
    end
    t = 1 / ztz
    nrminv = sqrt(t)
    resid = abs(minγ) * nrminv
    rqcorr = minγ * t

    convstuff = (nrminv, resid, rqcorr)
    return negcnt, ztz, minγ, r, isuppz, convstuff
end

# Refine approximate eigvals w[ifirst-offset:ilast-offset]; also update w_err.
# translation of  DLARRJ from LAPACK
function _st_refine!(n, D::AbstractVector{T}, Esq, ifirst, ilast, rtol,
                        offset, w, w_err, pivmin, spdiam) where T
    maxit = floor(Int, log2(spdiam + pivmin) - log2(pivmin)) + 2
    i1 = ifirst
    i2 = ilast
    llist = Matrix{Int}(undef,2,n)
    endpts = Matrix{T}(undef,2,n)
    # number of unconverged intervals
    nint = 0
    # last unconverged interval found
    prev = 0
    for i in i1:i2
        ii = i - offset
        left = w[ii] - w_err[ii]
        mid = w[ii]
        right = w[ii] + w_err[ii]
        width = right - mid
        t = max(abs(left), abs(right))
        # prevent test of converged intervals
        if width < rtol * t
            llist[1,i] = -1
            # make sure i1 always points to the first unconverged interval
            if i==i1 && i < i2
                i1 = i + 1
            end
            if prev >= i1 && i <= i2
                llist[1,prev] = i + 1
            end
        else
            # unconverged interval
            prev = i
            # insure containment
            fac = T(1)
            cnt = i # loop setup
            while cnt > i-1
                cnt = 0
                s = left
                d₊ = D[1] - s
                if d₊ < 0
                    cnt += 1
                end
                for j in 2:n
                    d₊ = D[j] - s - Esq[j-1] / d₊
                    if d₊ < 0
                        cnt += 1
                    end
                end
                if cnt > i-1
                    left -= w_err[ii] * fac
                    fac *= 2
                end
            end
            cnt = i-1 # loop setup
            fac = T(1)
            while cnt < i
                cnt = 0
                s = right
                d₊ = D[1] - s
                if d₊ < 0
                    cnt += 1
                end
                for j in 2:n
                    d₊ = D[j] - s - Esq[j-1] / d₊
                    if d₊ < 0
                        cnt += 1
                    end
                end
                if cnt < i
                    right += w_err[ii] * fac
                    fac *= 2
                end
            end
            nint += 1
            llist[1,i] = i + 1
            llist[2,i] = cnt
        end
        endpts[1,i] = left
        endpts[2,i] = right
    end
    save_i1 = i1
    iter = 0
    while (nint > 0) && (iter < maxit)
        prev = i1 - 1
        i = i1
        old_nint = nint

        for p in 1:old_nint
            ii = i - offset
            next = llist[1,i]
            left = endpts[1,i]
            right = endpts[2,i]
            mid = (left + right) /2
            width = right - mid
            t = max(abs(left), abs(right))
            if (width < rtol * t) || (iter == maxit)
                nint -= 1
                llist[1,i] = 0 # mark as converged
                if i1 == i
                    i1 = next
                else
                    if prev >= i1
                        llist[1,prev] = next
                    end
                end
                i = next
                continue
            end
            prev = i
            # perform a bisection step
            cnt = 0
            s = mid
            d₊ = D[1] - s
            if d₊ < 0
                cnt += 1
            end
            for j in 2:n
                d₊ = D[j] - s - Esq[j-1] / d₊
                if d₊ < 0
                    cnt += 1
                end
            end
            if cnt <= i-1
                endpts[1,i] = mid
            else
                endpts[2,i] = mid
            end
            i = next
        end
        iter += 1
    end
    # all intervals have converged
    for i in save_i1:ilast
        ii = i - offset
        # all marked intervals have been refined
        if llist[1,i] == 0
            w[ii] = (endpts[1,i] + endpts[2,i]) / 2
            w_err[ii] = endpts[2,i] - w[ii]
        end
    end
end

# test whether sym. trid. warrants guaranteed high relative accuracy in eigvals
# translation of DLARRR from LAPACK
function _check_RAC(D::AbstractVector{T},E) where T <: AbstractFloat
    n = length(D)
    safmin = safemin(T)
    myeps = eps(T)
    smallnum = safmin / myeps
    rmin = sqrt(smallnum)
    relcond = T(0.999)

    # tests for guaranteed relative accuracy

    res = false

    # Test for scaled diagonal dominance:
    # Scale diag to unity and check whether sum of off-diags < 1.
    # See DLARRR for use of relcond.
    sdd = true
    offdiag = T(0)
    t = sqrt(abs(D[1]))
    if t < rmin
        sdd = false
    else
        for i in 2:n
            t2 = sqrt(abs(D[i]))
            if t2 < rmin
                sdd = false
                break
            end
            offdiag2 = abs(E[i-1]) / (t*t2)
            if offdiag + offdiag2 >= relcond
                sdd = false
                break
            end
            t = t2
            offdiag = offdiag2
        end
    end
    res |= sdd

    # other possible tests suggested by LAPACK authors:

    # Test if lower bidiagonal L from T = LDLᵀ (w/ zero shift) is well-conditioned.

    # Test if upper bidiagonal U from T = UDUᵀ (w/ zero shift) is well-conditioned.
    # To exploit this one needs to flip the matrix and then the eigenvectors.

    return res
end

struct StBlocks{T}
    m::Int # total nr of eigvals of all Lᵢ Dᵢ Lᵢᵀ found
    nsplit::Int # nr of blocks
    isplit::Vector{Int} # splitting points
    iblock::Vector{Int} # which block
    indexw::Vector{Int} # which entry in block
    gersl::Vector{T} # endpts of Geršgorin intervals
    gersu::Vector{T}
    pivmin::T # min pivot in Sturm sequence
end

#   w, w_err, w_gap, m, blocks, (vl,vu) = _st_vals!(D, E, Esq,
#                                                   rtol1, rtol2, spltol, rac,
#                                                   select)
# Given diagonals of a sym. tridiag. matrix:
# Zeroes tiny off-diagonal elts.
# For each unreduced block, finds base representation and eigvals.
# Modifies D, E, Esq.
# WARNING: uses nth entry of E for workspace
# Translation of DLARRE from LAPACK

function _st_vals!(D::AbstractVector{T}, E, Esq, rtol1, rtol2, spltol, rac,
                      select) where T <: AbstractFloat
    n = length(D)
    z0 = T(0)
    myeps = eps(T)
    safmin = safemin(T)

    # assorted parameters slavishly copied from DLARRE
    rtl = sqrt(myeps) # tolerance for _kahan_eig
    bsrtol = sqrt(myeps) # tolerance for _st_approxeig
    ceps = 100 * myeps
    maxgrowth = T(64) # bound on element growth for base RRR
    fudge = T(2)
    one_half = 1/T(2)
    one_qtr = 1/T(4)
    fac = one_half
    pert = T(8)
    maxtry = 6 # bound on attempts to find base RRR

    if n==1
        w1 = D[1]
        if   ((select.style == :all)
              || ((select.style == :val) && (select.vl < w1 <= select.vu))
              || ((select.style == :idx) && (1 == select.il == select.iu))
              )
            m = 1
            w = [w1]
            w_err = [z0]
            w_gap = [z0]
            # StBlocks(m,nsplit,isplit,iblock,indexw,gersl,gersu,pivmin)
            blocks = StBlocks(1,1,[1],[1],[1],[1],[w1],[w1],z0)
        else
            TODO() # null result
        end
        # store shift for initial RRR
        E[1] = z0
        return w, w_err, w_gap, m, blocks, (w1,w1)
    end

    m = 0 # to be incremented as we go

    # initialize w_err, w_gap; compute Geršgorin intervals and sp. diam.
    gl = D[1]
    gu = D[1]
    e_old = z0
    e_max = z0
    if length(E) == n-1
        push!(E, z0)
    elseif length(E) == n
        E[n] = z0
    else
        throw(DimensionMismatch("Diagonal D and off-diagonal E are incommensurate"))
    end
    w_err = fill(z0,n)
    w_gap = fill(z0,n)
    gersl = fill(z0,n)
    gersu = fill(z0,n)
    for i in 1:n
        e_abs = abs(E[i])
        if e_abs >= e_max
            e_max = e_abs
        end
        t1 = e_abs + e_old
        gersl[i] = D[i] - t1
        gl = min(gl, gersl[i])
        gersu[i] = D[i] + t1
        gu = max(gu, gersu[i])
        e_old = e_abs
    end
    pivmin = safmin * max(T(1), e_max^2)
    # Geršgorin bound on spectral diameter
    spdiam = gu - gl

    nsplit, isplit = _compute_splits!(D, E, Esq, spltol, spdiam, Val(rac))

    useDQD0 = select.style == :all && !force_bisect[]
    useDQD = useDQD0 # may be revised later

    if useDQD
        vl = gl
        vu = gu
        w = fill(z0,n)
        iblock = fill(0,n)
        indexw = fill(0,n)
    else
        # find crude approximations to eigvals in desired range
        mm, w, w_err, vl, vu, iblock, indexw, converged =
            _st_approxeig(D, E, Esq, pivmin, nsplit, isplit,
                             gersl, gersu, bsrtol,
                             select)
        if !converged
            # FIXME: not helpful
            throw(ErrorException("convergence failure in approxeig"))
        end
        # clear dead entries
        # CHECKME: truncate?
        w[mm+1:n] .= z0
        iblock[mm+1:n] .= 0
        indexw[mm+1:n] .= 0
    end

    ibegin = 1
    w_begin = 1
    for jblk in 1:nsplit
        iend = isplit[jblk]
        bsize = iend - ibegin + 1
        if bsize == 1
            if   ((select.style == :all)
                  || ((select.style == :val) && (select.vl < D[ibegin] <= select.vu))
                  || ((select.style == :idx) && (iblock[w_begin] == jblk))
                  )
                m += 1
                w[m] = D[ibegin]
                w_err[m] = z0
                w_gap[m] = z0 # arbitrary
                iblock[m] = jblk
                indexw[m] = 1
                w_begin += 1
            end
            E[iend]  = z0 # shift for initial RRR
            ibegin = iend + 1
            continue
        end

        # code for a nontrivial block
        E[iend] = z0
        # find local outer bounds
        gl = min(D[ibegin],minimum(view(gersl,ibegin:iend)))
        gu = max(D[ibegin],maximum(view(gersu,ibegin:iend)))
        spdiam = gu - gl

        if !useDQD0
            # count eigvals in current block
            mb = 0
            for i in w_begin:mm
                if iblock[i] == jblk
                    mb += 1
                else
                    break
                end
            end
            if mb == 0
                # no eigval in current block is in desired range
                E[iend] = z0
                ibegin = iend + 1
                continue # CHECKME: should go to next jblk
            else
                useDQD = (mb > fac * bsize) && !force_bisect[]
                w_end = w_begin + mb - 1
                # calculate gaps for current block
                # In later stages when representations for individual
                # eigvals are different, use σ = E[iend]
                σ = z0
                for i in w_begin:w_end-1
                    w_gap[i] = max(z0, w[i+1] - w_err[i+1]
                                   - (w[i] + w_err[i]))
                end
                w_gap[w_end] = max(z0, vu - σ - (w[w_end] + w_err[w_end]))
                indl = indexw[w_begin]
                indu = indexw[w_end]
            end
        end
        if useDQD0 || useDQD
            # find approximate extremal eigvals of block

            λ1, λ1_err, converged = _kahan_eig(1,view(D,ibegin:iend),
                                               view(Esq,ibegin:iend-1),
                                               gl, gu, pivmin, rtl)
            if !converged
                # FIXME: this is not helpful
                throw(ErrorException("convergence failure"))
            end
            isleft = max(gl, λ1 -  λ1_err - ceps * abs(λ1 - λ1_err))

            λ1, λ1_err, converged = _kahan_eig(bsize, view(D,ibegin:iend),
                                               view(Esq,ibegin:iend-1),
                                               gl, gu, pivmin, rtl)
            if !converged
                # FIXME: this is not helpful
                throw(ErrorException("convergence failure"))
            end
            isright = min(gu, λ1 +  λ1_err + ceps * abs(λ1 + λ1_err))
            spdiam = isright - isleft
        else
            # prepare for bisection
            isleft = max(gl, w[w_begin] - w_err[w_begin]
                         - ceps * abs(w[w_begin] - w_err[w_begin]))
            isright = min(gu, w[w_end] + w_err[w_end]
                          + ceps * abs(w[w_end] + w_err[w_end]))
        end

        # Decide whether base representation for current block
        # should be on left or right end.
        # Strategy is to shift to the "more populated" end.
        # CHECKME: can the ones at the other end be much smaller?
        # Plan for dqds if all eigvals are desired or the number is
        # large compared to the blocksize.
        if useDQD0
            useDQD = true
            indl = 1
            indu = bsize
            mb = bsize
            w_end = w_begin + mb - 1
            s1 = isleft + spdiam * one_qtr
            s2 = isright - spdiam * one_qtr
        else
            if useDQD
                s1 = isleft + spdiam * one_qtr
                s2 = isright - spdiam * one_qtr
            else
                t = min(isright, vu) - max(isleft, vl)
                s1 = max(isleft, vl) + t * one_qtr
                s2 = min(isright, vu) - t * one_qtr
            end
        end

        if mb > 1
            # compute negcount at quartile points
            cnt, cnt1, cnt2 = _st_count_eigs(bsize, s1, s2,
                                                view(D,ibegin:iend),
                                                view(E,ibegin:iend), pivmin)
        end

        # find initial shift σ for LDLᵀ decomp of A - σ I
        if mb == 1
            σ = gl
            sgndef = T(1)
        elseif cnt1 - indl >= indu - cnt2
            # Use Geršgorin bound to get SPD matrix for dqds
            # or use approximate first desired eigval
            if useDQD0
                σ = max(isleft, gl)
            elseif useDQD
                σ = isleft
            else
                σ = max(isleft, vl)
            end
            sgndef = T(1)
        else
            # Use Geršgorin bound as shift to get neg def matrix for dqds
            # or approximate last desired eigval.
            if useDQD0
                σ = min(isright, gu)
            elseif useDQD
                σ = isright
            else
                σ = min(isright, vu)
            end
            sgndef = -T(1)
        end

        # An initial σ has been chosen to compute A - σI = L D Lᵀ.
        # Now compute shift increment τ to avoid excessive element growth.
        if useDQD
            τ = max(spdiam * myeps * n + 2 * pivmin, 2*myeps * abs(σ))
        else
            if mb > 1
                clwidth = w[w_end] + w_err[w_end] - w[w_begin] - w_err[w_begin]
                avgap = abs(clwidth / (w_end - w_begin))
                if sgndef == 1
                    τ = max(w_gap[w_begin], avgap) * one_half
                    τ = max(τ, w_err[w_begin])
                else
                    τ = max(w_gap[w_end-1], avgap) * one_half
                    τ = max(τ, w_err[w_end])
                end
            else
                τ = w_err[w_begin]
            end
        end

        Dfac = zeros(T,bsize)
        Lfac = zeros(T,bsize)
        inv_pivots = zeros(T,bsize)
        for itry in 1:maxtry
            # compute LDLᵀ factorization of tridiagonal A - σ I
            dpivot = D[ibegin] - σ
            Dfac[1] = dpivot
            dmax = abs(Dfac[1])
            j = ibegin
            for i in 1:bsize-1
                inv_pivots[i] = 1 / Dfac[i]
                t = E[j] * inv_pivots[i]
                Lfac[i] = t
                dpivot = (D[j+1] - σ) - t * E[j]
                Dfac[i+1] = dpivot
                dmax = max(dmax, abs(dpivot))
                j += 1
            end
            # check for element growth
            norep = (dmax > maxgrowth * spdiam)
            if useDQD && !norep
                # ensure definiteness
                for i in 1:bsize
                    t = sgndef * Dfac[i]
                    if t < 0
                        norep = true
                    end
                end
            end
            if norep
                if itry == maxtry-1
                    if sgndef == 1
                        σ = (gl - fudge * spdiam * myeps * n
                             - fudge * 2 * pivmin)
                    else
                        σ = (gu + fudge * spdiam * myeps * n
                             + fudge * 2 * pivmin)
                    end
                else
                    σ -= sgndef * τ
                    τ *= 2
                end
            else
                # initial RRR is found
                break
            end
            if itry == maxtry
                # FIXME: not helpful
                throw(ErrorException("failed to find base representation in maxtry iterations"))
            end
        end # shifted LDLᵀ trial loop

        # We have found an initial base representation with acceptable
        # element growth. Store the shift.
        E[iend] = σ
        # store the decomposition
        D[ibegin:iend] .= Dfac
        E[ibegin:iend-1] .= Lfac[1:bsize-1]

        if mb > 1
            # Perturb entries in base representation to
            # overcome difficulties with glued matrices.
            # See Parlett & Vömel, LAWN 163
            # FIXME: make this repeatable.
            Dfac .= (2*rand(bsize) .- 1) # treat as workspace
            Lfac .= (2*rand(bsize) .- 1)
            D[ibegin:iend-1] .*= (1 .+ myeps * pert * Dfac[1:bsize-1])
            E[ibegin:iend-1] .*= (1 .+ myeps * pert * Lfac[1:bsize-1])
            D[iend] *= (1 + myeps * 4 * Dfac[bsize])
        end

        # Don't update Geršgorin intervals to avoid extra work in eigenvector code.
        # Instead update `w` and use it to locate new Geršgorin intervals.

        if !useDQD # bisecting
            # Shift the approximate eigvals according to their representation.
            # This is necessary for compatibility with dqds logic since
            # W holds unshifted approximants in the eigenvector code.
            for j in w_begin:w_end
                w[j] -= σ
                w_err[j] += abs(w[j]) * myeps
            end
            # find eigvals from indl to indu
            tmp = D[ibegin:iend-1] .* E[ibegin:iend-1] .^ 2
            _bisect_limited!(bsize, view(D, ibegin:iend), tmp,
                             indl, indu, rtol1, rtol2, indl-1,
                             view(w, w_begin:w_end), view(w_gap, w_begin:w_end),
                             view(w_err, w_begin:w_end), pivmin, spdiam, bsize)
            # bisection code computes all gaps except the last
            w_gap[w_end] = max(0, (vu - σ) - (w[w_end] + w_err[w_end]))
            for i in indl:indu
                m += 1
                iblock[m] = jblk
                indexw[m] = i
            end
        else
            # compute eigenvals of LDLᵀ by dqds
            # Note that dqds finds eigvals to high relative accuracy (HRA).
            # HRA may be lost when the shift of the RRR is subtracted to obtain
            # eigvals of A, but A may not define them to HRA anyway.

            # This is the order of tolerance used in DLASQ2,
            rtol = DQDS._dqds_tol(bsize, T)

            # initialize the qd vector
            Zqd = zeros(T,4*bsize)
            ii = 0
            jj = ibegin
            for i in 1:bsize
                ii += 1
                Zqd[ii] = abs(D[jj])
                ii += 1
                if i < bsize
                    Zqd[ii] = E[jj]^2 * Zqd[ii-1]
                end
                jj += 1
            end

            # call the tangled mess...
            DQDS._dqds_eigvals!(bsize, Zqd)

            # CHECKME: if we use return values, handle them here

            for i in 1:bsize
                if Zqd[i] < 0
                    throw(ErrorException("nonpositive eigenvalue from dqds"))
                end
            end
            # extract w[m] from Z
            #  set iblock[m], indexw[m]
            if sgndef > 0
                for i in indl:indu
                    m += 1
                    w[m] = Zqd[bsize-i+1]
                    iblock[m] = jblk
                    indexw[m] = i
                end
            else
                for i in indl:indu
                    m += 1
                    w[m] = -Zqd[i]
                    iblock[m] = jblk
                    indexw[m] = i
                end
            end

            #  set w_err[m-mb+1:m], w_gap[m-mb+1:m]
            for i in m-mb+1:m
                w_err[i] = rtol * abs(w[i])
            end
            for i in m-mb+1:m-1
                w_gap[i] = max(z0, w[i+1] - w_err[i+1] - (w[i] + w_err[i]))
            end
            w_gap[m] = max(z0,(vu - σ) - (w[m] + w_err[m]))
        end
        ibegin = iend + 1
        w_begin = w_end + 1
    end # block loop

    blocks = StBlocks(m,nsplit,isplit,iblock,indexw,gersl,gersu,pivmin)
    return w, w_err, w_gap, m, blocks, (vl, vu)
end

# compute splitting points with specified threshold
# returns nsplit, isplit
# modifies E, Esq
# translation of DLARRA from LAPACK
# use criterion based on absolute off-diagonal
function _compute_splits!(D::AbstractVector{T}, E, Esq, spltol, tnrm, ::Val{false}) where T
    n = length(D)
    z0 = T(0)
    nsplit = 1
    isplit = Int[]

    t1 = abs(spltol) * tnrm
    for i in 1:n-1
        eabs = abs(E[i])
        if eabs <= t1
            E[i] = z0
            Esq[i] = z0
            push!(isplit, i)
            nsplit += 1
        end
    end
    push!(isplit, n)
    return nsplit, isplit
end

# use criterion guaranteeing relative accuracy
function _compute_splits!(D::AbstractVector{T}, E, Esq, spltol, tnrm, ::Val{true}) where T
    n = length(D)
    z0 = T(0)
    nsplit = 1
    isplit = Int[]

    for i in 1:n-1
        eabs = abs(E[i])
        if eabs <= spltol * sqrt(abs(D[i])) * sqrt(abs(D[i+1]))
            E[i] = z0
            Esq[i] = z0
            push!(isplit, i)
            nsplit += 1
        end
    end
    push!(isplit, n)
    return nsplit, isplit
end

# Given RRR L D Lᵀ, refine w[ifirst-offset:ilast-offset].
# translation of DLARRB from LAPACK
function _bisect_limited!(n, D::AbstractVector{T}, LLD,
                          ifirst, ilast, rtol1, rtol2, offset,
                          w, w_gap, w_err, pivmin, spdiam, twist;
                          verbose=false
                          ) where T
    one_half = 1/T(2)
    maxit = floor(Int, log2(spdiam + pivmin) - log2(pivmin)) + 2
    minwidth = 2pivmin
    r = twist
    if r < 1 || r > n
        r = n
    end
    llist = Matrix{Int}(undef,2,n)
    endpts = Matrix{T}(undef,2,n)
    # initialize unconverged intervals
    i1 = ifirst
    nint = 0 # number of unconverged intervals
    prev = 0 # last unconverged interval
    rgap = w_gap[i1 - offset]
    for i in i1:ilast
        negcnt = 0
        ii = i - offset
        left = w[ii] - w_err[ii]
        right = w[ii] + w_err[ii]
        lgap = rgap
        rgap = w_gap[ii]
        gap = min(lgap, rgap)

        verbose && println("BS initial interval $i: $left $right")
        # make sure [left,right] contains the desired eigval
        # compute Sturm count from dstqds factor L₊D₊L₊ᵀ = LDLᵀ - left
        back = w_err[ii]
        done = false
        while !done
            negcnt = _sturm_count(n, D, LLD, left, pivmin, r)
            done = (negcnt <= i-1)
            if !done
                left -= back
                back = 2back
            end
        end
        # compute Sturm count from dstqds factor L₊D₊L₊ᵀ = LDLᵀ - right
        back = w_err[ii]
        done = false
        while !done
            negcnt = _sturm_count(n, D, LLD, right, pivmin, r)
            done = (negcnt >= i)
            if !done
                right += back
                back = 2back
            end
        end
        verbose && println("BS adjusted interval $i: $left $right")
        width = abs(left - right) * one_half
        t = max(abs(left), abs(right))
        converged = max(rtol1 * gap, rtol2 * t)
        if (width <= converged) || (width <= minwidth)
            # This interval has converged, so remove it from list.
            # Gaps might change through refinement, but can only get bigger.
            llist[1,i] = -1
            # Make sure i1 points to first unconverged interval
            if i == i1 && i < ilast
                i1 = i + 1
            end
            if prev >= i1 && i <= ilast
                llist[1,prev] = i + 1
            end
        else
            # unconverged interval found
            prev = i
            nint += 1
            llist[1,i] = i + 1
            llist[2,i] = negcnt
        end
        endpts[1,i] = left
        endpts[2,i] = right
    end # i loop

    # there are still unconverged intervals
    iter = 0
    while (nint > 0) && (iter < maxit)
        prev = i1 - 1
        i = i1
        oldnint = nint
        for ip in 1:oldnint
            ii = i - offset
            rgap = w_gap[ii]
            if ii > 1
                lgap = w_gap[ii-1]
            else
                lgap = rgap
            end
            gap = min(lgap, rgap)
            next = llist[1,i]
            left = endpts[1,i]
            right = endpts[2,i]
            verbose && println("BS refining interval $i: $left $right")
            mid = (left + right) * one_half
            # semiwidth
            width = right - mid
            t = max(abs(left), abs(right))
            converged = max(rtol1 * gap, rtol2 * t)
            if width <= converged || width <= minwidth || iter == maxit
                # reduce nr of unconv. intervals
                nint -= 1
                # mark as converged
                llist[1,i] = 0
                if i1 == i
                    i1 = next
                elseif prev >= i1
                        llist[1,prev] = next
                end
                i = next
                continue
            end
            prev = i
            # perform a bisection step
            negcnt = _sturm_count(n, D, LLD, mid, pivmin, r)
            verbose && println("sturm at $mid: $negcnt")
            if negcnt <= i-1
                endpts[1,i] = mid
            else
                endpts[2,i] = mid
            end
            i = next
        end
        iter += 1
    end

    # all intervals have converged
    for i in ifirst:ilast
        ii = i - offset
        if llist[1,i] == 0
            w[ii] = (endpts[1,i] + endpts[2,i]) * one_half
            w_err[ii] = endpts[2,i] - w[ii]
        end
    end
    for ii in ifirst+1-offset:ilast-offset
        w_gap[ii-1] = max(0, w[ii] - w_err[ii] - w[ii-1] - w_err[ii-1])
    end
end

# Compute the nr of negative pivots arising in the LDLᵀ factorization
# of A - σ I, from the previously computed factors.
# translation of LANEG from LAPACK
# see dlaneg.f for restrictions
function _sturm_count(n, D::AbstractVector{T}, LLD, σ, pivmin, r) where T
    blklen = 128

    negcnt = 0

    # I) upper part: L D Lᵀ - σ I = L₊ D₊ L₊ᵀ
    t = -σ
    @inbounds for bj in 1:blklen:r-1
        neg1 = 0
        bsav = t
        for j in bj:min(bj+blklen-1,r-1)
            d₊ = D[j] + t
            if d₊ < 0
                neg1 += 1
            end
            t1 = t / d₊
            t = t1 * LLD[j] - σ
        end
        sawNaN = isnan(t)
        # run a slower version if NaN is detected (zero pivot after infinite)
        if sawNaN
            neg1 = 0
            t = bsav
            for j in bj:min(bj+blklen-1,r-1)
                d₊ = D[j] + t
                if d₊ < 0
                    neg1 += 1
                end
                t1 = t / d₊
                if isnan(t1)
                    t1 = one(T)
                end
                t = t1 * LLD[j] - σ
            end
        end
        negcnt += neg1
    end
    # II) lower part
    p = D[n] - σ
    @inbounds for bj in n-1:-blklen:r
        neg2 = 0
        bsav = p
        for j in bj:-1:max(bj-blklen+1,r)
            d₋ = LLD[j] + p
            if d₋ < 0
                neg2 += 1
            end
            t1 = p / d₋
            p = t1 * D[j] - σ
        end
        sawNaN = isnan(p)
        if sawNaN
            neg2 = 0
            p = bsav
            for j in bj:-1:max(bj-blklen+1,r)
                d₋ = LLD[j] + p
                if d₋ < 0
                    neg2 += 1
                end
                t1 = p / d₋
                if isnan(t)
                    t1 = T(1)
                end
                p = t1 * D[j] - σ
            end
        end
        negcnt += neg2
    end
    # III) Twist index
    γ = (t + σ) + p
    if γ < 0
        negcnt += 1
    end
    return negcnt
end

# FIXME: apparently have translations of both dlarrc and zlarrc - pick one!

# translation of LARRC from LAPACK
# find nr of eigvals in interval (vl,vu]
# i.e. Sturm sequence count on sym. tridiag. (D,E)
function _st_count_eigs(n, vl, vu, D, E, pivmin)

    eigcnt = lcnt = rcnt = 0
    lpivot = D[1] - vl
    rpivot = D[1] - vu
    if lpivot <= 0
        lcnt += 1
    end
    if rpivot <= 0
        rcnt += 1
    end
    for i in 1:n-1
        t = E[i]^2
        lpivot = (D[i+1] - vl) - t / lpivot
        rpivot = (D[i+1] - vu) - t / rpivot
        if lpivot <= 0
            lcnt += 1
        end
        if rpivot <= 0
            rcnt += 1
        end
    end
    eigcnt = rcnt - lcnt
    return eigcnt, lcnt, rcnt
end

# translation of LARRC from LAPACK
# find nr of eigvals in interval (vl,vu]
# i.e. Sturm sequence count on L D Lᵀ (L is unit_tri(E))
function _hetrf_count_eigs(n, vl, vu, D, E, pivmin)

    eigcnt = lcnt = rcnt = 0
    sl = -vl
    su = -vu
    for i in 1:n-1
        lpivot = D[i] + sl
        rpivot = D[i] + su
        if lpivot <= 0
            lcnt += 1
        end
        if rpivot <= 0
            rcnt += 1
        end
        t = E[i] * D[i] * E[i]
        t2 = t / lpivot
        if t2 == 0
            sl = t - vl
        else
            sl = sl * t2 - vl
        end
        t2 = t / rpivot
        if t2 == 0
            su = t - vu
        else
            su = su * t2 - vu
        end
    end
    lpivot = D[n] + sl
    rpivot = D[n] + su
    if lpivot <= 0
        lcnt += 1
    end
    if rpivot <= 0
        rcnt += 1
    end
    eigcnt = rcnt - lcnt
    return eigcnt, lcnt, rcnt
end

# translation of DLARRD from LAPACK
# This is also very close to DSTEBZ
function _st_approxeig(D::AbstractVector{T}, E, Esq, pivmin, nsplit, isplit,
                       gersl, gersu, reltol,
                       select;
                       fudge=T(2), abstol=T(0)) where T <: AbstractFloat
    stebz_tol = false

    n = length(D)
    z0 = T(0)
    one_half = 1/T(2)
    myeps = eps(T)
    uflow = floatmin(T)
    if n==1
        w1 = D[1]
        if   ((select.style == :all)
              || ((select.style == :val) && (select.vl < w1 <= select.vu))
              || ((select.style == :idx) && (1 == select.il == select.iu))
              )
            m = 1
            w = [w1]
            w_err = [z0]
            iblock = [1]
            indexw = [1]
            wl = w1
            wu = w1
            converged = true
        else
            TODO() # null result
        end
        return m, w, w_err, wl, wu, iblock, indexw, converged
    end
    # nb is minimum vector length for vector bisection.
    # LAPACK calls a block config utility.
    # FIXME: we punt.
    nb = 32 # 0 to force scalar branch

    # find global spectral radius
    gl = min(D[1], minimum(gersl))
    gu = max(D[1], maximum(gersu))

    # Compute global Geršgorin bounds and spectral diameter
    tnorm = max(abs(gl), abs(gu))
    gl = gl - fudge * tnorm * myeps * n - fudge * 2 * pivmin
    gu = gu + fudge * tnorm * myeps * n + fudge * 2 * pivmin
    rtoli = reltol
    if stebz_tol
        atoli = abstol <= 0 ? myeps * tnorm : abstol
    else
        atoli = fudge * 2 * uflow + fudge * 2 * pivmin
    end

    if select.style == :idx
        itmax = round(Int,(log2(tnorm + pivmin) - log2(pivmin)) + 2)
        # compute an interval containing eigvals il:iu.
        # Initial interval (global Geršgorin bounds) is refined.
        ab = [gl gu; gl gu]
        nab = [-1 n+1; -1 n+1]
        c = [gl, gu]
        nval = [select.il-1, select.iu]
        nmax = 2; minp=2
        # dlaebz(3,itmax,n,nmax,minp,nb,atol,rtol,pivmin,d,e,e2,
        #  nval, ab, c, mout, nab, work, iwork, info)
        mmax = 2
        minp = 2
        n_intervals, n_left = _find_eigval_jumps!(itmax, n, mmax, minp, nb,
                                                  atoli, rtoli, pivmin,
                                                  D, E, Esq,
                                                  nval, ab, nab, c)
        if n_left > 0
            # DSTEBZ doesn't check for this, but DLARRD does
            throw(ErrorException("convergence failure"))
        end

        if nval[2] == select.iu
            wl = ab[1,1]
            wlu = ab[1,2]
            nwl = nab[1,1]
            wu = ab[2,2]
            wul = ab[2,1]
            nwu = nab[2,2]
        else
            wl = ab[2,1]
            wlu = ab[2,2]
            nwl = nab[2,1]
            wu = ab[1,2]
            nwu = nab[1,2]
        end
        # [wl,wlu] should have negcount nwl, and [wul,wu] negcount nwu.
        if nwl < 0 || nwl >= n || nwu < 1 || nwu > n
            throw(ErrorException("Gersgorin interval was too small: suspect arithmetic."))
        end
    elseif select.style == :val
        wl = select.vl
        wu = select.vu
    else
        wl = gl
        wu = gu
    end
    m = 0
    iend = 0
    nwl = 0 # accumulated count of eigvals <= wl
    nwu = 0 # accumulated count of eigvals <= wu

    w = zeros(T,n)
    w_err = zeros(T,n)
    iblock = fill(0,n)
    indexw = fill(0,n)
    converged = true

    for jblk in 1:nsplit
        ioff = iend
        ibegin = ioff + 1
        iend = isplit[jblk]
        bsize = iend - ioff
        if bsize == 1
            if wl >= D[ibegin] - pivmin
                nwl += 1
            end
            if wu >= D[ibegin] - pivmin
                nwu += 1
            end
            if (select.style == :all) || ((wl < D[ibegin] - pivmin)
                                          && (wu >= D[ibegin] - pivmin))
                m += 1
                w[m] = D[ibegin]
                w_err[m] = z0
                iblock[m] = jblk
                indexw[m] = 1
            end
        else
            gu = D[ibegin]
            gl = D[ibegin]
            t1 = z0
            gl = min(gl, minimum(view(gersl,ibegin:iend)))
            gu = max(gl, maximum(view(gersu,ibegin:iend)))
            gl = gl - fudge * tnorm * myeps * bsize - fudge * pivmin
            gu = gu + fudge * tnorm * myeps * bsize + fudge * pivmin

            if select.style != :all
                if gu < wl
                    nwl += bsize
                    nwu += bsize
                    continue
                end
                gl = max(gl, wl)
                gu = min(gu, wu)
                if gl >= gu
                    continue
                end
            end

            # find negcount for initial interval boundaries gl, gu
            # tails of work, w,  iblock used as workspace
            # iwork -> nab, iinfo -> istat
            ab = zeros(T,bsize,2)
            ab[1,1] = gl
            ab[1,2] = gu
            if stebz_tol
                atoli = abstol <= 0 ? myeps * max(abs(gl), abs(gu)) : abstol
            end
            # (bound on interval count) mmax <- bsize
            m_inc, nab = _find_nab(bsize, 1, atoli, rtoli, pivmin,
                          view(D,ibegin:iend), view(E,ibegin:iend),
                          view(Esq,ibegin:iend-1),
                                       ab)
            nwl += nab[1,1]
            nwu += nab[1,2]
            iwoff = m - nab[1,1]


            itmax = floor(Int,log2(gu - gl + pivmin) - log2(pivmin)) + 2

            n_intervals, n_left, c = _squeeze_eigvals!(itmax, bsize, 1, nb, atoli, rtoli, pivmin,
                                  view(D,ibegin:iend), view(E,ibegin:iend),
                                  view(Esq,ibegin:iend-1),
                                                       ab, nab)
            if n_left > 0
                # CHECKME: maybe just flag the poor ones as in LAPACK
                throw(ErrorException("convergence failure"))
            end
            # copy eigvals into w and set iblock
            # use -jblk for block nr for unconverged eigvals
            # loop over nr of output intervals from finder
            for j in 1:n_intervals
                # approx eigval is midpt of interval
                t1 = (ab[j,1] + ab[j,2]) /2
                t2 = abs(ab[j,1] - ab[j,2]) * one_half
                ib = (j > n_intervals - n_left) ? -jblk : jblk
                for je in nab[j,1] + 1 + iwoff: nab[j,2] + iwoff
                    w[je] = t1
                    w_err[je] = t2
                    indexw[je] = je-iwoff
                    iblock[je] = ib
                end
            end
            m += m_inc
        end # bsize > 1 branch
    end # jblk loop

    if select.style == :idx
        idiscl = select.il - 1 - nwl
        idiscu = nwu-select.iu
        if idiscl > 0
            # remove some of the smallest eigvals from the left so that at the end
            # idiscl = 0. Move all to the left.
            bsize = 0
            for je in 1:m
                if w[je] <= wlu && idiscl > 0
                    idiscl -= 1
                else
                    bsize += 1
                    w[bsize] = w[je]
                    w_err[bsize] = w_err[je]
                    indexw[bsize] = indexw[je]
                    iblock[bsize] = iblock[je]
                end
            end
            m = bsize
        end
        if idiscu > 0
            # remove some of the largest eigvals from the right so that at the end
            # idiscu = 0
            bsize += 1
            for je in m:-1:1
                if w[je] >= wul && idiscu > 0
                    idiscu -= 1
                else
                    bsize -= 1
                    w[bsize] = w[je]
                    w_err[bsize] = w_err[je]
                    indexw[bsize] = indexw[je]
                    iblock[bsize] = iblock[je]
                end
            end
            m = m-bsize+1
        end
        if idiscl > 0 || idiscu > 0
            # deal with effects of bad arithmetic
            # if N(w) is monotone decreasing, this should never happen
            if idiscl > 0
                wkill = wu
                for jdisc in 1:idiscl
                    iw = 0
                    for je in 1:m
                        if iblock[je] != 0 && w[je] < wkill || iw == 0
                            iw = je
                            wkill = w[je]
                        end
                    end
                    iblock[iw] = 0
                end
            end
            if idiscu > 0
                wkill = wl
                for jdisc in 1:idiscu
                    iw = 0
                    for je in 1:m
                        if iblock[je] != 0 && w[je] >= wkill || iw == 0
                            iw = je
                            wkill = w[je]
                        end
                    end
                    iblock[iw] = 0
                end
            end
            # erase all eigvals w/ iblock set to 0
            bsize = 0
            for je in 1:m
                if iblock[je] != 0
                    bsize += 1
                    w[bsize] = w[je]
                    w_err[bsize] = w_err[bsize]
                    indexw[bsize] = indexw[bsize]
                    iblock[bsize] = iblock[je]
                end
            end
            m = bsize
        end
        if idiscl < 0 || idiscu < 0
            converged = false
            @info "eigvals lost because of bad arithmetic"
        end
    end

    if    ((select.style == :all) && (m != n)
           || (select.style == :idx) && (m != select.iu - select.il + 1))
        converged = false
        @info "too few ($m) eigvals found"
    end

    return m, w, w_err, wl, wu, iblock, indexw, converged
end

# Given intervals defined in `ab`, count corresponding eigvals.
# translation of dlaebz, ijob=1
function _find_nab(mmax, minp, abstol, reltol, pivmin,
                          D, E, Esq,
                   ab)
    n = length(D)
    ntot = 0
    nab = zeros(Int,n,2)
    for ji in 1:minp
        for jp in 1:2
            t1 = D[1] - ab[ji,jp]
            if abs(t1) < pivmin
                t1 = -pivmin
            end
            nab[ji, jp] = 0
            if t1 <= 0
                nab[ji, jp] = 1
            end
            for j=2:n
                t1 = D[j] - Esq[j-1] / t1 - ab[ji,jp]
                if abs(t1) < pivmin
                    t1 = -pivmin
                end
                if t1 < 0
                    nab[ji,jp] += 1
                end
            end
        end
        ntot += nab[ji,2] - nab[ji,1]
    end
    return ntot, nab
end

# Given initial intervals `ab` and counts `nab`, revise so results are sufficiently
# small intervals containing the same eigvals.
# translation of dlaebz, ijob=2
# modifies ab, nab
function _squeeze_eigvals!(nitmax, mmax, minp, nbmin, abstol, reltol,
                        pivmin, D::AbstractVector{T}, E, Esq,
                        ab, nab) where T
    one_half = 1/T(2)
    n = length(D)
    # as we go, kf:kl is range of intervals needing refinement
    kf = 1
    kl = minp
    c = zeros(T,minp)
    for ji in 1:minp
        c[ji] = (ab[ji,1] + ab[ji,2]) /2
    end
    if nbmin > 0
        work = zeros(T,mmax)
        nbelow = fill(0,mmax)
    end
    for jit in 1:nitmax
        # loop over intervals
        if kl-kf+1 >= nbmin && nbmin > 0
            nbelow = fill(0,kl)
            # parallel version of loop
            for ji in kf:kl
                # compute N(c), nr eigvals < c
                work[ji] = D[1] - c[ji]
                nbelow[ji] = 0
                if work[ji] <= pivmin
                    nbelow[ji] = 1
                    work[ji] = min(work[ji], -pivmin)
                end
                for j in 2:n
                    work[ji] = D[j] - Esq[j-1] / work[ji] - c[ji]
                    if work[ji] < pivmin
                        nbelow[ji] += 1
                        work[ji] = min(work[ji], -pivmin)
                    end
                end
            end
            # choose all intervals containing eigvals
            klnew = kl
            for ji in kf:kl
                # insure N(w) monotone
                nbelow[ji] = min(nab[ji,2], max(nab[ji,1], nbelow[ji]))
                # update queue
                if nbelow[ji] == nab[ji,2]
                    # no ew in upper interval, uses lower
                    ab[ji,2] = c[ji]
                elseif nbelow[ji] == nab[ji,1]
                    # no ew in lower interval, use upper
                    ab[ji,1] = c[ji]
                else
                    klnew += 1
                    if klnew <= mmax
                        # ew in both intervals -- add upper to queue
                        ab[klnew,2] = ab[ji,2]
                        nab[klnew, 2] = nab[ji,2]
                        ab[klnew, 1] = c[ji]
                        nab[klnew, 1] = nbelow[ji]
                        ab[ji,2] = c[ji]
                        nab[ji,2] = nbelow[ji]
                    else
                        return 0, mmax, c
                    end
                end
            end
            kl = klnew
        else
            # serial version of loop
            klnew = kl
            for ji in kf:kl
                t1 = c[ji]
                t2 = D[1] - t1
                nbelow1 = 0
                if t2 <= pivmin
                    nbelow1 = 1
                    t2 = min(t2, -pivmin)
                end
                for j in 2:n
                    t2 = D[j] - Esq[j-1] / t2 - t1
                    if t2 <= pivmin
                        nbelow1 += 1
                        t2 = min(t2, -pivmin)
                    end
                end
                # insure monotone N(w)
                nbelow1 = min(nab[ji,2], max(nab[ji,1], nbelow1))
                # update queue
                if nbelow1 == nab[ji,2]
                    # use lower
                    ab[ji,2] = t1
                elseif nbelow1 == nab[ji,1]
                    # use upper
                    ab[ji,1] = t1
                elseif klnew < mmax
                    # in both intervals; add upper to queue
                    klnew += 1
                    ab[klnew,2] = ab[ji,2]
                    nab[klnew,2] = nab[ji,2]
                    ab[klnew,1] = t1
                    nab[klnew,1] = nbelow1
                    ab[ji,2] = t1
                    nab[ji,2] = nbelow1
                else
                    return 0, mmax, c
                end
            end
            kl = klnew
        end # parallel/serial switch
        # check for convergence
        kfnew = kf
        for ji in kf:kl
            t1 = abs(ab[ji,2] - ab[ji,1])
            t2 = max(abs(ab[ji,2]), abs(ab[ji,1]))
            if (t1 < max(abstol, pivmin, reltol * t2)) ||
                (nab[ji,1] >= nab[ji,2])
                # converged; swap w/ kfnew and increment
                if ji > kfnew
                    for jj=1:2
                        ab[ji,jj], ab[kfnew,jj] = ab[kfnew,jj], ab[ji,jj]
                        nab[ji,jj], nab[kfnew,jj] = nab[kfnew,jj], nab[ji,jj]
                    end
                end
                kfnew += 1
            end
        end
        kf = kfnew

        # choose midpoints
        resize!(c, kl)
        for ji in kf:kl
            c[ji] = (ab[ji,1] + ab[ji,2]) * one_half
        end
        # if fully refined, quit
        if kf > kl
            break
        end
    end
    n_left = max(kl + 1 - kf, 0)
    return kl, n_left, c
end

# Let N(w) be nr. of eigvals <= w.
# Given initial intervals `ab` and counts `nab`, search each interval `j`
# for wⱼ s.t. N(wⱼ) == nval[j], starting search at `c[j]`.
# If found, store in `ab[j,:]`, else store an interval containing the jump of N(w)
# through `nval[j]`, unless that is not in the original interval.
# translation of dlaebz, ijob=3
# modifies ab, nab, c
function _find_eigval_jumps!(itmax, n, mmax, minp, nbmin, abstol, reltol, pivmin,
                         D::AbstractVector{T}, E, Esq, nval, ab, nab, c) where T
    one_half = 1/T(2)
    # kf:kl is range of intervals needing refinement as we go
    kf = 1
    kl = minp
    if nbmin > 0
        work = zeros(T,mmax)
        nbelow = fill(0,mmax)
    end
    for jit in 1:itmax
        # loop over intervals
        if kl-kf+1 >= nbmin && nbmin > 0
            nbelow = fill(0,kl)
            # parallel version of loop
            for ji in kf:kl
                # compute N(c), nr eigvals < c
                work[ji] = D[1] - c[ji]
                nbelow[ji] = 0
                if work[ji] <= pivmin
                    nbelow[ji] = 1
                    work[ji] = min(work[ji], -pivmin)
                end
                for j in 2:n
                    work[ji] = D[j] - Esq[j-1] / work[ji] - c[ji]
                    if work[ji] < pivmin
                        nbelow[ji] += 1
                        work[ji] = min(work[ji], -pivmin)
                    end
                end
            end
            for ji in kf:kl
                if nbelow[ji] <= nval[ji]
                    ab[ji,1] = c[ji]
                    nab[ji,1] = nbelow[ji]
                end
                if nbelow[ji] >= nval[ji]
                    ab[ji,2] = c[ji]
                    nab[ji,2] = nbelow[ji]
                end
            end
        else
            # serial version of loop
            klnew = kl
            for ji in kf:kl
                t1 = c[ji]
                t2 = D[1] - t1
                nbelow1 = 0
                if t2 <= pivmin
                    nbelow1 = 1
                    t2 = min(t2, -pivmin)
                end
                for j in 2:n
                    t2 = D[j] - Esq[j-1] / t2 - t1
                    if t2 <= pivmin
                        nbelow1 += 1
                        t2 = min(t2, -pivmin)
                    end
                end
                # binary search; keep only inverval containing w s.t. N(w)=nval
                if nbelow1 <= nval[ji]
                    ab[ji,1] = t1
                    nab[ji,1] = nbelow1
                end
                if nbelow1 >= nval[ji]
                    ab[ji,2] = t1
                    nab[ji,2] = nbelow1
                end
            end
            kl = klnew
        end # parallel/serial switch
        # check for convergence
        kfnew = kf
        for ji in kf:kl
            t1 = abs(ab[ji,2] - ab[ji,1])
            t2 = max(abs(ab[ji,2]), abs(ab[ji,1]))
            if (t1 < max(abstol, pivmin, reltol * t2)) ||
                (nab[ji,1] >= nab[ji,2])
                # converged; swap w/ kfnew and increment
                if ji > kfnew
                    for jj=1:2
                        ab[ji,jj], ab[kfnew,jj] = ab[kfnew,jj], ab[ji,jj]
                        nab[ji,jj], nab[kfnew,jj] = nab[kfnew,jj], nab[ji,jj]
                    end
                end
                nval[ji], nval[kfnew] = nval[kfnew], nval[ji]
                kfnew += 1
            end
        end
        kf = kfnew

        # choose midpoints
        resize!(c, kl)
        for ji in kf:kl
            c[ji] = (ab[ji,1] + ab[ji,2]) * one_half
        end
        # if fully refined, quit
        if kf > kl
            break
        end
    end
    n_left = max(kl + 1 - kf, 0)
    return kl, n_left
end

# Compute one accurate eigenvalue of a symm. tridiag. matrix (D,E).
# Assumes scaling so magnitude of largest element is (somewhat) smaller than
# sqrt(floatmax) * sqrt(sqrt(floatmin)).
#
# Ref. W.Kahan, "Accurate Eigenvalues of a Symmetric Tridiagonal Matrix",
# Report CS41, Computer Science Dept., Stanford University, July 21, 1966.
# Translation of DLARRK from LAPACK
function _kahan_eig(iw, D::AbstractVector{T},
                    Esq, lb, ub, pivmin, reltol; fudge=2) where T <: AbstractFloat
    n = length(D)
    myeps = eps(T)
    one_half = 1/T(2)
    tnorm = max(abs(lb),abs(ub))
    atol = fudge * 2 * pivmin
    maxit = floor(Int, log2(tnorm + pivmin) - log2(pivmin)) + 2
    left = lb - fudge * tnorm * myeps * n - fudge * 2 * pivmin
    right = ub + fudge * tnorm * myeps * n + fudge * 2 * pivmin
    converged = false
    for it in 0:maxit
        t1 = abs(right - left)
        t2 = max(abs(right), abs(left))
        if t1 < max(atol, pivmin, reltol * t2)
            converged = true
            break
        end

        # count negative pivots for midpt
        mid = (left + right) * one_half
        negcnt = 0
        t1 = D[1] - mid
        if abs(t1) < pivmin
            t1 = -pivmin
        end
        if t1 <= 0
            negcnt += 1
        end
        @inbounds for i in 2:n
            t1 = D[i] - Esq[i-1] / t1 - mid
            if abs(t1) < pivmin
                t1 = -pivmin
            end
            if t1 <= 0
                negcnt += 1
            end
        end
        if negcnt >= iw
            right = mid
        else
            left = mid
        end
    end
    λ = (left + right) * one_half
    λ_err = abs(right - left) * one_half

    return λ, λ_err, converged
end

# Accurate eigendecomposition of a symmetric real 2x2 Matrix
# translation of dlae2 / dlaev2 from LAPACK
# returns a vector of 2 eigenvalues, largest magnitude first,
# and optionally the unit right eigenvectors
function _eig_sym2x2(a::T, b::T, c::T, wantV) where T <: AbstractFloat
    one_half = 1/T(2)
    sm = a + c
    df = a - c
    adf = abs(df)
    tb = b + b
    ab = abs(tb)
    acmx = a
    acmn = c
    if abs(a) <= abs(c)
        acmn, acmx = acmx, acmn
    end
    if adf > ab
        rt = adf * sqrt(1 + (ab/adf)^2)
    elseif adf < ab
        rt = ab * sqrt(1 + (adf/ab)^2)
    else
        rt = ab * sqrt(T(2))
    end
    # Order of execution important:
    # to get fully accurate smaller EV,
    # rt2 computations should be done in higher precision
    if sm < 0
        rt1 = (sm - rt) * one_half
        sgn1 = -1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    elseif sm > 0
        rt1 = (sm + rt) * one_half
        sgn1 = 1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    else
        rt1 = rt * one_half
        rt2 = - rt * one_half
        sgn1 = 1
    end
    if wantV
        if df >= 0
            cs = df + rt
            sgn2 = 1
        else
            cs = df - rt
            sgn2 = -1
        end
        acs = abs(cs)
        if acs > ab
            ct = -tb / cs
            sn1 = 1 / sqrt(1 + ct^2)
            cs1 = ct * sn1
        else
            if ab == 0
                cs1 = T(1)
                sn1 = T(0)
            else
                tn = -cs / tb
                cs1 = 1 / sqrt(1 + tn^2)
                sn1 = tn * cs1
            end
        end
        if sgn1 == sgn2
            cs1, sn1 = -sn1, cs1
        end
        return [rt1, rt2], [cs1 sn1; -sn1 cs1]
    else # not wantV
        return [rt1,rt2], nothing
    end
end
