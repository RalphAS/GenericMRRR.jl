# GenericMRRR.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/RalphAS/GenericMRRR.jl.svg?branch=master)](https://travis-ci.org/RalphAS/GenericMRRR.jl)
[![codecov.io](http://codecov.io/github/RalphAS/GenericMRRR.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/GenericMRRR.jl?branch=master)
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://RalphAS.github.io/GenericMRRR.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://RalphAS.github.io/GenericMRRR.jl/dev)
-->

## The Symmetric Eigensystem Problem in Julia: MRRR methods

This package provides methods for computing some or all of the
eigenvalues and (optionally) eigenvectors of a real symmetric
tridiagonal matrix. It is designed to work with
element types supported by arithmetic similar to IEEE standards (especially
NaN propagation), such as `BigFloat`, `Quadmath.Float128`, and
`DoubleFloat`.

## Usage

The package exports `geigen!` and `geigvals!` functions with the
same API as the `SymTridiagonal` versions of `eigen!` and `eigvals!`
in the `LinearAlgebra` standard library.

For convenience, it also implements `geigen!` and `geigvals!` methods
for dense Hermitian matrices, but the reduction to tridiagonal form
(`ghessenberg!`) is rather simplistic.


## Multiple Relatively Robust Representation methods

The methods implement the Relatively Robust Representation scheme
of Dhillon, Parlett, and associates.

Almost all of this package was translated from the LAPACK subroutines
written by Dhillon, Parlett, Marques, et al. Translation errors are
(of course) not their responsibility. Efforts have been made to
be consistent with LAPACK behavior, but the agreement is not perfect.

There are a few matrices for which
the methods fail in `Float64` in LAPACK, and this package typically also
fails for them - but often succeeds after promoting the same matrices to
a wider type.

Caveats: the limits of overflow and underflow have not been thoroughly tested.

## References

I.S.Dhillon & B.N.Parlett, "Multiple representations to compute orthogonal eigenvectors of
symmetric tridiagonal matrices," Linear Algebra and Appl. 387, 1-28 (2004).

I.S.Dhillon, B.N.Parlett & C. Vömel, "The design and implementation of the MRRR algorithm,"
ACM TOMS 32, 533-560 (2006).

B.N.Parlett & O.A.Marques, "An implementation of the dqds algorithm," Linear Algebra Appl. 309, 217-259 (2000).
