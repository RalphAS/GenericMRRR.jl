# GenericMRRR.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
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

## A Symmetric Tridiagonal Eigen-solver in Julia: MRRR and dqds methods

This package provides methods for computing some or all of the
eigenvalues and (optionally) eigenvectors of a real symmetric
tridiagonal matrix. It is designed to work with
element types supported by arithmetic similar to IEEE standards (especially
NaN propagation), such as `BigFloat`, `Quadmath.Float128`, and
`DoubleFloat`.

## Usage

The package exports `geigen!` and `geigvals!` functions with the
same API as the `SymTridiagonal` versions of `eigen!` and `eigvals!`
in the `LinearAlgebra` standard library. For example,

```
n=128
D = rand(n)
D1 = rand(n-1)
A = SymTridiagonal(D, D1)
# only compute the first 10 eigenvalues and corresponding eigenvectors
E = geigen!(copy(A), 1:10)
# only compute eigenvalues in an interval
E = geigvals!(copy(A), -1.0, 1.0)
```

It also implements `geigen!` and `geigvals!` methods
for dense `Hermitian` matrices, when the `MRRR` algorithm is selected.
```
A = Hermitian(rand(n,n))
E = geigen!(copy(A), MRRR())
```

## Algorithms

The methods implement the Relatively Robust Representation scheme
of Dhillon, Parlett, and associates, along with the dqds scheme for
computing eigenvalues to high precision where feasible. The implementations
closely follow those in LAPACK.

### Notes

There are a few matrices for which the `Float64` implementations in
LAPACK fail; our implementation typically also fails for them -
but often succeeds after promoting the same matrices to a wider type.
In most such cases a `RepresentationFailure` exception is thrown.

Although these algorithms are usually more efficient than the Francis-QR
scheme (which is implemented in `GenericLinearAlgebra`), they are far
more complicated so the initial compilation takes several seconds.

A small pseudo-random perturbation is used in the algorithm.
Unlike the LAPACK implementation, we use the default Julia RNG(s) without
altering the seed, so there will be variations between results on a given matrix.
These should almost always be roundoff-level, except for eigenvectors of effectively
indistinguishable eigenvalues.

## Credits

Almost all of this package was translated from LAPACK subroutines
written by Dhillon, Parlett, Marques, et al. Translation errors are
(of course) not their responsibility. Efforts have been made to
be consistent with LAPACK behavior, but the agreement is not perfect.

Special thanks to Osni Marques for providing a collection of matrices
for testing and debugging.

## References

I.S.Dhillon & B.N.Parlett, "Multiple representations to compute orthogonal eigenvectors of
symmetric tridiagonal matrices," Linear Algebra Appl. 387, 1-28 (2004).

I.S.Dhillon, B.N.Parlett & C. Vömel, "The design and implementation of the MRRR algorithm,"
ACM TOMS 32, 533-560 (2006).

B.N.Parlett & O.A.Marques, "An implementation of the dqds algorithm,"
Linear Algebra Appl. 309, 217-259 (2000).
