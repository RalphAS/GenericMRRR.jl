using GenericMRRR
using Test
using Random

using Aqua

Aqua.test_all(GenericMRRR)

Random.seed!(1103)

# TODO: git clone https://github.com/oamarques/stester.git stester

include("symtridiag.jl")

include("hermitian.jl")
