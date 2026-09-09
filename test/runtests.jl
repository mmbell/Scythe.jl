# Scythe.jl test suite.
#
#   julia --project=. test/runtests.jl                 # everything, canonical order (~35 min)
#   GROUP=io julia --project=. test/runtests.jl        # one group
#   GROUP=core,physics julia --project=. test/runtests.jl
#
# Groups and the file table live in test/test_groups.jl. GROUP is resolved BEFORE Scythe is
# loaded so a typo fails in under a second, not after the package compiles. (Pkg.test stalls
# under check-bounds; run this file directly.)
include("test_groups.jl")
const GROUP = get(ENV, "GROUP", "all")
const FILES = test_files_for(GROUP)
println("Scythe tests: GROUP=$(GROUP) ($(length(FILES)) of $(length(TEST_FILES)) files)")

using Test
using Scythe

@testset "Scythe.jl" begin
    for f in FILES
        include(f)
    end
end
