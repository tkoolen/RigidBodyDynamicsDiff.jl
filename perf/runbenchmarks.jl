using RBDJac
using RigidBodyDynamics
using BenchmarkTools
using Random
using Profile

const ScalarType = Float64
# const ScalarType = Float32

struct MyTag end

function create_benchmark_suite()
    suite = BenchmarkGroup()
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    cache = ConfigurationJacobianCache{MyTag}(state)
    nq, nv = num_positions(mechanism), num_velocities(mechanism)
    Mjac = zeros(nv * nv, nq)
    suite["mass_matrix!"] = @benchmarkable begin
        setdirty!($state)
        mass_matrix_jacobian!($Mjac, $cache)
    end evals = 10
end

function runbenchmarks()
    @show Threads.nthreads()
    suite = create_benchmark_suite()
    Profile.clear_malloc_data()
    overhead = BenchmarkTools.estimate_overhead()
    Random.seed!(1)
    results = run(suite, verbose=true, overhead=overhead, gctrial=false)
    @show results
    # for result in results
    #     println("$(first(result)):")
    #     display(last(result))
    #     println()
    # end
end

runbenchmarks()
