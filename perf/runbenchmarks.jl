using RBDJac
using RigidBodyDynamics
using BenchmarkTools
using Random
using Profile

struct MyTag end

function create_benchmark_suite()
    # T = Float32
    T = Float64
    suite = BenchmarkGroup()
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf, scalar_type=T)
    state = MechanismState{T}(mechanism)
    cache = ConfigurationJacobianCache{MyTag}(state)
    nq, nv = num_positions(mechanism), num_velocities(mechanism)
    Mjac = zeros(T, nv * nv, nq)
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
end

runbenchmarks()
