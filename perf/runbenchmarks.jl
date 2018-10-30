using RBDJac
using RigidBodyDynamics
using BenchmarkTools
using ForwardDiff
using DiffResults
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
    cache = DifferentialCache{MyTag}(state)
    nq, nv = num_positions(mechanism), num_velocities(mechanism)
    Mjac = zeros(T, nv * nv, nq)
    suite["mass_matrix!"] = @benchmarkable begin
        setdirty!($state)
        mass_matrix_differential!($Mjac, $cache)
    end evals = 10

    statecache = StateCache(mechanism)
    resultcache = DynamicsResultCache(mechanism)
    mass_matrix_vec! = let statecache=statecache, resultcache=resultcache
        function (Mvec, q::AbstractVector{T}) where T
            state = statecache[T]
            setdirty!(state)
            result = resultcache[T]
            set_configuration!(state, q)
            mass_matrix!(result, state)
            copyto!(Mvec, result.massmatrix)
        end
    end
    Mvec = zeros(nv * nv)
    q = configuration(state)
    suite["mass_matrix_vec!"] = @benchmarkable begin
        $mass_matrix_vec!($Mvec, $q)
    end evals=10

    fdjacresult = DiffResults.JacobianResult(Mvec, q)
    config = ForwardDiff.JacobianConfig(mass_matrix_vec!, Mvec, q, ForwardDiff.Chunk(3)) # Chunk(4) and higher results in inference issues
    suite["mass_matrix! ForwardDiff"] = @benchmarkable begin
        ForwardDiff.jacobian!($fdjacresult, $mass_matrix_vec!, $Mvec, $q, $config)
    end evals=10

    return suite
end

function runbenchmarks()
    @show Threads.nthreads()
    suite = create_benchmark_suite()
    Profile.clear_malloc_data()
    overhead = BenchmarkTools.estimate_overhead()
    Random.seed!(1)
    results = run(suite, verbose=true, overhead=overhead, gctrial=false)
    for result in results
        println("$(first(result)):")
        display(last(result))
        println()
    end
end

runbenchmarks()
