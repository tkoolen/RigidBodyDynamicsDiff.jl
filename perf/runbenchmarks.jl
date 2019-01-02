using RigidBodyDynamicsDiff
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
    rand!(state)
    q = configuration(state)
    v = velocity(state)
    nv = num_velocities(mechanism)

    statecache = StateCache(mechanism)
    resultcache = DynamicsResultCache(mechanism)
    diffcache = DifferentialCache{MyTag}(state)

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
    suite["mass_matrix_vec!"] = @benchmarkable begin
        $mass_matrix_vec!($Mvec, $q)
    end evals=10

    fdjacresult = DiffResults.JacobianResult(Mvec, q)
    config = ForwardDiff.JacobianConfig(mass_matrix_vec!, Mvec, q, ForwardDiff.Chunk(3)) # Chunk(4) and higher results in inference issues
    suite["mass_matrix! ForwardDiff"] = @benchmarkable begin
        ForwardDiff.jacobian!($fdjacresult, $mass_matrix_vec!, $Mvec, $q, $config)
    end evals=10

    dM = zeros(T, nv * nv, nv)
    suite["mass_matrix!"] = @benchmarkable begin
        setdirty!($state)
        mass_matrix_differential!($dM, $diffcache)
    end evals = 10

    dynamics_bias_vec! = let statecache=statecache, resultcache=resultcache, v=v
        function (cvec, q::AbstractVector{T}) where T
            state = statecache[T]
            setdirty!(state)
            result = resultcache[T]
            set_configuration!(state, q)
            set_velocity!(state, v)
            dynamics_bias!(result, state)
            copyto!(cvec, result.dynamicsbias)
        end
    end

    cvec = zeros(nv)
    suite["dynamics_bias_vec!"] = @benchmarkable begin
        $dynamics_bias_vec!($cvec, $q)
    end evals=10

    fdjacresult = DiffResults.JacobianResult(cvec, q)
    config = ForwardDiff.JacobianConfig(dynamics_bias_vec!, cvec, q, ForwardDiff.Chunk(3)) # Chunk(4) and higher results in inference issues
    suite["dynamics_bias! ForwardDiff"] = @benchmarkable begin
        ForwardDiff.jacobian!($fdjacresult, $dynamics_bias_vec!, $cvec, $q, $config)
    end evals=10
    dc = zeros(T, nv, nv)
    suite["dynamics_bias!"] = @benchmarkable begin
        setdirty!($state)
        dynamics_bias_differential!($dM, $diffcache)
    end evals = 10

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
