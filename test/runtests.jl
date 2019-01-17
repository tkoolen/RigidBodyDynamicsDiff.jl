using RigidBodyDynamicsDiff
using RigidBodyDynamics
using ForwardDiff
using DiffResults
using Random
using Test

using RigidBodyDynamics: velocity_to_configuration_derivative_jacobian
using RigidBodyDynamics: configuration_derivative_to_velocity_jacobian

include(joinpath(@__DIR__, "..", "test", "forwarddiff_compatible.jl"))

@show Threads.nthreads()

struct MyTag end

@testset "differential" begin
    Random.seed!(1)

    T = Float64
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf, scalar_type=T, root_joint_type=QuaternionFloating{T}())
    state = MechanismState{T}(mechanism)
    rand!(state)
    q = configuration(state)
    v = velocity(state)
    v̇ = rand!(similar(velocity(state)))
    nv = num_velocities(mechanism)
    statecache = StateCache(mechanism)
    resultcache = DynamicsResultCache(mechanism)
    normalize = Ref(false)

    configs = []
    push!(configs, (
        "mass_matrix",
        mass_matrix_differential!,
        forwarddiff_compatible(mass_matrix!, statecache, resultcache, normalize),
        zeros(nv * nv)
    ))
    push!(configs, (
        "dynamics_bias",
        dynamics_bias_differential!,
        forwarddiff_compatible(dynamics_bias!, statecache, resultcache, v, normalize),
        zeros(nv)
    ))

    push!(configs, (
        "inverse_dynamics",
        (dest, cache) -> inverse_dynamics_differential!(dest, v̇, cache),
        forwarddiff_compatible(inverse_dynamics!, statecache, resultcache, v, v̇, normalize),
        zeros(nv)
    ))

    for (name, differentialfun, fdfun, out) in configs
        @testset "$name" begin
            diffcache = DifferentialCache{MyTag}(state)

            fdjacresult = DiffResults.JacobianResult(out, q)
            fdconfig = ForwardDiff.JacobianConfig(fdfun, out, q, ForwardDiff.Chunk(1))
            differential = fill(T(NaN), length(out), nv)
            differentialfun(differential, diffcache)

            # Without normalization:
            normalize[] = false
            ForwardDiff.jacobian!(fdjacresult, fdfun, out, q, fdconfig)
            fdjac = DiffResults.jacobian(fdjacresult)
            @test differential ≈ fdjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10

            # Without normalization, the normal component in dM is zero (since we're only working
            # in the tangent plane), but the ForwardDiff version *does* have a normal component,
            # so the following is not true:
            #   @test dM * configuration_derivative_to_velocity_jacobian(state) ≈ fdjac atol = 1e-10
            # But with normalization:

            normalize[] = true
            ForwardDiff.jacobian!(fdjacresult, fdfun, out, q, fdconfig)
            fdjac = DiffResults.jacobian(fdjacresult)
            @test differential * configuration_derivative_to_velocity_jacobian(state) ≈ fdjac atol = 1e-10
            @test differential ≈ fdjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10
        end
    end
end

@testset "benchmarks" begin
    include(joinpath(@__DIR__, "..", "perf", "runbenchmarks.jl"))
end
