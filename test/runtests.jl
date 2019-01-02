using RigidBodyDynamicsDiff
using RigidBodyDynamics
using ForwardDiff
using DiffResults
using Random
using Test

using RigidBodyDynamics: velocity_to_configuration_derivative_jacobian
using RigidBodyDynamics: configuration_derivative_to_velocity_jacobian


@show Threads.nthreads()

struct MyTag end

@testset "mass_matrix_differential!" begin
    Random.seed!(1)
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf, root_joint_type=QuaternionFloating{Float64}())
    state = MechanismState(mechanism)
    result = DynamicsResult(mechanism)
    rand!(state)

    nq = num_positions(state)
    nv = num_velocities(state)
    q = configuration(state)
    normalize = Ref(false)

    mass_matrix_vec! = function (Mvec, q::AbstractVector{T}) where T
        set_configuration!(fdstate, q)
        if normalize[]
            normalize_configuration!(fdstate)
        end
        mass_matrix!(fdresult, fdstate)
        copyto!(Mvec, fdresult.massmatrix)
    end

    Mvec = zeros(nv * nv)
    fdjacresult = DiffResults.JacobianResult(Mvec, q)
    config = ForwardDiff.JacobianConfig(mass_matrix_vec!, Mvec, q, ForwardDiff.Chunk(1))
    fdstate = MechanismState{eltype(config)}(mechanism)
    fdresult = DynamicsResult{eltype(config)}(mechanism)

    cache = DifferentialCache{MyTag}(state)
    dM = Matrix{Float64}(undef, nv^2, nv)
    mass_matrix_differential!(dM, cache)

    # Without normalization:
    normalize[] = false
    ForwardDiff.jacobian!(fdjacresult, mass_matrix_vec!, Mvec, q, config)
    fdMjac = DiffResults.jacobian(fdjacresult)
    # Without normalization, the normal component in dM is zero (since we're only working
    # in the tangent plane), but the ForwardDiff version *does* have a normal component,
    # so the following is not true:
    #   @test dM * configuration_derivative_to_velocity_jacobian(state) ≈ fdMjac atol = 1e-10
    # but this is:
    @test dM ≈ fdMjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10

    # With normalization:
    normalize[] = true
    ForwardDiff.jacobian!(fdjacresult, mass_matrix_vec!, Mvec, q, config)
    fdMjac = DiffResults.jacobian(fdjacresult)
    @test dM * configuration_derivative_to_velocity_jacobian(state) ≈ fdMjac atol = 1e-10
    @test dM ≈ fdMjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10
end

@testset "benchmarks" begin
    include(joinpath(@__DIR__, "..", "perf", "runbenchmarks.jl"))
end
