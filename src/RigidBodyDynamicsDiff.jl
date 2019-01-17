module RigidBodyDynamicsDiff

export
    DifferentialCache,
    mass_matrix_differential!,
    dynamics_bias_differential!,
    inverse_dynamics_differential!

using RigidBodyDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays

using RigidBodyDynamics: successorid, isdirty, Spatial.colwise, Spatial.hat, Spatial.se3_commutator
using RigidBodyDynamics: update_transforms!, update_bias_accelerations_wrt_world!, update_twists_wrt_world!
using RigidBodyDynamics: update_motion_subspaces!, update_spatial_inertias!, update_crb_inertias!
using RigidBodyDynamics: configuration_index_to_joint_id, velocity_index_to_joint_id
using RigidBodyDynamics: velocity_to_configuration_derivative_jacobian, velocity_to_configuration_derivative_jacobian!
using RigidBodyDynamics.CustomCollections: SegmentedBlockDiagonalMatrix
using ForwardDiff: Dual

# TODO:
# * more frame checks for timederiv methods

function timederiv(H::Transform3D, twist::Twist, ::Type{Tag}) where Tag
    # @framecheck H.from twist.body
    # @framecheck H.to twist.base
    # @framecheck twist.frame twist.base
    R = rotation(H)
    p = translation(H)
    ω = angular(twist)
    v = linear(twist)
    dR = colwise(×, ω, R)
    dp = ω × p + v
    T = promote_type(eltype(dR), eltype(dp))
    dH = hcat(
        vcat(dR, zeros(similar_type(dR, Size(1, 3)))),
        vcat(dp, zeros(similar_type(dR, Size(1, 1))))
    )
    Transform3D(H.from, H.to,
        map(Dual{Tag}, H.mat, dH))
end

function timederiv(J::GeometricJacobian, twist::Twist, ::Type{Tag}) where Tag
    # TODO: more frame checks needed?
    @framecheck J.frame twist.frame
    ω = angular(twist)
    v = linear(twist)
    Jω = angular(J)
    Jv = linear(J)
    dJω = colwise(×, ω, Jω)
    dJv = colwise(×, v, Jω) + colwise(×, ω, Jv)
    GeometricJacobian(J.body, J.base, J.frame,
        map(Dual{Tag}, Jω, dJω),
        map(Dual{Tag}, Jv, dJv))
end

function timederiv(twist1::Twist, twist2::Twist, ::Type{Tag}) where Tag
    @framecheck twist1.frame twist2.frame
    dangular, dlinear = se3_commutator(angular(twist2), linear(twist2), angular(twist1), linear(twist1))
    Twist(twist1.body, twist1.base, twist1.frame,
        map(Dual{Tag}, angular(twist1), dangular),
        map(Dual{Tag}, linear(twist1), dlinear))
end

function timederiv(accel::SpatialAcceleration, twist::Twist, ::Type{Tag}) where Tag
    @framecheck accel.frame twist.frame
    dangular, dlinear = se3_commutator(angular(accel), linear(accel), angular(twist), linear(twist))
    SpatialAcceleration(accel.body, accel.base, accel.frame,
        map(Dual{Tag}, angular(accel), dangular),
        map(Dual{Tag}, linear(accel), dlinear))
end

function timederiv(inertia::SpatialInertia, twist::Twist, ::Type{Tag}) where Tag
    # TODO: more frame checks needed?
    @framecheck inertia.frame twist.frame
    ω = angular(twist)
    v = linear(twist)
    Iω = inertia.moment
    mc = inertia.cross_part
    m = inertia.mass
    mv = m * v
    dmc = mv + ω × mc
    ωhat = hat(ω)
    vhat = hat(v)
    mchat = hat(mc)
    dIω = ωhat * Iω - Iω * ωhat - vhat * mchat - mchat * vhat
    SpatialInertia(inertia.frame,
        map(Dual{Tag}, Iω, dIω),
        map(Dual{Tag}, mc, dmc),
        Dual{Tag}(m, zero(m)))
end

struct DifferentialCache{Tag, T, M, S<:MechanismState{T, M}, DS<:MechanismState{Dual{Tag, T, 1}, M}}
    state::S
    dualstates::Vector{DS}
    dualresults::Vector{DynamicsResult{Dual{Tag, T, 1}, M}}
    v_to_q̇_jacobian::SegmentedBlockDiagonalMatrix{T, Matrix{T}}
end

function DifferentialCache{Tag}(state::MechanismState{T}) where {Tag, T}
    mechanism = state.mechanism
    if !mapreduce(has_fixed_subspaces, &, state.treejoints, init=true)
        throw(ArgumentError("Can only handle Mechanisms with fixed motion subspaces."))
    end
    D = ForwardDiff.Dual{Tag, T, 1}
    n = Threads.nthreads()
    dualstates = Vector{typeof(MechanismState{D}(mechanism))}(undef, n)
    dualresults = Vector{typeof(DynamicsResult{D}(mechanism))}(undef, n)
    Threads.@threads for i = 1 : n
        # Create MechanismStates and DynamicsResults in separate threads to avoid the possibility of false sharing.
        id = Threads.threadid()
        dualstate = dualstates[id] = MechanismState{D}(mechanism)
        dualresults[id] = DynamicsResult{D}(mechanism)
        dualstate.q .= NaN
        dualstate.v .= NaN
        dualstate.s .= NaN
    end
    v_to_q̇_jacobian = velocity_to_configuration_derivative_jacobian(state)
    DifferentialCache(state, dualstates, dualresults, v_to_q̇_jacobian)
end

function copy_differential_column!(dest::Matrix, src::AbstractVector{<:Dual}, destcol::Integer)
    deststart = (destcol - 1) * size(dest, 1)
    @inbounds @simd for row in eachindex(src)
        dest[deststart + row] = ForwardDiff.partials(src[row], 1)
    end
end

function copy_differential_column!(dest::Matrix, src::Symmetric{<:Dual}, destcol::Integer)
    deststart = (destcol - 1) * size(dest, 1)
    n = size(src, 1)
    upper = src.uplo === 'u'
    @inbounds for col in Base.OneTo(n)
        @simd for row in col : n
            u_index = (row - 1) * n + col
            l_index = (col - 1) * n + row
            data_index = ifelse(upper, u_index, l_index)
            val = ForwardDiff.partials(src.data[data_index], 1)
            dest[deststart + u_index] = val
            dest[deststart + l_index] = val
        end
    end
end

function dual_state_init!(dualstate::MechanismState{<:Dual{Tag}}, state::MechanismState{T}, v_index::Integer,
        transforms::Bool, motion_subspaces::Bool, inertias::Bool, crb_inertias::Bool,
        twists::Bool, bias_accelerations::Bool) where {Tag, T}
    dualstate.q .= NaN # TODO: actually compute this?
    copyto!(dualstate.v, state.v)
    copyto!(dualstate.s, state.s)
    setdirty!(dualstate)
    k = velocity_index_to_joint_id(state, v_index)
    twist = Twist(state.motion_subspaces.data[v_index], SVector(one(T)))

    # TODO:
    if twists || bias_accelerations
        transforms = true
    end

    if transforms
        dtransforms_to_root = dualstate.transforms_to_root
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            H = transform_to_root(state, bodyid, false)
            if κ_i[k]
                dtransforms_to_root[bodyid] = timederiv(H, twist, Tag)
            else
                dtransforms_to_root[bodyid] = H
            end
        end
        dualstate.transforms_to_root.dirty = false
    end

    if motion_subspaces
        dmotion_subspaces = dualstate.motion_subspaces.data
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            for v_index_i in velocity_range(state, i)
                S = state.motion_subspaces.data[v_index_i]
                if κ_i[k]
                    dmotion_subspaces[v_index_i] = timederiv(S, twist, Tag)
                else
                    dmotion_subspaces[v_index_i] = S
                end
            end
        end
        dualstate.motion_subspaces.dirty = false
    end

    if inertias
        dinertias = dualstate.inertias
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            I_i = state.inertias[bodyid]
            if κ_i[k]
                dinertias[bodyid] = timederiv(I_i, twist, Tag)
            else
                dinertias[bodyid] = I_i
            end
        end
        dualstate.inertias.dirty = false
    end

    if crb_inertias
        dcrb_inertias = dualstate.crb_inertias
        κ_k = state.ancestor_joint_masks[k]
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            Ic_i = state.crb_inertias[bodyid]
            if κ_i[k] || κ_k[i]
                p = max(i, k)
                Ic_p = state.crb_inertias[successorid(p, state)]
                Ic_p_dual = timederiv(Ic_p, twist, Tag)
                Iω = map(Dual{Tag}, Ic_i.moment, map(ForwardDiff.partials, Ic_p_dual.moment))
                mc = map(Dual{Tag}, Ic_i.cross_part, map(ForwardDiff.partials, Ic_p_dual.cross_part))
                m = Dual{Tag}(Ic_i.mass, ForwardDiff.partials(Ic_p_dual.mass))
                dcrb_inertias[bodyid] = SpatialInertia(Ic_i.frame, Iω, mc, m)
            else
                dcrb_inertias[bodyid] = Ic_i
            end
        end
        dualstate.crb_inertias.dirty = false
    end

    # if twists
    #     dtwists = dualstate.twists_wrt_world
    #     @inbounds for i in state.treejointids
    #         κ_i = state.ancestor_joint_masks[i]
    #         bodyid = successorid(i, state)
    #         twist_i = twist_wrt_world(state, bodyid, false)
    #         if κ_i[k]
    #             dtwists[bodyid] = timederiv(twist_i, twist, Tag)
    #         else
    #             dtwists[bodyid] = twist_i
    #         end
    #     end
    #     dualstate.twists_wrt_world.dirty = false
    # end

    # if bias_accelerations
    #     dbias_accelerations = dualstate.bias_accelerations_wrt_world
    #     @inbounds for i in state.treejointids
    #         κ_i = state.ancestor_joint_masks[i]
    #         bodyid = successorid(i, state)
    #         accel_i = bias_acceleration(state, bodyid, false)
    #         if κ_i[k]
    #             dbias_accelerations[bodyid] = timederiv(accel_i, twist, Tag)
    #         else
    #             dbias_accelerations[bodyid] = accel_i
    #         end
    #     end
    #     dualstate.bias_accelerations_wrt_world.dirty = false
    # end

    nothing
end

function threaded_differential!(f::F, dest::Matrix, cache::DifferentialCache;
        transforms::Bool=false, motion_subspaces::Bool=false, inertias::Bool=false, crb_inertias::Bool=false, twists::Bool=false, bias_accelerations::Bool=false) where F
    velocity_to_configuration_derivative_jacobian!(cache.v_to_q̇_jacobian, cache.state)
    let state = cache.state
        transforms && update_transforms!(state)
        motion_subspaces && update_motion_subspaces!(state)
        inertias && update_spatial_inertias!(state)
        crb_inertias && update_crb_inertias!(state)
        twists && update_twists_wrt_world!(state)
        bias_accelerations && update_bias_accelerations_wrt_world!(state)
    end
    let dest = dest, cache = cache, state = cache.state
        nv = num_velocities(state)
        Threads.@threads for v_index in Base.OneTo(nv)
            id = Threads.threadid()
            dualstate = cache.dualstates[id]
            dual_state_init!(dualstate, state, v_index, transforms, motion_subspaces,
                inertias, crb_inertias, twists, bias_accelerations)
            dualresult = cache.dualresults[id]
            copy_differential_column!(dest, f(dualresult, dualstate), v_index)
            nothing
        end
    end
    return dest
end

function mass_matrix_differential!(dest::Matrix, cache::DifferentialCache)
    nv = num_velocities(cache.state)
    size(dest) == (nv^2, nv) || throw(DimensionMismatch())
    threaded_differential!(mass_matrix!, dest, cache;
        motion_subspaces=true, crb_inertias=true)
end

function dynamics_bias_differential!(dest::Matrix, cache::DifferentialCache)
    nv = num_velocities(cache.state)
    size(dest) == (nv, nv) || throw(DimensionMismatch())
    threaded_differential!(dynamics_bias!, dest, cache;
        motion_subspaces=true, inertias=true, bias_accelerations=true, twists=true)
end

function inverse_dynamics_differential!(dest::Matrix, v̇::SegmentedVector{JointID}, cache::DifferentialCache)
    nv = num_velocities(cache.state)
    size(dest) == (nv, nv) || throw(DimensionMismatch())
    f = let v̇ = v̇
        function (result, state)
            # TODO: abusing the dynamicsbias and totalwrenches fields of DynamicsResult.
            inverse_dynamics!(result.dynamicsbias, result.jointwrenches, result.accelerations, state, v̇, result.totalwrenches)
            result.dynamicsbias
        end
    end
    threaded_differential!(f, dest, cache;
        transforms=true, motion_subspaces=true, inertias=true, crb_inertias=true, bias_accelerations=true, twists=true)
end

end # module
