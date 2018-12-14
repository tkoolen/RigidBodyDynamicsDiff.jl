module RigidBodyDynamicsDiff

export
    DifferentialCache,
    mass_matrix_differential!

using RigidBodyDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays

using RigidBodyDynamics: successorid, isdirty, Spatial.colwise, Spatial.hat
using RigidBodyDynamics: update_motion_subspaces!, update_crb_inertias!
using RigidBodyDynamics: configuration_index_to_joint_id, velocity_index_to_joint_id
using RigidBodyDynamics: velocity_to_configuration_derivative_jacobian, velocity_to_configuration_derivative_jacobian!
using RigidBodyDynamics.CustomCollections: SegmentedBlockDiagonalMatrix
using ForwardDiff: Dual

# TODO:
# * need mapping from q index / v index to JointID (stored as `Vector{JointID}` in `MechanismState`)
# * frame checks for timederiv methods
# * create MechanismStates in separate threads (after making MechanismState construction threadsafe)

function timederiv(H::Transform3D, twist::Twist, ::Type{Tag}) where Tag
    # @framecheck H.from twist.body
    # @framecheck H.to twist.base
    # @framecheck twist.frame twist.base
    R = rotation(H)
    p = translation(H)
    ω = angular(twist)
    v = linear(twist)
    top = [colwise(×, ω, R) ω × p + v]
    bottom = zeros(similar_type(top, Size(1, 4)))
    dH = [top; bottom]
    Transform3D(H.from, H.to, map(Dual{Tag}, H.mat, dH))
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
    GeometricJacobian(J.body, J.base, J.frame, map(Dual{Tag}, Jω, dJω), map(Dual{Tag}, Jv, dJv))
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
    SpatialInertia(inertia.frame, map(Dual{Tag}, Iω, dIω), map(Dual{Tag}, mc, dmc), Dual{Tag}(m, zero(m)))
end

struct DifferentialCache{Tag, T, M, S<:MechanismState{T, M}, DS<:MechanismState{Dual{Tag, T, 1}, M}}
    state::S
    dualstates::Vector{DS}
    dualresults::Vector{DynamicsResult{Dual{Tag, T, 1}, M}}
    v_to_q̇_jacobian::SegmentedBlockDiagonalMatrix{T, Matrix{T}}
end

function DifferentialCache{Tag}(state::MechanismState{T}) where {Tag, T}
    mechanism = state.mechanism
    mapreduce(has_fixed_subspaces, &, state.treejoints, init=true) || error("Can only handle Mechanisms with fixed motion subspaces.")
    D = ForwardDiff.Dual{Tag, T, 1}
    n = Threads.nthreads()
    dualstates = [MechanismState{D}(mechanism) for _ = 1 : n]
    dualresults = [DynamicsResult{D}(mechanism) for _ = 1 : n]
    for dualstate in dualstates
        dualstate.q .= NaN
        dualstate.v .= NaN
        dualstate.s .= NaN
    end
    v_to_q̇_jacobian = velocity_to_configuration_derivative_jacobian(state)
    DifferentialCache(state, dualstates, dualresults, v_to_q̇_jacobian)
end

function update_jac_state!(cache::DifferentialCache{Tag, T}, jac_index::Integer, v_index::Integer;
        transforms::Bool, motion_subspaces::Bool, inertias::Bool, crb_inertias::Bool) where {Tag, T}
    state = cache.state
    dualstate = cache.dualstates[jac_index]
    setdirty!(dualstate)
    k = velocity_index_to_joint_id(state, v_index)
    twist = Twist(state.motion_subspaces.data[v_index], SVector(one(T)))
    if transforms
        # RigidBodyDynamics.update_transforms!(state)
        dtransforms_to_root = dualstate.transforms_to_root
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            H = transform_to_root(state, bodyid)
            if κ_i[k]
                dtransforms_to_root[bodyid] = timederiv(H, twist, Tag)
            else
                dtransforms_to_root[bodyid] = H
            end
        end
        dualstate.transforms_to_root.dirty = false
    end

    if motion_subspaces
        # RigidBodyDynamics.update_motion_subspaces!(state)
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
        # RigidBodyDynamics.update_spatial_inertias!(state)
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
        # RigidBodyDynamics.update_crb_inertias!(state)
        dcrbinertias = dualstate.crb_inertias
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
                dcrbinertias[bodyid] = SpatialInertia(Ic_i.frame, Iω, mc, m)
            else
                dcrbinertias[bodyid] = Ic_i
            end
        end
        dualstate.crb_inertias.dirty = false
    end
    return nothing
end

function copy_differential_column!(dest::Matrix, src::Symmetric{<:Dual}, destcol::Integer)
    deststart = (destcol - 1) * size(dest, 1)
    n = size(src, 1)
    upper = src.uplo === 'u'
    @inbounds for col in 1 : n
        for row in col : n
            u_index = (row - 1) * n + col
            l_index = (col - 1) * n + row
            data_index = ifelse(upper, u_index, l_index)
            val = ForwardDiff.partials(src.data[data_index], 1)
            dest[deststart + u_index] = val
            dest[deststart + l_index] = val
        end
    end
end

function mass_matrix_differential!(differential::Matrix, cache::DifferentialCache)
    state = cache.state
    nv = num_velocities(state)
    @boundscheck size(differential) === (nv^2, nv) || throw(DimensionMismatch())
    update_motion_subspaces!(state)
    update_crb_inertias!(state)
    velocity_to_configuration_derivative_jacobian!(cache.v_to_q̇_jacobian, state)

    let cache=cache, nv = nv
        Threads.@threads for v_index in Base.OneTo(nv)
            id = Threads.threadid()
            update_jac_state!(cache, id, v_index, transforms=false, motion_subspaces=true, inertias=false, crb_inertias=true)
            @inbounds dualstate = cache.dualstates[id]
            @inbounds jacresult = cache.dualresults[id]
            mass_matrix!(jacresult, dualstate)
            copy_differential_column!(differential, jacresult.massmatrix, v_index)
        end
    end
    differential
end

# function dynamics_bias_differential!(jac::Matrix, cache::DifferentialCache)
#     state = cache.state
#     dualstates = cache.dualstates
#     dualresults = cache.dualresults
#     nq = num_positions(state)
#     nv = num_velocities(state)
#     @boundscheck size(jac) === (nv, nq) || throw(DimensionMismatch())
#     update_transforms!(state)
#     update_twists_wrt_world!(state)
#     update_bias_accelerations_wrt_world!(state)
#     update_spatial_inertias!(state)

# end

end # module
