using RigidBodyDynamics

function forwarddiff_compatible(::typeof(mass_matrix!),
        statecache::StateCache, resultcache::DynamicsResultCache, normalize = Ref(false))
    function (out::AbstractVector, q::AbstractVector{T}) where T
        state = statecache[T]
        result = resultcache[T]
        set_configuration!(state, q)
        if normalize[]
            normalize_configuration!(state)
        end
        mass_matrix!(result, state)
        copyto!(out, result.massmatrix)
    end
end

function forwarddiff_compatible(::typeof(dynamics_bias!),
        statecache::StateCache, resultcache::DynamicsResultCache, v::AbstractVector, normalize = Ref(false))
    function (out::AbstractVector, q::AbstractVector{T}) where T
        state = statecache[T]
        result = resultcache[T]
        set_configuration!(state, q)
        set_velocity!(state, v)
        if normalize[]
            normalize_configuration!(state)
        end
        dynamics_bias!(result, state)
        copyto!(out, result.dynamicsbias)
    end
end

function forwarddiff_compatible(::typeof(inverse_dynamics!),
        statecache::StateCache, resultcache::DynamicsResultCache, v::AbstractVector, v̇::AbstractVector, normalize = Ref(false))
    function (out::AbstractVector, q::AbstractVector{T}) where T
        state = statecache[T]
        result = resultcache[T]
        set_configuration!(state, q)
        set_velocity!(state, v)
        if normalize[]
            normalize_configuration!(state)
        end
        inverse_dynamics!(result.dynamicsbias, result.jointwrenches, result.accelerations, state, v̇, result.totalwrenches)
        copyto!(out, result.dynamicsbias)
    end
end
