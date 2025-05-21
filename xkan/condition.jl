"""
    UBR(p::Float64, q::Float64)

Unordered Bound hyperrectangular representation (UBR) for hyperrectangle bounds [Stone & Bull, 2003].
Stores bounds as unordered pair (p, q) with utility methods for ordered access.

# Fields
- `p::Float64`: First bound value (unordered)
- `q::Float64`: Second bound value (unordered)
"""
mutable struct UBR
    p::Float64
    q::Float64

    # Inner constructor for value validation
    function UBR(p::Float64, q::Float64)
        new(p, q)
    end
end

# function UBR(p, q)
#     return UBR(p, q)
# end


"""
    get_lower_bound(ubr::UBR) -> Float64

Returns the mathematically lower bound of the hyperrectangle.
Handles unordered storage by returning min(p, q).
"""
function get_lower_bound(ubr::UBR)::Float64
    return min(ubr.p, ubr.q)
end

"""
    get_upper_bound(ubr::UBR) -> Float64

Returns the mathematically upper bound of the hyperrectangle.
Handles unordered storage by returning max(p, q).
"""
function get_upper_bound(ubr::UBR)::Float64
    return max(ubr.p, ubr.q)
end

"""
    get_lower_upper_bounds(ubr::UBR) -> Tuple{Float64,Float64}

Returns ordered bounds as tuple (lower, upper).
Guarantees lower <= upper regardless of storage order.
"""
function get_lower_upper_bounds(ubr::UBR)::Tuple{Float64,Float64}
    l = min(ubr.p, ubr.q)
    u = max(ubr.p, ubr.q)
    return (l, u)
end

"""
    is_equal(a::UBR, b::UBR) -> Bool

Structural equality check considering unordered nature.
Returns true if both UBRs represent the same hyperrectangle bounds.

# Notes
- Uses exact floating-point comparison (no epsilon tolerance)
- (p,q) vs (q,p) are considered equal if bounds match
"""
function is_equal(a::UBR, b::UBR)::Bool
    (min(a.p, a.q), max(a.p, a.q)) == (min(b.p, b.q), max(b.p, b.q))
end

