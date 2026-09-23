# PositionSpaces.jl — policies for physical positions embedded in tensor registers

"""
    AbstractPositionSpace

Policy object describing how physical positions are embedded in the tensor-product
register. `BinaryPositionSpace` is the ordinary `N = 2^L` quantics basis. Other
position spaces specialize `physical_projector`, `physical_site_state`, `site_axis`,
and `site_permutation` after `TBHamiltonian` is defined.
"""
abstract type AbstractPositionSpace end

"""Ordinary binary position register containing all `2^L` basis states."""
struct BinaryPositionSpace <: AbstractPositionSpace end
