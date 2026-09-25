# aux_projection.jl — Auxiliary-index projection utilities
#
# Contains project_aux (and its Nothing overload), _autoenable_proj and
# aux_site, used by the get_bands projection pipeline.  Moved verbatim from
# section 5b of physics/QFT_tk.jl; the overview and file map of physics/qft/
# are at the top of bands.jl.  Slated to move to core/AuxDOF.jl (Tier 1 step 4).

# ============================================================
# 5b. Auxiliary index projection utilities
#
# Any auxiliary DOF (spin, Nambu, layer, sublattice) added with prepend_op /
# postpend_op lives at the first or last site of the MPO as a dim-1-bonded
# tensor.  The functions below implement the removal step used in Steps 0–1c
# of the get_bands projection pipeline.
#
# project_aux(W, aux_s, sec; side)
#   Contracts the projector |sec⟩⟨sec| onto the bra (aux_s') and ket (aux_s)
#   physical indices of the aux tensor.  The resulting dim-1 link is absorbed
#   into the adjacent position site, returning an (L−1)-site MPO.
#   `side=:pre` for prepended indices (spin, Nambu, layer);
#   `side=:post` for postpended indices (sublattice).
#
# aux_site(H, which) -> (Index, Symbol)
#   Extracts the auxiliary Index and its side (:pre or :post) from H.sites.
#   `which` ∈ :spin, :nambu, :layer, :sublattice.
#   Used by the TBHamiltonian overload to auto-detect all auxiliary indices
#   and pass them to the low-level get_bands without user intervention.
# ============================================================

"""
    project_aux(W, aux_s, σ; side=:pre) -> MPO

Remove an auxiliary site from MPO `W` by projecting onto state `σ`.

- `side=:pre`  — aux site is at position 1 (prepended, e.g. spin).
- `side=:post` — aux site is at the last position (postpended, e.g. sublattice).

Contracts the projector |σ⟩⟨σ| on both bra and ket physical indices of the
aux tensor; the resulting dim-1 link is absorbed into the adjacent position
site.  Returns an (L−1)-site MPO suitable for `conjugate_by_qft`.
"""
function project_aux(W::MPO, aux_s::Index, σ::Integer; side::Symbol = :pre)
    L        = length(W)
    pos      = side === :pre ? 1 : L
    aux_proj = W[pos] * setelt(aux_s' => σ) * setelt(aux_s => σ)
    new_tensors = Vector{ITensor}(undef, L - 1)
    if side === :pre
        new_tensors[1] = W[2] * aux_proj
        for i in 2:L-1; new_tensors[i] = W[i+1]; end
    else  # :post
        for i in 1:L-2; new_tensors[i] = W[i]; end
        new_tensors[L-1] = W[L-1] * aux_proj
    end
    return MPO(new_tensors)
end

# Nothing-overloads: give Julia a compilable method when the Index is nothing,
# so branches in get_bands can be type-checked without a MethodError.
project_aux(::MPO, ::Nothing, ::Integer; side::Symbol=:pre) =
    error("sublat_proj=true requires sublat_s to be set (detected from H.sublattice_s)")


"""
    _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)
        -> (nambu_proj, spin_proj, layer_proj, sublat_proj)

Enable projection flags for any auxiliary DOF detected on `H`, printing one
info line per auto-enabled flag.  Called at the top of every `TBHamiltonian`
spectral method before any aux-index logic runs.
"""
function _autoenable_proj(H::TBHamiltonian,
                           nambu_proj::Bool, spin_proj::Bool,
                           layer_proj::Bool, sublat_proj::Bool)
    if !isnothing(H.nambu_s) && !nambu_proj
        println("Info: H.nambu_s detected; auto-enabling nambu_proj=true ",
                "(pass proj_nambu=1/2 to select particle/hole sector).")
        nambu_proj = true
    end
    if !isnothing(H.spin_s) && !spin_proj
        println("Info: H.spin_s detected; auto-enabling spin_proj=true ",
                "(pass proj_s=1/2 to select ↑/↓ sector).")
        spin_proj = true
    end
    if !isnothing(H.layer_s) && !layer_proj
        println("Info: H.layer_s detected; auto-enabling layer_proj=true ",
                "(pass proj_layer=k to select a layer).")
        layer_proj = true
    end
    if !isnothing(H.sublattice_s) && !sublat_proj
        println("Info: H.sublattice_s detected; auto-enabling sublat_proj=true ",
                "(pass proj_sl=k to select a sublattice).")
        sublat_proj = true
    end
    return nambu_proj, spin_proj, layer_proj, sublat_proj
end


"""
    aux_site(H, which) -> (Index, Symbol)

Return the auxiliary `Index` and its position side (`:pre` or `:post`) for
the named auxiliary degree of freedom in `H`.

`which` ∈ `:spin`, `:sublattice`, `:nambu`, `:layer`.

Useful for passing the correct arguments to `project_aux` without manually
inspecting `H.sites`.

```julia
s, side = aux_site(H, :sublattice)
W_A = project_aux(W, s, 1; side=side)   # sublattice-A channel
```
"""
function aux_site(H::TBHamiltonian, which::Symbol)
    s = which === :spin       ? H.spin_s        :
        which === :sublattice ? H.sublattice_s  :
        which === :nambu      ? H.nambu_s       :
        which === :layer      ? H.layer_s       :
        error("Unknown auxiliary type :$which.  Use :spin, :sublattice, :nambu, or :layer.")
    isnothing(s) && error("H has no $which auxiliary index.")
    pos  = findfirst(==(s), H.sites)
    isnothing(pos) && error("Auxiliary index not found in H.sites — this is a bug.")
    side = pos == 1             ? :pre  :
           pos == length(H.sites) ? :post :
           error("Auxiliary $which index found at interior position $pos (unsupported).")
    return s, side
end
