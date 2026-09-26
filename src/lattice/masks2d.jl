# masks2d.jl — diagonal mask MPOs for 2D lattice geometries: the single-qubit
# projectors sigma_d/sigma_u and the row-break, row/column-select and
# checkerboard masks used by the hopping builders (hopping2d.jl) and the
# sublattice presets (sublattice.jl). Split from lattice/2Dlattice_tk.jl.

# ============================================================
# 1. Single-qubit projectors
# ============================================================

ITensors.op(::OpName"sigma_d",::SiteType"Qubit") = [0 0; 0 1]   # |1><1|
ITensors.op(::OpName"sigma_u",::SiteType"Qubit") = [1 0; 0 0]   # |0><0|


# ============================================================
# 2. Row/column/checkerboard mask MPOs (diagonal, exact)
#    Bit layout: sites 1..Ly → iy (MSB first), sites Ly+1..L → ix (MSB first)
# ============================================================

"""
    _row_break_mpo(Lx, Ly, sites; which) -> MPO

Diagonal mask that zeroes wrap-around couplings at row boundaries of a
`2^Lx × 2^Ly` grid (row-major encoding).

- `which = :xplus`  → 0 where ix == 2^Lx − 1  (end of each row)
- `which = :xplain` → 0 where ix == 0           (start of each row)

Multiply a kinetic MPO by this mask on the appropriate side to suppress the
bond that crosses a row boundary.
"""
function _row_break_mpo(Lx, Ly, sites; which::Symbol)
    L     = Lx + Ly
    Id_op = MPO(sites, "Id")
    proj  = which === :xplus  ? "sigma_d" :
            which === :xplain ? "sigma_u" :
            error("unknown which=:$(which); use :xplus or :xplain")
    # projector onto ix = Nx-1 (:xplus) or ix = 0 (:xplain):
    # product of proj on all Lx x-bit sites (Ly+1 .. L)
    os = OpSum()
    os += 1, proj, Ly+1
    for i in Ly+2:L; os *= 1, proj, i; end
    return Id_op - MPO(os, sites)
end


"""
    _row_select_mpo(_, Ly, sites; keep=:even) -> MPO

Diagonal mask that retains only even or odd rows of a `2^Lx × 2^Ly` grid.

- `keep = :even` → 1 where iy % 2 == 1  (0-based; LSB of iy = 1)
- `keep = :odd`  → 1 where iy % 2 == 0  (0-based; LSB of iy = 0)
"""
function _row_select_mpo(::Any, Ly, sites; keep::Symbol = :even)
    proj = keep === :even ? "sigma_d" :
           keep === :odd  ? "sigma_u" :
           error("unknown keep=:$(keep); use :even or :odd")
    os = OpSum()
    os += 1, proj, Ly    # site Ly is the LSB of iy
    return MPO(os, sites)
end


"""
    _col_select_mpo(Lx, Ly, sites; keep=:even) -> MPO

Diagonal mask that retains only even or odd columns of a `2^Lx × 2^Ly` grid.

- `keep = :even` → 1 where ix % 2 == 1  (0-based; LSB of ix = 1)
- `keep = :odd`  → 1 where ix % 2 == 0  (0-based; LSB of ix = 0)
"""
function _col_select_mpo(Lx, Ly, sites; keep::Symbol = :even)
    proj = keep === :even ? "sigma_d" :
           keep === :odd  ? "sigma_u" :
           error("unknown keep=:$(keep); use :even or :odd")
    os = OpSum()
    os += 1, proj, Lx + Ly   # site L = Lx+Ly is the LSB of ix
    return MPO(os, sites)
end


"""
    _row_checker_mpo(Lx, Ly, sites) -> MPO

Diagonal checkerboard mask: 1 where (ix + iy) is even, 0 otherwise.
Equivalent to projecting onto LSB(ix) == LSB(iy), i.e. both qubits agree:
  proj_{iy-LSB=0, ix-LSB=0}  +  proj_{iy-LSB=1, ix-LSB=1}
"""
function _row_checker_mpo(Lx, Ly, sites)
    os = OpSum()
    os += 1, "sigma_u", Ly, "sigma_u", Lx + Ly   # both LSBs = 0
    os += 1, "sigma_d", Ly, "sigma_d", Lx + Ly   # both LSBs = 1
    return MPO(os, sites)
end
