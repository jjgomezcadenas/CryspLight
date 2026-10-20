# The crystal is a single axis-aligned box [0,Lx]×[0,Ly]×[0,Lz], the SiPM plane at z = Lz.
# Faces are coded so the transport can branch on wrapped vs readout.

"Face code (Int32 1) of the wrapped face x = Lx of the crystal box."
const FACE_XP = Int32(1)   # x = Lx  (wrapped)
"Face code (Int32 2) of the wrapped face x = 0 of the crystal box."
const FACE_XM = Int32(2)   # x = 0   (wrapped)
"Face code (Int32 3) of the wrapped face y = Ly of the crystal box."
const FACE_YP = Int32(3)   # y = Ly  (wrapped)
"Face code (Int32 4) of the wrapped face y = 0 of the crystal box."
const FACE_YM = Int32(4)   # y = 0   (wrapped)
"Face code (Int32 5) of the readout face z = Lz, where the SiPM plane sits."
const FACE_BACK  = Int32(5) # z = Lz (SiPM plane)
"""
Face code (Int32 6) of the face z = 0, opposite the SiPMs: wrapped, or a second readout
face in a two-sided readout.
"""
const FACE_FRONT = Int32(6) # z = 0  (wrapped)

"""
    Box(L::NTuple{3,Float32})

The crystal: the axis-aligned box [0, Lx] x [0, Ly] x [0, Lz], with the sides L in mm.
The SiPM plane is the face z = Lz (FACE_BACK); the face z = 0 (FACE_FRONT) is the one
opposite the SiPMs.
"""
struct Box
    L::NTuple{3,Float32}
end

"""
    wall_hit(x, y, z, ux, uy, uz, box::Box) -> (d::Float32, face::Int32)

Distance d in mm along the unit direction (ux, uy, uz) from the point (x, y, z) inside
the box (mm, Float32) to the first wall, and the code of the face hit (FACE_XP to
FACE_FRONT). Closed-form axis-aligned slab test.
"""
@inline function wall_hit(x::Float32, y::Float32, z::Float32,
                          ux::Float32, uy::Float32, uz::Float32, box::Box)
    Lx, Ly, Lz = box.L
    dx = ux > 0f0 ? (Lx - x) / ux : (ux < 0f0 ? -x / ux : Inf32)
    dy = uy > 0f0 ? (Ly - y) / uy : (uy < 0f0 ? -y / uy : Inf32)
    dz = uz > 0f0 ? (Lz - z) / uz : (uz < 0f0 ? -z / uz : Inf32)
    if dx <= dy && dx <= dz
        return dx, (ux > 0f0 ? FACE_XP : FACE_XM)
    elseif dy <= dz
        return dy, (uy > 0f0 ? FACE_YP : FACE_YM)
    else
        return dz, (uz > 0f0 ? FACE_BACK : FACE_FRONT)
    end
end
