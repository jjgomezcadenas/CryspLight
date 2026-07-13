# The crystal is a single axis-aligned box [0,Lx]×[0,Ly]×[0,Lz], the SiPM plane at z = Lz.
# Faces are coded so the transport can branch on wrapped vs readout.

const FACE_XP = Int32(1)   # x = Lx  (wrapped)
const FACE_XM = Int32(2)   # x = 0   (wrapped)
const FACE_YP = Int32(3)   # y = Ly  (wrapped)
const FACE_YM = Int32(4)   # y = 0   (wrapped)
const FACE_BACK  = Int32(5) # z = Lz (SiPM plane)
const FACE_FRONT = Int32(6) # z = 0  (wrapped)

struct Box
    L::NTuple{3,Float32}
end

"""
Distance along (ux,uy,uz) from (x,y,z) inside the box to the first wall, and the face hit.
Closed-form axis-aligned slab test.
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
