# Photon generation: isotropic directions and the scintillation emission time.

"Isotropic unit vector."
@inline function isotropic_dir(s::PhiloxStream)
    uz = 1f0 - 2f0 * randu(s)
    r = sqrt(max(0f0, 1f0 - uz * uz))
    phi = 2f0 * Float32(pi) * randu(s)
    return (r * cos(phi), r * sin(phi), uz)
end

"Scintillation emission delay: Exp(tau), or 0 for the delta (transport-only) profile."
@inline emission_time(s::PhiloxStream, tau_ns::Float32) =
    tau_ns > 0f0 ? -tau_ns * log(randu(s)) : 0f0
