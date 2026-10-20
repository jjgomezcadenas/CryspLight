# Photon generation: isotropic directions and the scintillation emission time.

"""
    isotropic_dir(s::PhiloxStream) -> (ux, uy, uz)

Isotropic unit vector (Float32 components) from two draws of the stream: the cosine of
the polar angle, then the azimuth.
"""
@inline function isotropic_dir(s::PhiloxStream)
    uz = 1f0 - 2f0 * randu(s)
    r = sqrt(max(0f0, 1f0 - uz * uz))
    phi = 2f0 * Float32(pi) * randu(s)
    return (r * cos(phi), r * sin(phi), uz)
end

"Scintillation emission delay: Exp(tau), or 0 for the delta (transport-only) profile."
@inline emission_time(s::PhiloxStream, tau_ns::Float32) =
    tau_ns > 0f0 ? -tau_ns * log(randu(s)) : 0f0

"""
Two-component scintillation delay: with probability frac1 Exp(tau1), otherwise Exp(tau2).
tau2 = 0 means a single component and draws exactly as emission_time(s, tau1).
"""
@inline function emission_time(s::PhiloxStream, tau1_ns::Float32, tau2_ns::Float32,
                               frac1::Float32)
    tau2_ns > 0f0 || return emission_time(s, tau1_ns)
    tau = randu(s) <= frac1 ? tau1_ns : tau2_ns
    return -tau * log(randu(s))
end
