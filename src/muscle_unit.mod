NEURON {
    SUFFIX motoneuron
    NONSPECIFIC_CURRENT il
    RANGE gna, gk_fast, gk_slow, gl, vt, el
    GLOBAL alpha_m, alpha_h, alpha_n, pinf, beta_m, beta_h, beta_n, ptau
    THREADSAFE : assigned GLOBALs will be per thread

}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    Tc = 0.9 (nS/um2) <0,1e9>
    A = 1
}

ASSIGNED {
    spike
}

STATE {
    F, dF
}

INITIAL {
    F = 0
    dF = 0
    spike = 0
}

? currents
BREAKPOINT {
    SOLVE states METHOD cnexp   
}

DERIVATIVE states {
    dF' = -2/Tc*dF-1/(Tc*Tc)*F+spike/Tc
    F' = dF
    spike = 0
}

NET_RECEIVE (weight) {
	spike = 1/dt
}

