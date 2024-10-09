NEURON {
    POINT_PROCESS muscle_unit
    POINTER spike
    RANGE Tc, A
}


PARAMETER {
    Tc = 0.9
    A = 1
}

ASSIGNED {
    spike
}

STATE {
    F dF
}

INITIAL {
    F = 0
    dF = 0
    spike = 0
}


BREAKPOINT {
    SOLVE states METHOD cnexp
    F = A*F
}

DERIVATIVE states {
    dF' = -2/Tc*dF-1/(Tc*Tc)*F+spike/Tc
    F' = dF
    spike = 0
}

NET_RECEIVE (weight) {
	spike = 1/dt
}

