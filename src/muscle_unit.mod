NEURON {
    ARTIFICIAL_CELL muscle_unit
    POINTER spike
    RANGE Tc, A
}


PARAMETER {
    Tc = 0.9
    A = 1
}

ASSIGNED {
    spike
    F
}

STATE {
    F1 dF1
}

INITIAL {
    F1 = 0
    dF1 = 0
    spike = 0
}


BREAKPOINT {
    SOLVE states METHOD cnexp
    F = A*F1
}

DERIVATIVE states {
    dF1' = -2/Tc*dF1-1/(Tc*Tc)*F1+spike/Tc
    F1' = dF1
    spike = 0
}

NET_RECEIVE (weight) {
	spike = 1/dt
}

