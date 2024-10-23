NEURON {
    POINT_PROCESS muscle_unit
    RANGE Tc, A, spike, F
}


PARAMETER {
    Tc = 100
    A = 1
}

ASSIGNED {
    spike
    F
}

STATE {
    x1 x2
}

INITIAL {
    x1 = 0
    x2 = 0
    spike = 0
}


BREAKPOINT {
    SOLVE states METHOD cnexp
    F = A*x1
}

DERIVATIVE states {
    x1' = x2
    x2' = -2/Tc*x2-1/(Tc*Tc)*x1+spike/Tc
    spike = 0
}

NET_RECEIVE (weight) {
	spike = 2.7182818/dt
}

