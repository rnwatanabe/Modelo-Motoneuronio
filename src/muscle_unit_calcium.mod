NEURON {
    POINT_PROCESS muscle_unit
    RANGE Tc, A, spike, F
}


PARAMETER {
    Tc = 100
    Fmax = 1
    :: Calcium dynamics ::
	k1 = 3000		: M-1*ms-1
	k2 = 3			: ms-1
	k3 = 400		: M-1*ms-1
	k4 = 1			: ms-1
	k5i = 4e5		: M-1*ms-1
	k6i = 150		: ms-1
	k = 850			: M-1
	SF_AM = 5
	Rmax = 10		: ms-1
	Umax = 2000		: M-1*ms-1
	tau1 = 3			: ms
	tau2 = 25			: ms
	phi1 = 0.03
	phi2 = 1.23
	phi3 = 0.01
	phi4 = 1.08
	CS0 = 0.03     	:[M]
	B0 = 0.00043	:[M]
	T0 = 0.00007 	:[M]
    
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
    F = Fmax*x1
}

DERIVATIVE states {
    x1' = x2
    x2' = -2/Tc*x2-1/(Tc*Tc)*x1+spike/Tc
    spike = 0
}

NET_RECEIVE (weight) {
	spike = 2.7182818/dt
}

