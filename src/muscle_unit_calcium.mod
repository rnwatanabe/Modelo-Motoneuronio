NEURON {
    POINT_PROCESS muscle_unit_calcium
    RANGE Tc, A, spike, F
	RANGE k1, k2, k3, k4, k5, k6, k, k5i, k6i
	RANGE Umax, Rmax, tau1, tau2, R
	RANGE AMinf, AMtau, SF_AM

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
    R = 0
	R1 = 0
	R2 = 0 
}

ASSIGNED {
    spike
    F
	R
	k5
	k6
	AMinf
	AMtau
	spike
	xm_temp1
	xm_temp2
	vm
	acm
}

STATE {
    x1 x2 CaSR  CaSRCS Ca CaB CaT xm
}

INITIAL {
    x1 = 0
    x2 = 0
    spike = 0
}


BREAKPOINT {
	R = CaSR*Rmax*(exp(-t/tau2)*R1-exp(-t/tau1-t/tau2)*R2)
	
    SOLVE states METHOD cnexp
    F = Fmax*x1
}

DERIVATIVE states {
    x1' = x2
    x2' = -2/Tc*x2-1/(Tc*Tc)*x1+spike/Tc
    spike = 0
	rate (CaT, AM, t)
	CaSR' = -k1*CS0*CaSR + (k1*CaSR+k2)*CaSRCS - R + U(Ca)
	CaSRCS' = k1*CS0*CaSR - (k1*CaSR+k2)*CaSRCS
	Ca' = - k5*T0*Ca + (k5*Ca+k6)*CaT - k3*B0*Ca + (k3*Ca+k4)*CaB + R - U(Ca)
	CaB' = k3*B0*Ca - (k3*Ca+k4)*CaB
	CaT' = k5*T0*Ca - (k5*Ca+k6)*CaT

}

PROCEDURE rate (CaT (M), AM (M), t(ms)) {
	k5 = phi(-8)*k5i
	k6 = k6i/(1 + SF_AM*AM)
	AMinf = 0.5*(1+tanh(((CaT/T0)-c1)/c2))
	AMtau = c3/(cosh(((CaT/T0)-c4)/(2*c5)))
}

NET_RECEIVE (weight) {
	spike = 2.7182818/dt
	R1 = R1 + exp(t/tau2)
	R2 = R2 + exp(t/tau1+t/tau2)
}

