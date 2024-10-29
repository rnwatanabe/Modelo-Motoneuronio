NEURON {
    POINT_PROCESS muscle_unit_calcium
    RANGE Tc, Fmax, spike, F
	RANGE k1, k2, k3, k4, k5, k6, k, k5i, k6i
	RANGE Umax, Rmax, tau1, tau2, R, R1, R2
	RANGE phi0, phi1, phi2, phi3, phi4
	RANGE AMinf, AMtau, SF_AM, T0
	RANGE c1, c2, c3, c4, c5
	:RANGE acm, alpha, alpha1, alpha2, alpha3, beta, gamma
}


PARAMETER{
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

	:: Muscle activation::
	c1 = 0.128
	c2 = 0.093
	c3 = 61.206
	c4 = -13.116
	c5 = 5.095
	:alpha = 2
	:alpha1 = 4.77
	:alpha2 = 400
	:alpha3 = 160
	:beta = 0.47
	:gamma = 0.001
}

ASSIGNED{
    spike
    F
	k5
	k6
	AMinf
	AMtau
	:xm_temp1
	:xm_temp2
	:vm
	:acm
}

STATE{
    x1 x2 CaSR CaT AM CaSRCS Ca CaB
	: xm
}

INITIAL{
    x1 = 0
    x2 = 0
    spike = 0
	CaSR = 0.0025  		:[M]
	CaSRCS = 0		    :[M]
	Ca = 1e-10		    :[M]
	CaT = 0				:[M]
	AM = 0				:[M]
	CaB = 0				:[M]
}


BREAKPOINT{
	R = CaSR*Rmax*(exp(-t/tau2)*R1 - exp(-t/tau1-t/tau2)*R2)
	printf("R = %g", R)
	rate (CaT, AM, t)
    SOLVE states_force METHOD derivimplicit
    F = Fmax*x1
}

DERIVATIVE states_force{
    :x1' = x2
    :x2' = -2/Tc*x2 - 1/(Tc*Tc)*x1 + spike/Tc    
	spike = 0
	CaSR' = -k1*CS0*CaSR + (k1*CaSR+k2)*CaSRCS - R + U(Ca)
	: printf("CaSR = %g", CaSR)
	CaSRCS' = k1*CS0*CaSR - (k1*CaSR+k2)*CaSRCS
	:Ca' = - k5*T0*Ca + (k5*Ca+k6)*CaT - k3*B0*Ca + (k3*Ca+k4)*CaB + R - U(Ca)
	:CaB' = k3*B0*Ca - (k3*Ca+k4)*CaB
	:CaT' = k5*T0*Ca - (k5*Ca+k6)*CaT
	:AM' = (AMinf -AM)/AMtau
	
}

PROCEDURE rate(CaT (M), AM (M), t(ms)) {
	k5 = phi(-8)*k5i
	k6 = k6i/(1 + SF_AM*AM)
	AMinf = 0.5*(1+tanh(((CaT/T0)-c1)/c2))
	AMtau = c3/(cosh(((CaT/T0)-c4)/(2*c5)))
}

FUNCTION U(x) {
	if (x >= 0) {U = Umax*(x^2*k^2/(1+x*k+x^2*k^2))^2}
	else {U = 0}
}

FUNCTION phi(x) {
	if (x <= -8) {phi = phi1*x + phi2}
	else {phi = phi3*x + phi4}
}

NET_RECEIVE (weight) {
	spike = 2.7182818/dt
    R1 = R1 + exp(t/tau2)
 	R2 = R2 + exp(t/tau1+t/tau2)
	:printf("R1 = %g", R1)
	:printf("R2 = %g", R2)
}

