NEURON {
    SUFFIX motoneuron
    USEION na READ ena WRITE ina
    USEION ks READ eks WRITE iks
    USEION kf READ ekf WRITE ikf
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
    gl = 0.0003 (nS/um2) <0,1e9>
    gna = 0.0003 (nS/um2) <0,1e9>
    gk_fast = 0.0003 (nS/um2) <0,1e9>
    gk_slow = 0.0003 (nS/um2) <0,1e9>    
    el = -70 (mV)
    tau_max_p = 4
    vt = -58
}

ASSIGNED {
    v (mV)
    ena (mV)
    eks (mV)
    ekf (mV)
    ina (mA/cm2)
    iks (mA/cm2)
    ikf (mA/cm2)
    il (mA/cm2)
}

STATE {
    m h n p
}

INITIAL {
    rates(v)
    m = 0
    h = 1
    n = 0
    p = pinf
}

? currents
BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gna*m*m*m*h*(v - ena)
    ikf = gk_fast*n*n*n*n*(v - ekf)
    iks = gk_slow*p*p*(v - eks)
    il = gl*(v - el)
    :printf("ina = %g", ina)
}

DERIVATIVE states {
    rates(v)
    m' = 0.1*alpha_m*(1-m) - 0.1*beta_m*m
    h' = 0.035*alpha_h*(1-h) - 0.035*beta_h*h
    n' = 0.035*alpha_n*(1-n) - 0.035*beta_n*n
    p' = (pinf - p) / ptau
}

PROCEDURE rates(v(mV)) {
    : Canais de Sodio
    alpha_m = (-0.32*(v-vt-13))/(exp(-(v-vt-13)/4)-1)
    beta_m = 0.28*(v-vt-40)/(exp((v-vt-40)/5)-1)
    alpha_h = 0.128*exp(-(v-vt-17)/18)
    beta_h = 4/(1+exp(-(v-vt-40)/5))

    : Canais de Potassio Rapido
    alpha_n = (-0.032*(v-vt-15))/(exp(-(v-vt-15)/5)-1)
    beta_n = 0.5*exp(-(v-vt-10)/40)

    : Canais de Potassio Lento
    pinf = 1/(1+exp(-(v+35)/10))
    ptau = tau_max_p/(3.3*exp((v+35)/20)+exp(-(v+35)/20))
}