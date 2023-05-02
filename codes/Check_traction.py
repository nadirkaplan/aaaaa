'''
Molecular clutch model with standard-linear-solid substrate 
'''

import math as m
import numpy as np
from random import random
import pdb


'''
Backward Euler to update fraction/probablity of closed bond
'''
def closed_bond_prob(alpha, epsi, ks, kc, fc, f, rho_n, dt):
    kon = alpha +  epsi*f
    koff = ks * np.exp(f) + kc * np.exp(-f/fc)
    rho = (rho_n + kon*dt)/(1 + kon*dt + koff*dt)
    return (rho, kon, koff)


'''
Update displacements of the substrate (xs) and actin bundle (xc)
'''
def update_disps(xs0, xc0, rho_ary, nc, Kc, eta, Ka, Kl, Ks, Fstall, Fck, v0, dt, opt):
    rho = rho_ary[0];
    kon = rho_ary[1];
    koff = rho_ary[2];
    if opt == 0:
        m11 = Kc*nc*rho;
        m12 = - (m11 + Ks);
        r1 = 0;
    else:
        m11 = -(Ka*Kc*nc*rho*dt + eta*Kc*nc*rho + eta*Kc*nc*dt*((1-rho)*kon - rho*koff)) 
        m12 = ( (Ka + Kl)*eta + Ka*Kl*dt + Ka*Kc*nc*rho*dt + eta*Kc*nc*rho + eta*Kc*nc*dt*((1-rho)*kon - rho*koff))
        r1 = (Ka + Kl)*eta*xs0 - eta*Kc*nc*rho*(xc0 - xs0)        
    m21 = 1 + v0*dt*Kc*nc*rho/Fstall
    m22 = - v0*dt*Kc*nc*rho/Fstall
    matrix = np.array([[m11, m12], [m21, m22]])
    r2 = v0*dt + xc0 + v0*dt*Fck/Fstall
    rside = np.array([r1, r2])
    #print(matrix)
    xcs = np.linalg.solve(matrix, rside)
    return xcs


'''
calculate reinforcement force 'fa' 
'''
def check_faval(alpha, epsi, ks, kc, fc, nc, nm, fm, v0, opt, Ka, Kl, Ks, K_clutch, eta, F_p):
    if opt == 0:
        T = min(20*200/120/Ks, 50); 
    else:
        T = min(20*200/120/Kl, 50);  
    Nstep = int(T*1000)
    dt = T/Nstep

    Fstall = nm*fm

    xc = np.zeros([Nstep])
    xs = np.zeros([Nstep])
    tarray = np.zeros([Nstep])
    farray = np.zeros([Nstep])
    Ro_ary = np.zeros([Nstep])

    xc0 = 0.
    xs0 = 0.
    rho_0 = 0.
    f = 0.
    fa = 0

    tm = 0.0
    step = 0
    
    while step<Nstep:
        
        Pf_iter = 0;  fdiff = 1.0;   ro_diff = 1.0;   ff_0 = 0;   rro_0 = 0;  Pf_imax = 3;
        
        while (abs(fdiff) > 1e-6  or   abs(ro_diff) > 1e-6) and Pf_iter < Pf_imax:
        
            rho_ary = closed_bond_prob(alpha, epsi, ks, kc, fc, f, rho_0, dt) ;
            xc_xs = update_disps(xs0, xc0, rho_ary, nc, K_clutch, eta, Ka, Kl, Ks, Fstall, F_p, v0, dt, opt)
            f = K_clutch*(xc_xs[0] - xc_xs[1]) ;
            rho = rho_ary[0];

            fdiff = f - ff_0 ;   ro_diff = rho - rro_0;
            
            ff_0 = f;    rro_0 = rho;
            
            Pf_iter += 1;
            
            if rho<1e-8  or f<0 or f>100: f=0;  rho = 0.;  xc0 = 0.;  xs0 = 0.;  Pf_iter = Pf_imax*2;
            else:    xc0 = xc_xs[0]; xs0 = xc_xs[1];

           
        xc[step] = xc0;  xs[step] = xs0;  
        rho_0 = rho;
        Ro_ary[step] = rho
        tarray[step] = tm; 
        farray[step] = f;  
        step += 1
        tm += dt

    circle_num = np.where(abs(farray)<1e-16)[0];
    #circle_num = find_peaks(1.0-Ro_ary)[0]
    avg_nn = int(np.size(circle_num));
    if avg_nn >0:
        avg_step_num = circle_num[avg_nn-1];
        fa = np.mean(farray[:avg_step_num]);
    else:
        fa = 0.0;

    return fa   
    
