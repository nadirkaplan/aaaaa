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
def bond_rates(alpha, eps, ks, kc, f, fc):
    kon = alpha 
    koff = ks * np.exp(f) + kc * np.exp(-f/fc)
    return (kon, koff)



'''
def Jacobian matrix
''' 
def Jacobian_res(alpha, eps, kr0, kc0, fc, ro, xr, xs, ro0, xr0, xs0, nc, Kc, eta, Ka, Kl, Fstall, Fck, v0, dt):
    f = Kc*(xr - xs);
    [kon, koff]  = bond_rates(alpha, eps, kr0, kc0, f, fc)
    partial1_ro = 1 + kon*dt + koff*dt
    partial1_xr = (kr0 * np.exp(f) * Kc - kc0 * np.exp(-f/fc) * Kc/fc)*dt*ro
    partial1_xs = (-kr0 * np.exp(f) * Kc + kc0 * np.exp(-f/fc) * Kc/fc)*dt*ro

    partial2_ro = (Ka*nc*Kc*dt - eta*Kc*nc*dt*(kon + koff) + eta*nc*Kc)*(xs-xr)-eta*nc*Kc*(xs0 - xr0)
    partial2_xr = -(Ka*Kc*nc*ro*dt + eta*Kc*nc*ro + eta*Kc*nc*dt*((1-ro)*kon - ro*koff))
    partial2_xs = ( (Ka + Kl)*eta + Ka*Kl*dt + Ka*Kc*nc*ro*dt + eta*Kc*nc*ro + eta*Kc*nc*dt*((1-ro)*kon - ro*koff))
    

    partial3_ro = v0*dt*nc*Kc*(xr-xs)
    partial3_xr = Fstall + v0*dt*nc*Kc*ro ;
    partial3_xs = - v0*dt*nc*Kc*ro

    Jacobian_matrix = np.array([[partial1_ro, partial1_xr, partial1_xs],  \
                    [partial2_ro, partial2_xr, partial2_xs], [partial3_ro, partial3_xr, partial3_xs]]);

    residue1 = kon*dt + ro0 - (1 + kon*dt + koff*dt)*ro;
    residue2 = (Ka + Kl)*eta*xs0 - eta*Kc*nc*ro*(xr0 - xs0) - ((Ka+Kl)*eta + Ka*Kl*dt)*xs  \
             + (Ka*nc*ro*dt*Kc + eta*Kc*nc*dt*((1-ro)*kon - ro*koff) + eta*nc*ro*Kc)*(xr-xs);
    
    residue3 = v0*dt*Fstall + xr0*Fstall + v0*dt*Fck - Fstall*xr - v0*dt*nc*ro*Kc*(xr - xs);

    res_vector = np.array([residue1, residue2, residue3]);

    #pdb.set_trace()
    
    xrs = np.linalg.solve(Jacobian_matrix, res_vector)

    return xrs



'''
calculate reinforcement force 'fa' using monolithic scheme
'''
def check_faval_newton(alpha, epsi, ks, kc, fc, nc, nm, fm, v0, K_a, K_l, K_c, eta, F_p):
                     
    T = min(20*200/120/K_l, 50);  
    Nstep = int(T*5000)
    dt = T/Nstep
    

    F_stall = nm*fm

    xr_array = np.zeros([Nstep])
    xs_array = np.zeros([Nstep])
    tarray = np.zeros([Nstep])
    farray = np.zeros([Nstep])
    Ro_array = np.zeros([Nstep])
    Fsub_array = np.zeros([Nstep])

    xr0 = 0.
    xs0 = 0.
    rho_0 = 0.
    rho = 0.
    xr = 0.
    xs = 0.
    f = 0.
    fa = 0

    tm = 0.0
    step = 0

    while step<Nstep-1:
        Pf_iter = 0;  tol_check = 0.1
        while abs(tol_check)>1e-12 and Pf_iter < 10:
            del_ro_xr_xs = Jacobian_res(alpha, epsi, ks, kc,  fc, rho, xr, xs, rho_0, xr0, xs0, nc, K_c, eta, K_a, K_l, F_stall, F_p, v0, dt);
            drho = del_ro_xr_xs[0];
            dxr = del_ro_xr_xs[1];
            dxs = del_ro_xr_xs[2];
            rho += drho;
            xr += dxr;
            xs += dxs;
            tol_check = dxr*dxs;
            Pf_iter += 1;

            f = K_c*(xr - xs) ;

            if rho<1e-9/K_l  or f > 100 or f< 0:    #if rho<1e-2 and f>10:  #f>200 or fdiff > 1:
                f=0;       rho = 0.; 
                xr = 0.;  xs = 0.;
                Pf_iter = 200
            #else:
            #    xr0 = xr;  xs0 = xs;
            
        xr0 = xr;  xs0 = xs;
        tm += dt;
        step += 1;
        #pdb.set_trace()
        xr_array[step] = xr0;  xs_array[step] = xs0;  
        Ro_array[step] = rho;   rho_0 = Ro_array[step];           
        tarray[step] = tm; 
        farray[step] = f;
        Fsb = rho*nc*f;   Fsub_array[step] = Fsb; 
        

    circle_num = np.where(abs(farray)<1e-16);
    avg_nn = int(np.size(circle_num));
    if avg_nn > 0:
        avg_step_num = circle_num[0][avg_nn-1];
        fa = np.mean(farray[:avg_step_num]);
    else:
        fa = 0.0;

        
    return fa   
    



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
def update_disps(xs0, xc0, rho_ary, nc, Kc, eta, Ka, Kl, Fstall, Fck, v0, dt):
    rho = rho_ary[0];
    kon = rho_ary[1];
    koff = rho_ary[2];
    m11 = -(Ka*Kc*nc*rho*dt + eta*Kc*nc*rho + eta*Kc*nc*dt*((1-rho)*kon - rho*koff)) 
    m12 = ( (Ka + Kl)*eta + Ka*Kl*dt + Ka*Kc*nc*rho*dt + eta*Kc*nc*rho + eta*Kc*nc*dt*((1-rho)*kon - rho*koff))
    r1 = (Ka + Kl)*eta*xs0 - eta*Kc*nc*rho*(xc0 - xs0)        
    m21 = 1 + v0*dt*Kc*nc*rho/Fstall
    m22 = - v0*dt*Kc*nc*rho/Fstall
    matrix = np.array([[m11, m12], [m21, m22]])
    r2 = v0*dt + xc0 + v0*dt*Fck/Fstall
    rside = np.array([r1, r2])
    xcs = np.linalg.solve(matrix, rside)
    return xcs


'''
calculate reinforcement force 'fa' using staggered scheme
'''
def check_faval(alpha, epsi, ks, kc, fc, nc, nm, fm, v0, Ka, Kl, K_clutch, eta, F_p):
    T = min(20*200/120/Kl, 50);  
    Nstep = int(T*1000)
    dt = T/Nstep

    Fstall = nm*fm

    xc = np.zeros([Nstep])
    xs = np.zeros([Nstep])
    tarray = np.zeros([Nstep])
    farray = np.zeros([Nstep])
    Ro_ary = np.zeros([Nstep])
    Fsub_array = np.zeros([Nstep])

    xc0 = 0.
    xs0 = 0.
    rho_0 = 0.
    f = 0.
    fa = 0

    tm = 0.0
    step = 0
    
    while step<Nstep:
        
        #Pf_iter = 0;  fdiff = 1.0;   ro_diff = 1.0;   ff_0 = 0;   rro_0 = 0;  Pf_imax = 1;
        #while (abs(fdiff) > 1e-6  or   abs(ro_diff) > 1e-6) and Pf_iter < Pf_imax:
        rho_ary = closed_bond_prob(alpha, epsi, ks, kc, fc, f, rho_0, dt) ;
        xc_xs = update_disps(xs0, xc0, rho_ary, nc, K_clutch, eta, Ka, Kl, Fstall, F_p, v0, dt)
        f = K_clutch*(xc_xs[0] - xc_xs[1]) ;
        rho = rho_ary[0];

        #fdiff = f - ff_0 ;   ro_diff = rho - rro_0;
            
        #ff_0 = f;    rro_0 = rho;
          
        #Pf_iter += 1;
          
        if rho<1e-8/Kl or f<0 or f>100: f=0;  rho = 0.;  xc_xs[0]=0.; xc_xs[1]=0.;   # Pf_iter = Pf_imax*2;  
        

        xc0 = xc_xs[0];  xs0 = xc_xs[1];
        xc[step] = xc0;  xs[step] = xs0;  
        rho_0 = rho;
        Ro_ary[step] = rho
        tarray[step] = tm; 
        farray[step] = f;
        Fsb = rho*nc*f;   Fsub_array[step] = Fsb; 
        step += 1
        tm += dt
        #if f<1e-16: break;

    #pdb.set_trace()
    circle_num = np.where(abs(farray)<1e-16);
    avg_nn = int(np.size(circle_num));
    if avg_nn > 0:
        avg_step_num = circle_num[0][avg_nn-1];
        fa = np.mean(farray[:avg_step_num]);
    else:
        fa = 0.0;


    return fa   
    
