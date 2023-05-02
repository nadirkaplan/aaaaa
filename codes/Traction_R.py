import math as m
import numpy as np
from random import random
import pdb


'''
Backward Euler to update fraction/probablity of closed bond
'''
def closed_bond_prob(alpha, zeta, ks, kc, fa, fcr, fc, f, rho_n, dt):
    if fa > fcr:
        kon = alpha*np.exp(zeta*(fa-fcr))   # + f 
    else:
        kon = alpha 
    koff = ks * np.exp(f) + kc * np.exp(-f/fc)
    rho = (rho_n + kon*dt)/(1 + kon*dt + koff*dt)
    return (rho, kon, koff)


'''
Retrograde velocity calculation
'''
def Retrograde_velocity(F_st, F_stall, v0):
    vf = v0*(1-F_st/F_stall)
    if vf < float(0.): vf = 0.
    #if vf > v0:  vf=v0
    return vf


'''
Update displacements of the substrate (xs) and actin bundle (xc)
'''
def update_disps(xs0, xc0, rho_ary, nc, Kc, eta, Ka, Kl, Ks, Fstall, F_p, v0, dt, opt):
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
    r2 = v0*dt + xc0 + v0*dt*F_p/Fstall
    rside = np.array([r1, r2])
    xcs = np.linalg.solve(matrix, rside)
    return xcs


'''
calculate nucleus velocity based on traction force input and nucleus parameters
'''
def nucleus_velocity(pst_vx, pst_vy, Ftr, Force_ck, R_nuc, eta_nuc):
    Fx = (Ftr-Force_ck)*pst_vx;   Fy = (Ftr-Force_ck)*pst_vy;
    Fx_sum = np.sum(Fx);  Fy_sum = np.sum(Fy); 
    coef = 6*np.pi*R_nuc*eta_nuc*1e3;
    Vnuc_x = Fx_sum/coef;  Vnuc_y = Fy_sum/coef;  
    return (Vnuc_x, Vnuc_y)




def area_stiff(Area, kck0, kck1):
    aa = 0.00185;
    Kck = kck0 + kck1*np.exp(aa*Area);
    return Kck


def microtubule_force(k_ck0, k_ck1, Area1, Area2, Asoft, R_len1, R_len2):
    nlen = np.size(R_len1);  del_force_ck=np.zeros(nlen);
    energ_ck = np.zeros(nlen);
    for i in range(nlen):
        kck1 = area_stiff(Area1, k_ck0, k_ck1);
        kck2 = area_stiff(Area2, k_ck0, k_ck1);
        del_len = R_len1[i]-R_len2[i];
        del_force_ck[i] = del_len*(kck1+kck2)/2;
    #energ_ck[i]= 0.50*k_ck*stretch_change**2.0
    #ck_energy = np.sum(energ_ck)  , ck_energy
    return del_force_ck



'''
calculate angles between two stress fibers
'''
def calculate_angles(nm_pts_vx, nm_pts_vy):
    nlen = np.shape(nm_pts_vx)[0];   angles_between_fibers = np.zeros([nlen]);
    for i in range(nlen):
        angle1 = m.atan2(nm_pts_vy[i], nm_pts_vx[i]);
        if i < nlen -1:
            angle2 = m.atan2(nm_pts_vy[i+1], nm_pts_vx[i+1]);
        else:
            angle2 = m.atan2(nm_pts_vy[0], nm_pts_vx[0]);
        #print('angle1', angle1);      print('angle2', angle2);   
        del_angle = angle2 - angle1;  
        if del_angle < - np.pi:   del_angle = del_angle + 2*np.pi
        if del_angle > 1.5*np.pi:   del_angle = del_angle - 2*np.pi
        angles_between_fibers[i] = del_angle;
    return angles_between_fibers
    
    



'''
calculate membrane force and then get protrusion force based on equilibrium
<R> = [Rx, Ry] = [nm_pts_vx, nm_pts_vy] --- Direction vector
<R+> =[Ry, -Rx] = [nm_pts_vy, -nm_pts_vx] --- Orthogonal vector
'''
def membrane_protrusion_fv(vector_edge, k_mem, edge_L0, Force_ck, eta_mem, v0, Vs, Vp, Fsub, Fstall, nm_pts_vx, nm_pts_vy):
    nlen = np.shape(vector_edge)[0];  protrusion_force=np.zeros([nlen,2]); lavg = np.zeros([nlen]);   
    energ_mb = np.zeros([nlen]);   F_pro = np.zeros([nlen]);   vt = np.zeros([nlen]);   F_wall = np.zeros([nlen]);
    for i in range(nlen):
        ext_disp1 = (vector_edge[i,0]-edge_L0); #/edge_L0
        ext_disp2 = (vector_edge[i,3]-edge_L0);
        if ext_disp1<0: ext_disp1=0.0; 
        if ext_disp2<0: ext_disp2=0.0; 
        protrusion_force[i,0]= k_mem*(ext_disp1*vector_edge[i,1]+ext_disp2*vector_edge[i,4]); 
        protrusion_force[i,1]= k_mem*(ext_disp1*vector_edge[i,2]+ext_disp2*vector_edge[i,5]);
        lavg[i] = (vector_edge[i,0] + vector_edge[i,3])/2;
        energ_mb[i] = 0.50*k_mem*ext_disp2**2.0
        
    # membrane force in polar and tagent directions 
    Fmemb_nm = protrusion_force[:,0]*nm_pts_vx+protrusion_force[:,1]*nm_pts_vy 
    Fmemb_tg = protrusion_force[:,0]*nm_pts_vy-protrusion_force[:,1]*nm_pts_vx

    # protrusion forces in nucleus-membrane direction
    for i in range(nlen):
        F_pro[i] = (eta_mem*lavg[i]*(Vp[i]-v0*(1-Fsub[i]/Fstall[i]))-Fmemb_nm[i]-Force_ck[i])/(1+eta_mem*lavg[i]*v0/Fstall[i]) ;
        vt[i] = Fmemb_tg[i]/(eta_mem*lavg[i]);    
    memb_energy = np.sum(energ_mb)
    Vs_updated = Vp - v0*(1-(Fsub-F_pro)/Fstall)
    return (F_pro, vt, Vs_updated, Fmemb_nm, Fmemb_tg, memb_energy) 


def Protrusion_force_correction(Force_pro, Force_ck, Fstall, Fsub):
    nlen = np.shape(Force_ck)[0];  
    for i in range(nlen):
        Fst = Fsub[i]-Force_pro[i]
        if Force_pro[i]>0 and Fst<0:
           Force_ck[i]=Force_ck[i]+(Force_pro[i]-Fsub[i]); Force_pro[i]=Fsub[i]
        elif Force_pro[i]<0 and Fst>Fstall[i]:
           Force_ck[i]=Fstall[i]-Fst; Force_pro[i]=Fsub[i] - Fstall[i];
    return (Force_pro, Force_ck)


'''
Area conservation forces
'''
def Area_conservation_force(Aa, A0, K_Area, dvd_vec_all):
    if Aa>A0:
        F_Area = 0.
    else:
        F_Area = K_Area*(A0 - Aa) #/A0
    F_Area_vec = F_Area*dvd_vec_all
    #print('F_Area', F_Area)
    #print('dvd_vec_all', dvd_vec_all)
    return F_Area_vec

