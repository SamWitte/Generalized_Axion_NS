import numpy as np
from dispersion import *
from magneto_model import *

class Ray_Evolution(object):
    
    def __init__(self, model, init_x, init_k, Bnorm, Per, r_0, umaxFix=1e-1, m_ax=1e-6, init_dist=1e4):
        self.model = model(Bnorm, Per, r_0, umaxFix=umaxFix)
        # for time being assume B not rotating... and assume axes are already aligned (definitely not true)
        self.init_x = init_x
        self.init_k = init_k
        self.m_ax = m_ax
        self.init_dist = init_dist
        return
    
    def loop_steps(self, h_eta=0.5, h_x=1e-2, h_k=1e-8):
        stop = False
        x_V = self.init_x
        k_V = self.init_k
        eta_0 = 0.
        
        print x_V, k_V
        while not stop:
            dist = np.sqrt(np.dot(x_V, x_V))
            step_size = dist**h_eta
            x_V, k_V = self.solve_step(x_V, k_V, hh=step_size, h_x=h_x, h_k=h_k)
            
            eta_0 += step_size
            n_c = self.model.epm_den(x_V)
            omega_p = np.sum(np.sqrt(4.*np.pi/137. * n_c / 5.11e5 * (2.998e10 * 6.58e-16)**3))
            print x_V, k_V, omega_p, dist / self.model.r_0
            if omega_p > self.m_ax:
                print 'Found conversion surface... Plasma Freq {:.2e} eV, Axion Mass {:.2e}, Radius {:.2e} km'.format(omega_p, self.m_ax, np.sqrt(np.dot(x_V, x_V)))
                stop = True
            if dist > self.init_dist:
                print 'Did not encounter conversion radius...'
                stop = True
    
        return

    def solve_step(self, vec_x, vec_k, hh=0.5, h_x=1e-2, h_k=1e-8, dist=0.):
        # hh --> step size in [km]
        
        # ~~ probably need to apply rotation to vec_x, vec_k here to align B field with z direction...
        
        # RK step 1: k1 = f(k,x)
        
        rK1 = self.RK_eval(vec_x, vec_k, h_k, h_x)
        
        # RK step 2: k2 = f(k + step_size * k1_dk / 2, x + step_size * k1_dx / 2)
        vec_x_k2 = vec_x - hh * np.array([rK1[0], rK1[1], rK1[2]]) / 2.
        vec_k_k2 = vec_k - hh * np.array([rK1[3], rK1[4], rK1[5]]) / 2.
        rK2 = self.RK_eval(vec_x_k2, vec_k_k2, h_k, h_x)
        
        # RK step 3: k2 = f(k + step_size * k2_dk / 2, x + step_size * k2_dx / 2)
        vec_x_k3 = vec_x - hh * np.array([rK2[0], rK2[1], rK2[2]]) / 2.
        vec_k_k3 = vec_k - hh * np.array([rK2[3], rK2[4], rK2[5]]) / 2.
        rK3 = self.RK_eval(vec_x_k3, vec_k_k3, h_k, h_x)
    
        # RK step 4: k2 = f(k + step_size * k3_dk, x + step_size * k3_dx)
        vec_x_k4 = vec_x - hh * np.array([rK3[0], rK3[1], rK3[2]])
        vec_k_k4 = vec_k - hh * np.array([rK3[3], rK3[4], rK3[5]])
        rK4 = self.RK_eval(vec_x_k4, vec_k_k4, h_k, h_x)
    
#        print hh, np.array([rK1[0], rK1[1], rK1[2]]), np.array([rK2[0], rK2[1], rK2[2]]), np.array([rK3[0], rK3[1], rK3[2]]), np.array([rK4[0], rK4[1], rK4[2]])
#        exit()
        # Next Step
        x_Fin = vec_x - hh / 6. * np.asarray(rK4[:3])
        k_Fin = vec_k - hh / 6. * np.asarray(rK4[3:])
        return x_Fin, k_Fin
        
        
    def RK_eval(self, vec_x, vec_k, h_k, h_x):
        #print 'R1', vec_x
        B0 = self.model.B0(vec_x)
        n_dens = self.model.epm_den(vec_x)
        umaxVec = self.model.umax(vec_x)
        omega_std = dispersion_Eqns(vec_k, vec_x, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega()
        
        k1_domega_dkx = (dispersion_Eqns(vec_k + np.array([h_k, 0, 0]), vec_x, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega() - omega_std) / h_k
        k1_domega_dky = (dispersion_Eqns(vec_k + np.array([0, h_k, 0]), vec_x, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega() - omega_std) / h_k
        k1_domega_dkz = (dispersion_Eqns(vec_k + np.array([0, 0, h_k]), vec_x, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega() - omega_std) / h_k
        #print h_k, omega_std, dispersion_Eqns(vec_k + np.array([h_k, 0, 0]), vec_x, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega(), dispersion_Eqns(vec_k + np.array([0, h_k, 0]), vec_x, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega()
        

        #print 'R2'
        x_new = vec_x + np.array([h_x, 0, 0])
        B0 = self.model.B0(x_new)
        n_dens = self.model.epm_den(x_new)
        umaxVec = self.model.umax(x_new)
        k1_domega_dXx = -(dispersion_Eqns(vec_k, x_new, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega() - omega_std) / h_x
        
        #print 'R3'
        x_new = vec_x + np.array([0, h_x, 0])
        B0 = self.model.B0(x_new)
        n_dens = self.model.epm_den(x_new)
        umaxVec = self.model.umax(x_new)
        k1_domega_dXy = -(dispersion_Eqns(vec_k, x_new, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega() - omega_std) / h_x
        
        #print 'R4'
        x_new = vec_x + np.array([0, 0, h_x])
        B0 = self.model.B0(x_new)
        n_dens = self.model.epm_den(x_new)
        umaxVec = self.model.umax(x_new)
        k1_domega_dXz = -(dispersion_Eqns(vec_k, x_new, n_dens, B0, dist_func='Waterbag', umax=umaxVec).get_omega() - omega_std) / h_x
        
        return [k1_domega_dkx, k1_domega_dky, k1_domega_dkz, k1_domega_dXx, k1_domega_dXy, k1_domega_dXz]
        
    
