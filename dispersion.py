import numpy as np
import cmath
from scipy.integrate import quad
from scipy.optimize import fsolve

class dispersion_Eqns(object):
    
    def __init__(self, kvec, xvec, n_vec, B0, dist_func='Softbell', umax=[2, 2]):
        
        self.kvec = kvec # kvec wave
        self.xvec = xvec # position wave -- right now does nothing, but will be useful when n_vec, B0 and distr funcs are spatially dependent
        # assumed (x,y,z) coordinate system, and B along z direction
        # [e-, e+]
        self.n_vec = n_vec * (2.998e10 * 6.58e-16)**3 # number density, input taken in 1/cm^3, passed in eV^3
        
        self.B_0 = B0 * 1.95e-20 * (1e9)**2. # pass to func in units of Gauss, convert here to eV^2
        
        self.umax = umax
        if dist_func == 'Waterbag':
            self.F0 = self.Waterbag ## dont use... think you need to impliment W func analytically.
            self.Wfunc = self.W_Waterbag
        elif dist_func == 'Softbell':
            self.F0 = self.Softbell
            self.DF0 = self.Der_Softbell
            self.Wfunc = self.W_Softbell ## -- something wrong here...

        self.Omega_vec = [-np.sqrt(1. / 137) * self.B_0 / 5.11e5, np.sqrt(1. / 137) * self.B_0 / 5.11e5]
        #print 'Plasma....', self.n_vec
        self.omega_p = [np.sqrt(4. * np.pi / 137. * self.n_vec[0] / 5.11e5), np.sqrt(4. * np.pi / 137. * self.n_vec[1] / 5.11e5)]
        if np.max(np.asarray(self.omega_p)**2 / np.asarray(self.Omega_vec)**2.) > 1:
            # this invalidates melrose et al treatment
            self.Bfield_off = True
        else:
            self.Bfield_off = False
        ## Assumes distribution function only dependent on u_z (directed along B field), and symmmetrtic in +/-s
        
    def diE_tensor(self, omega):
        eps = np.zeros((3,3), dtype=complex)
        
        n_perp = np.sqrt(self.kvec[0]**2. + self.kvec[1]**2.) / omega
        n_para = self.kvec[2] / omega

        mean_uz = [0., 0.] # implicit assumption of Gedalin (justified near eq 45)
        mean_gamma = [0., 0.]
        mean_uz2_d_gamma = [0., 0.]
        mean_uz_d_gamma = [0., 0.]
        Wvec = np.zeros(2, dtype=complex)
        for i in range(2):
            mean_gamma[i] = quad(lambda x: self.F0(x, um=self.umax[i]) * np.sqrt(1. + x**2.), -self.umax[i], self.umax[i])[0]
            mean_uz2_d_gamma[i] = quad(lambda x: self.F0(x, um=self.umax[i]) * x**2. / np.sqrt(1. + x**2.), -self.umax[i], self.umax[i])[0]
            mean_uz_d_gamma[i] = quad(lambda x: self.F0(x, um=self.umax[i]) * x / np.sqrt(1. + x**2.), -self.umax[i], self.umax[i])[0]
#            vr = 1. / n_para
#            if vr >= 1:
#                vr = 1.
#                gamma_r = 1e10
#            else:
#                gamma_r = 1. / np.sqrt(1. - vr**2.)

            Wvec[i] = self.Wfunc(n_para, um=self.umax[i])
            
        eps[2, 2] = 1.
        eps[0, 0] = 1.
        eps[1, 1] = 1.
        for i in range(2):
            if not self.Bfield_off:
                eps[2, 2] += - self.omega_p[i]**2. / omega**2. * Wvec[i] + self.omega_p[i]**2 * n_perp**2. / self.Omega_vec[i]**2. * mean_uz2_d_gamma[i]
            
                eps[1, 2] += cmath.sqrt(-1.) * (self.omega_p[i]**2. * n_perp / (omega * self.Omega_vec[i]) * mean_uz_d_gamma[i])
                eps[2, 1] += - cmath.sqrt(-1.) * (self.omega_p[i]**2. * n_perp / (omega * self.Omega_vec[i]) * mean_uz_d_gamma[i])
                
                
                eps[0, 2] += self.omega_p[i]**2. * n_perp / self.Omega_vec[i]**2. * (mean_uz[i] - n_para * mean_uz2_d_gamma[i])
                eps[2, 0] += self.omega_p[i]**2. * n_perp / self.Omega_vec[i]**2. * (mean_uz[i] - n_para * mean_uz2_d_gamma[i])

                eps[0, 1] += - cmath.sqrt(-1) * self.omega_p[i]**2. / (omega * self.Omega_vec[i]) * (1. - n_para * mean_uz_d_gamma[i])
                eps[1, 0] += cmath.sqrt(-1) * self.omega_p[i]**2. / (omega * self.Omega_vec[i]) * (1. - n_para * mean_uz_d_gamma[i])
                
                eps[0, 0] += self.omega_p[i]**2 / (self.Omega_vec[i]**2.) * (mean_gamma[i] - 2. * n_para * mean_uz[i] + n_para**2. * mean_uz2_d_gamma[i])
                eps[1, 1] += self.omega_p[i]**2 / (self.Omega_vec[i]**2.) * (mean_gamma[i] - 2. * n_para * mean_uz[i] + n_para**2. * mean_uz2_d_gamma[i])
            else:
                eps[0, 0] += - self.omega_p[i]**2. / omega**2. * Wvec[i]
                eps[1, 1] += - self.omega_p[i]**2. / omega**2. * Wvec[i]
                eps[2, 2] += - self.omega_p[i]**2. / omega**2. * Wvec[i]
    
        return eps
        
    def disper_relation(self, omega):
        # this is matrix formalism in Gedalin 2001
        n_x = self.kvec[0] / omega
        n_y = self.kvec[1] / omega
        n_z = self.kvec[2] / omega
        n = np.sqrt(np.dot(self.kvec, self.kvec)) / omega
        eps = self.diE_tensor(omega)
        matrixA = np.zeros_like(eps)
        matrixA[0, 0] = n**2. - n_x**2.
        matrixA[0, 1] = - n_x*n_y
        matrixA[0, 2] = - n_x*n_z
        matrixA[1, 2] = - n_y*n_z
        matrixA[2, 0] = matrixA[0, 2]
        matrixA[1, 0] = matrixA[0, 1]
        matrixA[2, 1] = matrixA[1, 2]
        matrixA[1, 1] = n**2. - n_y**2.
        matrixA[2, 2] = n**2. - n_z**2.
        matrixA -= eps
        detV = np.linalg.det(matrixA)
        # taking real part for now, ignoring possible absorption effects...
        return np.abs(detV.real)
    
    
    def get_omega(self):
        log_omega = fsolve(lambda x: self.disper_relation(10.**x), np.log10(np.sqrt(np.dot(self.omega_p,self.omega_p) + np.dot(self.kvec, self.kvec))))
        #print 'Minimization Result...', 10.**log_omega, self.disper_relation(10.**log_omega), np.sqrt(self.omega_p[0]**2. + np.dot(self.kvec, self.kvec)), self.omega_p[0], self.kvec
        #exit()
        return 10.**log_omega[0]
    

    def Waterbag(self, uz, um=10):
        if um**2 > uz**2:
            return 1. / (2. * um)
        else:
            return 0.

    def W_Waterbag(self, n_para, um=10):
        gamma_m = np.sqrt(1. + um**2.)
        vm = um / gamma_m
        return 1. / (gamma_m * (1. - n_para**2. * vm**2.))
        
    def Softbell(self, uz, um=10):
        if um**2 > uz**2:
            return 15./(16*um**5.) * (um**2. - uz**2.)**2.
        else:
            return 0.

    def Der_Softbell(self, uz, um=10):
        if um**2 > uz**2.:
            return 15./(16*um**5.) * 2.* (um**2. - uz**2.) * -2. * uz
        else:
            return 0.

    def W_Softbell(self, n_para, um=10):
        # there is something weird with this distribution right now .... normalization check fails. maybe typo from paper?
        gamma_m = np.sqrt(1. + um**2.)
        vm = um / gamma_m
        
        pre_real = 15.*gamma_m**2. / (4.*um**5 * (n_para**2. - 1)**3.)
        real_term = np.log(np.abs((1. + vm) / (1. - vm)))/8. * ((3.+vm**2.)*(3*n_para**2.+1) - n_para**2.*(3.*vm**2. +1)*(n_para**2.+3))
        real_term += .25*um*gamma_m*(n_para**2. - 1)*(3*vm**2 *n_para**2. + vm**2. - n_para**2. - 3.)
        real_term += n_para*(n_para**2. * vm**2. - 1.) * np.log(np.abs( (n_para*vm + 1)/(n_para*vm - 1) ))
        if n_para*vm > 1:
            im_term = 15.*np.pi*gamma_m**2. * n_para*(n_para**2. * vm**2. - 1.) / (4.*um**5. * (n_para**2. - 1.)**3.)
        else:
            im_term = 0.
        return real_term * pre_real - cmath.sqrt(-1)*im_term
