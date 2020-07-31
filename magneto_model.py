import numpy as np

class GJ(object):
    # Goldreich-Julien Model, no misalignment angle between B and rotation
    def __init__(self, Bnorm, Per, r_0, umaxFix=1e-1):
        self.Bnorm = Bnorm # [Gauss]
        self.Per = Per # [sec]
        self.r_0 = r_0 # [km]
        self.umaxFix = umaxFix

        return

    def B0(self, vec_x):
        r = np.sqrt(np.dot(vec_x, vec_x))
        cos_theta = (vec_x[2] / r)
        
        return self.Bnorm / 2. * (self.r_0 / r)**3. * (3. * cos_theta**2.)

    def epm_den(self, vec_x):
        r = np.sqrt(np.dot(vec_x, vec_x))
        cos_theta = (vec_x[2] / r)
        sin_theta = np.sqrt(1. - cos_theta**2.)
        n_c = 2. * 2*np.pi / self.Per * self.B0(vec_x) * np.sqrt(137) / (1. - (2*np.pi / self.Per)**2. * (r/2.998e5)**2. * sin_theta**2.) # G / s
        n_c *= 1.95e-20 * (1e9)**2. / 2.998e10**3. / (6.58e-16)**2. # 1/cm^3
        return np.asarray([np.abs(n_c), 1e-8 * np.abs(n_c)]) # assume GJ n_c ~ n_e >> n_e+

    def umax(self, vec_x):
        return np.asarray([self.umaxFix, self.umaxFix])
