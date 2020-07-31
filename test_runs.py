from dispersion import *
from perturb_system import *



#kvec = np.array([1e-8, 1e-5, 1e-5]) # y-entry assumed 0 by definition
#xvec = np.array([0, 0, 0])
#nvec = np.array([1e10, 1e-10])
#B0 = 1e15
#dist_func = 'Waterbag'
#umax = [1e-3, 1e-3]
#disperClass = dispersion_Eqns(kvec, xvec, nvec, B0, dist_func=dist_func, umax=umax)
#disperClass.diE_tensor(10.*np.dot(kvec,kvec))
#slv = disperClass.get_omega()

angle = np.pi/2
kvec_norm = 1e-6
dist = 1e4 # km
B0 = 1e15

x_vec = np.array([dist * np.sin(angle), 0., dist * np.cos(angle)])
k_vec = np.array([kvec_norm * np.sin(angle), 0., kvec_norm * np.cos(angle)])
R_ev = Ray_Evolution(GJ, x_vec, k_vec, B0, 1., 10, umaxFix=1e-1, init_dist=dist).loop_steps(h_eta=0.8, h_x=1e-1, h_k=kvec_norm/1e5)
