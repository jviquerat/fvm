# Generic imports
import os
import time
import math
import matplotlib.pyplot as plt

# Custom imports
from mixing_nb import *

#######################################
# Cavity flow
#######################################

#########################
# Initialization steps
#########################

# Set parameters
l     = 1.0
h     = 1.0
dx    = 0.01
dy    = 0.01
t_max = 20.0
cfl   = 0.5
nu    = 0.01
re    = 100.0
pe    = 1000.0
utop  = re*nu/l
nx    = round(l/dx)
ny    = round(h/dy)

# Set fields (account for boundary cells)
u   = np.zeros((nx+2, ny+2))
v   = np.zeros((nx+2, ny+2))
p   = np.zeros((nx+2, ny+2))
C   = np.zeros((nx+2, ny+2))
us  = np.zeros((nx+2, ny+2))
vs  = np.zeros((nx+2, ny+2))
phi = np.zeros((nx+2, ny+2))

# Initialize concentration field
C[30:70,30:70] = 2.0

# Array to store iterations of poisson resolution
n_itp = np.array([], dtype=np.int16)

# Set time
it     = 0
it_plt = 0
t      = 0.0

# Set output data
plt_freq = 50
path     = "results/mixing"
os.makedirs(path, exist_ok=True)
path_u    = path+"/velocity"
os.makedirs(path_u,    exist_ok=True)
path_c    = path+"/concentration"
os.makedirs(path_c,    exist_ok=True)

# Compute timestep
tau  = l/utop
mdxy = min(dx, dy)
dt   = cfl*min(tau/re, tau*re*mdxy**2/(4.0*l**2))

# Set timer
s_time = time.time()

#########################
# Main loop
#########################
while (t < t_max):

    #########################
    # Set boundary conditions
    #########################

    # Left wall
    u[1,1:-1]  = 0.0
    v[0,2:-1]  =-v[1,2:-1]
    C[0,1:-1]  = C[1,1:-1]

    # Right wall
    u[-1,1:-1] = 0.0
    v[-1,2:-1] =-v[-2,2:-1]
    C[-1,1:-1] = C[-2,1:-1]

    # Top wall
    u[1:,-1]   = 2.0*utop - u[1:,-2]
    v[1:-1,-1] = 0.0
    C[1:-1,-1] = C[1:-1,-2]

    # Bottom wall
    u[1:,0]    =-2.0*utop - u[1:,1]
    v[1:-1,1]  = 0.0
    C[1:-1,0]  = C[1:-1,1]

    #########################
    # Predictor step
    # Computes starred fields
    #########################

    predictor(u, v, us, vs, p, nx, ny, dt, dx, dy, re)

    #########################
    # Poisson step
    # Computes pressure field
    #########################

    itp, ovf = poisson(us, vs, u, phi, nx, ny, dx, dy, dt)
    p[:,:]  += phi[:,:]

    n_itp = np.append(n_itp, np.array([it, itp]))

    if (ovf):
        print("\n")
        print("Exceeded max number of iterations in solver")
        exit(1)

    #########################
    # Corrector step
    # Computes div-free fields
    #########################

    corrector(u, v, us, vs, phi, nx, ny, dx, dy, dt)

    #########################
    # Transport step
    #########################

    transport(u, v, C, nx, ny, dx, dy, dt, pe)

    #########################
    # Printings
    #########################

    t  += dt
    it += 1

    end="\r"
    if (t>=t_max): end="\n"
    print('# t = '+'{:f}'.format(t)+'/'+'{:f}'.format(t_max), end=end)

    #########################
    # Plot fields
    #########################

    if (it%plt_freq==0):

        # Recreate fields at cells centers
        pu = np.zeros((nx, ny))
        pv = np.zeros((nx, ny))
        pp = np.zeros((nx, ny))
        pC = np.zeros((nx, ny))

        pu[:,:] = 0.5*(u[2:,1:-1] + u[1:-1,1:-1])
        pv[:,:] = 0.5*(v[1:-1,2:] + v[1:-1,1:-1])
        pp[:,:] = p[1:-1,1:-1]
        pC[:,:] = C[1:-1,1:-1]

        # Compute velocity norm
        vn = np.sqrt(pu*pu+pv*pv)

        # Rotate fields
        vn = np.rot90(vn)
        pp = np.rot90(pp)
        pC = np.rot90(pC)

        # Plot velocity
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(vn,
                   cmap = 'RdBu_r',
                   vmin = 0.0,
                   vmax = utop)

        filename = path_u+"/"+str(it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        # Plot concentration
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(pC,
                   cmap = 'RdBu_r',
                   vmin = 0.0,
                   vmax = 1.0)

        filename = path_c+"/"+str(it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        it_plt += 1

#########################
# Final timing
#########################

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))

#########################
# Plot iterations of poisson solver
#########################

n_itp = np.reshape(n_itp, (-1,2))

plt.clf()
fig, ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(n_itp[:,0], n_itp[:,1], color='blue')
ax.grid(True)
fig.tight_layout()
filename = path+"/iterations.png"
fig.savefig(filename)
np.savetxt(path+"/iterations.dat", n_itp, fmt='%d')
