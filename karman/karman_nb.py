# Generic imports
import numpy as np
import numba as nb

###############################################
# Predictor step
@nb.njit(cache=False)
def predictor(u, v, us, vs, p, nx, ny, dt, dx, dy, re):

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            uE = 0.5*(u[i+1,j] + u[i,j])
            uW = 0.5*(u[i,j]   + u[i-1,j])
            uN = 0.5*(u[i,j+1] + u[i,j])
            uS = 0.5*(u[i,j]   + u[i,j-1])
            vN = 0.5*(v[i,j+1] + v[i-1,j+1])
            vS = 0.5*(v[i,j]   + v[i-1,j])
            conv = (uE*uE-uW*uW)/dx + (uN*vN-uS*vS)/dy

            diff = ((u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/(dx**2) +
                    (u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/(dy**2))/re

            pres = (p[i,j] - p[i-1,j])/dx

            us[i,j] = u[i,j] + dt*(diff - conv - pres)

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            vE = 0.5*(v[i+1,j] + v[i,j])
            vW = 0.5*(v[i,j]   + v[i-1,j])
            uE = 0.5*(u[i+1,j] + u[i+1,j-1])
            uW = 0.5*(u[i,j]   + u[i,j-1])
            vN = 0.5*(v[i,j+1] + v[i,j])
            vS = 0.5*(v[i,j]   + v[i,j-1])
            conv = (uE*vE-uW*vW)/dx + (vN*vN-vS*vS)/dy

            diff = ((v[i+1,j] - 2.0*v[i,j] + v[i-1,j])/(dx**2) +
                    (v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/(dy**2))/re

            pres = (p[i,j] - p[i,j-1])/dy

            vs[i,j] = v[i,j] + dt*(diff - conv - pres)

###############################################
# Poisson step
@nb.njit(cache=False)
def poisson(us, vs, u, phi, nx, ny, c_xmin, c_xmax, c_ymin, c_ymax, dx, dy, dt):

    tol      = 1.0e-5
    err      = 1.0e10
    itp      = 0
    itmax    = 300000
    ovf      = False
    phi[:,:] = 0.0
    phin     = np.zeros((nx+2,ny+2))
    while(err > tol):

        phin[:,:] = phi[:,:]

        for i in range(2,nx):
            for j in range(1,ny+1):

                b = ((us[i+1,j] - us[i,j])/dx +
                     (vs[i,j+1] - vs[i,j])/dy)/dt

                phi[i,j] = 0.5*((phin[i+1,j] + phin[i-1,j])*dy*dy +
                                (phin[i,j+1] + phin[i,j-1])*dx*dx -
                                b*dx*dx*dy*dy)/(dx*dx+dy*dy)


        # Domain left (dirichlet)
        phi[1,1:-1] = (1.0/(dy*dy+2.0*dx*dx))*(dy*dy*phi[2,1:-1] + dx*dx*(phin[1,2:] + phin[1,:-2]) -
                            (dx*dx*dy*dy/dt)*((us[2,1:-1] - u[1,1:-1])/dx + (vs[1,2:] - vs[1,1:-1])/dy))

        # Domain right (dirichlet)
        phi[-1,1:-1] =-phi[-2,1:-1]

        # Domain top (neumann)
        phi[1:-1,-1] = phi[1:-1,-2]

        # Domain bottom (neumann)
        phi[1:-1, 0] = phi[1:-1, 1]

        phi[c_xmin:c_xmax+1,c_ymin:c_ymax+1] = 0.0

        # Obstacle left
        phi[c_xmin, c_ymin:c_ymax+1] = phi[c_xmin-1, c_ymin:c_ymax+1]

        # Obstacle right
        phi[c_xmax, c_ymin:c_ymax+1] = phi[c_xmax+1, c_ymin:c_ymax+1]

        # Obstacle top
        phi[c_xmin:c_xmax+1, c_ymax] = phi[c_xmin:c_xmax+1, c_ymax+1]

        # Obstacle bottom
        phi[c_xmin:c_xmax+1, c_ymin] = phi[c_xmin:c_xmax+1, c_ymin-1]

        # Compute error
        dphi = np.reshape(phi - phin, (-1))
        err  = np.dot(dphi,dphi)

        itp += 1
        if (itp > itmax):
            ovf = True
            break

    return itp, ovf

###############################################
# Corrector step
@nb.njit(cache=False)
def corrector(u, v, us, vs, phi, nx, ny, dx, dy, dt):

    u[2:-1,1:-1] = us[2:-1,1:-1] - dt*(phi[2:-1,1:-1] - phi[1:-2,1:-1])/dx
    v[1:-1,2:-1] = vs[1:-1,2:-1] - dt*(phi[1:-1,2:-1] - phi[1:-1,1:-2])/dy
