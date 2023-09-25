# Generic imports
import numpy as np
import numba as nb

###############################################
# Predictor step
@nb.njit(cache=False)
def predictor(u, v, us, vs, p, nx, ny, c_xmin, c_xmax, c_ymin, c_ymax, dt, dx, dy, re):

    for i in range(2,nx+1):
        for j in range(1,ny+1):

            uE  = u[i+1,j]
            uEE = u[i+2,j]
            uW  = u[i-1,j]
            uWW = u[i-2,j]
            uN  = u[i,j+1]
            uNN = u[i,j+2]
            uS  = u[i,j-1]
            uSS = u[i,j-2]
            uC  = u[i,j]
            uP  = 0.5*(uC + abs(uC))
            uM  = 0.5*(uC - abs(uC))

            vN = 0.5*(v[i,j+1] + v[i-1,j+1])
            vS = 0.5*(v[i,j]   + v[i-1,j])
            vC = 0.5*(vN + vS)
            vP = 0.5*(vC + abs(vC))
            vM = 0.5*(vC - abs(vC))

            conv = uP*(3.0*uC-4.0*uW+uWW)/(2.0*dx) + uM*(-uEE+4.0*uE-3.0*uC)/(2.0*dx) + vP*(3.0*uC-4.0*uS+uSS)/(2.0*dy) + vM*(-uNN+4.0*uN-3.0*uC)/(2.0*dy)

            diff = ((u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/(dx**2) +
                    (u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/(dy**2))/re

            pres = (p[i,j] - p[i-1,j])/dx

            us[i,j] = u[i,j] + dt*(diff - conv - pres)

    for i in range(1,nx+1):
        for j in range(2,ny+1):

            vE  = v[i+1,j]
            vEE = v[i+2,j]
            vW  = v[i-1,j]
            vWW = v[i-2,j]
            vN  = v[i,j+1]
            vNN = v[i,j+2]
            vS  = v[i,j-1]
            vSS = v[i,j-2]
            vC  = v[i,j]
            vP  = 0.5*(vC + abs(vC))
            vM  = 0.5*(vC - abs(vC))

            uE = 0.5*(u[i+1,j] + u[i+1,j-1])
            uW = 0.5*(u[i,j]   + u[i,j-1])
            uC = 0.5*(uE + uW)
            uP = 0.5*(uC + abs(uC))
            uM = 0.5*(uC - abs(uC))

            conv = uP*(3.0*vC-4.0*vW+vWW)/(2.0*dx) + uM*(-vEE+4.0*vE-3.0*vC)/(2.0*dx) + vP*(3.0*vC-4.0*vS+vSS)/(2.0*dy) + vM*(-vNN+4.0*vN-3.0*vC)/(2.0*dy)

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

        for i in range(1,nx+1):
            for j in range(1,ny+1):
                if ((i >= c_xmin) and (i <= c_xmax) and
                    (j >= c_ymin) and (j <= c_ymax)): continue

                b = ((us[i+1,j] - us[i,j])/dx +
                     (vs[i,j+1] - vs[i,j])/dy)/dt

                phi[i,j] = 0.5*((phin[i+1,j] + phin[i-1,j])*dy*dy +
                                (phin[i,j+1] + phin[i,j-1])*dx*dx -
                                b*dx*dx*dy*dy)/(dx*dx+dy*dy)


        # Domain left (dirichlet)
        phi[1,1:-1] = (1.0/(dy*dy+2.0*dx*dx))*(
            dy*dy*phi[2,1:-1] + dx*dx*(phin[1,2:] + phin[1,:-2]) -
            (dx*dx*dy*dy/dt)*((us[2,1:-1] - u[1,1:-1])/dx + (vs[1,2:] - vs[1,1:-1])/dy))

        # Domain right (dirichlet)
        phi[-2,1:-1] =-phi[-3,1:-1]
        phi[-1,1:-1] =-phi[-2,1:-1]

        # Domain top (neumann)
        phi[1:-1,-1] = phi[1:-1,-2]

        # Domain bottom (neumann)
        phi[1:-1, 0] = phi[1:-1, 1]

        # Inner obstacle
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
def corrector(u, v, us, vs, phi, nx, ny, c_xmin, c_xmax, c_ymin, c_ymax, dx, dy, dt):

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            if ((i >= c_xmin) and (i <= c_xmax+1) and
                (j >= c_ymin) and (j <= c_ymax)): continue
            u[i,j] = us[i,j] - dt*(phi[i,j] - phi[i-1,j])/dx

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            if ((i >= c_xmin) and (i <= c_xmax) and
                (j >= c_ymin) and (j <= c_ymax+1)): continue
            v[i,j] = vs[i,j] - dt*(phi[i,j] - phi[i,j-1])/dy
