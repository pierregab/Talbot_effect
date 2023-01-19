import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm

plt.rcParams['figure.dpi'] = 300


mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854187817e-12
c = 3e8

dx = 0.01
dy = 0.01  # spatial step
dt = 1.4*dx / c  # temporal step


fact = 0.005

x_max = 3000
y_max = 1300
t_max = 2000



Ez = np.zeros((x_max, y_max, 2), dtype= "float64")
Hx = np.zeros((x_max, y_max, 2), dtype= "float64")
Hy = np.zeros((x_max, y_max, 2), dtype= "float64")


    
def update_fields(Ez, Hx, Hy):

    """Yee"""
    for n in range(t_max-1):
        
        if n%2 == 0:
            
            Ez[1:-1,1:-1,1] = Ez[1:-1,1:-1,0] + dt/(2*epsilon_0)*((Hy[2:,1:-1,0]-Hy[0:-2,1:-1,0])/dx-(Hx[1:-1,2:,0]-Hx[1:-1,0:-2,0])/dy) #principal
            Ez[1:-1,1:-1,0] = Ez[1:-1,1:-1,1]
            
            """
            Ez[0,1:-1,n+1] = Ez[0,1:-1,n] + dt/(2*epsilon_0)*((Hy[1,1:-1,n]-Hy[-1,1:-1,n])/dx-(Hx[0,2:,n]-Hx[0,0:-2,n])/dy)
            Ez[-1,1:-1,n+1] = Ez[-1,1:-1,n] + dt/(2*epsilon_0)*((Hy[0,1:-1,n]-Hy[-2,1:-1,n])/dx-(Hx[-1,2:,n]-Hx[-1,0:-2,n])/dy)
            Ez[1:-1,0,n+1] = Ez[1:-1,0,n] + dt/(2*epsilon_0)*((Hy[2:,0,n]-Hy[2:,0,n])/dx-(Hx[1:-1,1,n]-Hx[1:-1,-1,n])/dy)
            Ez[1:-1,-1,n+1] = Ez[1:-1,-1,n] + dt/(2*epsilon_0)*((Hy[2:,-1,n]-Hy[0:-2,-1,n])/dx-(Hx[1:-1,0,n]-Hx[1:-1,-2,n])/dy)
            
            
            Ez[0,0,n+1] = Ez[0,0,n] + dt/(2*epsilon_0)*((Hy[1,0,n]-Hy[-1,0,n])/dx-(Hx[0,1,n]-Hx[0,-1,n])/dy)
            Ez[-1,0,n+1] = Ez[-1,0,n] + dt/(2*epsilon_0)*((Hy[0,0,n]-Hy[-2,0,n])/dx-(Hx[-1,1,n]-Hx[-1,-1,n])/dy)
            Ez[0,-1,n+1] = Ez[0,-1,n] + dt/(2*epsilon_0)*((Hy[1,-1,n]-Hy[-1,-1,n])/dx-(Hx[0,0,n]-Hx[0,-2,n])/dy)
            Ez[-1,-1,n+1] = Ez[-1,-1,n] + dt/(2*epsilon_0)*((Hy[0,-1,n]-Hy[-2,-1,n])/dx-(Hx[-1,0,n]-Hx[-1,-2,n])/dy)
            """
            
                        
            for i in range(np.floor(x_max/40).astype(int)):
                Ez[i*80:80*(i+1),0:2,0] = 0
                Ez[i*80:i*80+10,0:2,0] = 100*np.sin(n/2)

            
            Hx[:,:,1] = Hx[:,:,0]
            Hy[:,:,1] = Hy[:,:,0]
                
            
        else:

                    
            Hx[1:-1,1:-1,1] = Hx[1:-1,1:-1,0] - dt/(2*mu_0*dy)*(Ez[1:-1,2:,0]-Ez[1:-1,0:-2,0]) #principal
            Hx[1:-1,1:-1,0] = Hx[1:-1,1:-1,1]
            
            """
            Hx[0,1:-1,n+1] = Hx[0,1:-1,n] - dt/(2*mu_0*dy)*(Ez[0,2:,n]-Ez[0,0:-2,n])  
            Hx[-1,1:-1,n+1] = Hx[-1,1:-1,n] - dt/(2*mu_0*dy)*(Ez[-1,2:,n]-Ez[-1,0:-2,n])
            Hx[1:-1,0,n+1] = Hx[1:-1,0,n] - dt/(2*mu_0*dy)*(Ez[1:-1,1,n]-Ez[1:-1,-1,n])
            Hx[1:-1,-1,n+1] = Hx[1:-1,-1,n] - dt/(2*mu_0*dy)*(Ez[1:-1,0,n]-Ez[1:-1,-2,n])
            
            
            Hx[0,0,n+1] = Hx[0,0,n] - dt/(2*mu_0*dy)*(Ez[0,1,n]-Ez[0,-1,n])
            Hx[-1,0,n+1] = Hx[-1,0,n] - dt/(2*mu_0*dy)*(Ez[-1,1,n]-Ez[-1,-1,n])
            Hx[0,-1,n+1] = Hx[0,-1,n] - dt/(2*mu_0*dy)*(Ez[0,0,n]-Ez[0,-2,n])
            Hx[-1,-1,n+1] = Hx[-1,-1,n] - dt/(2*mu_0*dy)*(Ez[-1,0,n]-Ez[-1,-2,n])
            """
            
            Hy[1:-1,1:-1,1] = Hy[1:-1,1:-1,0] + dt/(2*mu_0*dx)*(Ez[2:,1:-1,0]-Ez[0:-2,1:-1,0]) #principal
            Hy[1:-1,1:-1,0] = Hy[1:-1,1:-1,1]
            
            """
            Hy[0,1:-1,n+1] = Hy[0,1:-1,n] + dt/(2*mu_0*dx)*(Ez[1,1:-1,n]-Ez[-1,1:-1,n])
            Hy[-1,1:-1,n+1] = Hy[-1,1:-1,n] + dt/(2*mu_0*dx)*(Ez[0,1:-1,n]-Ez[-2,1:-1,n])
            Hy[1:-1,0,n+1] = Hy[1:-1,0,n] + dt/(2*mu_0*dx)*(Ez[2:,0,n]-Ez[2:,0,n])
            Hy[1:-1,-1,n+1] = Hy[1:-1,-1,n] + dt/(2*mu_0*dx)*(Ez[2:,-1,n]-Ez[2:,-1,n])
            
            Hy[0,0,n+1] = Hy[0,0,n] + dt/(2*mu_0*dx)*(Ez[1,0,n]-Ez[-1,0,n])
            Hy[-1,0,n+1] = Hy[-1,0,n] + dt/(2*mu_0*dx)*(Ez[0,0,n]-Ez[-2,0,n])
            Hy[0,-1,n+1] = Hy[0,-1,n] + dt/(2*mu_0*dx)*(Ez[1,-1,n]-Ez[-1,-1,n])
            Hy[-1,-1,n+1] = Hy[-1,-1,n] + dt/(2*mu_0*dx)*(Ez[0,-1,n]-Ez[-2,-1,n])
            """
                    
            Ez[:,:,1] = Ez[:,:,0]
        
        

update_fields(Ez, Hx, Hy)



Energie = np.zeros((x_max,y_max))
Energie[:,:]=np.power(Ez[:,:,1],2)*epsilon_0/2




im = plt.imshow(Energie[1300:1700,0:1175],norm=PowerNorm(gamma=0.4))
im.set_cmap('gray')
cbar = plt.colorbar(im)
plt.show()


