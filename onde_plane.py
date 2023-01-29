import numpy as np 
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['figure.dpi'] = 350


mu_0 = 4 * np.pi * 1e-7
epsilon_0 = 8.854187817e-12
c = 3e8

dx = 0.01
dy = 0.01  # spatial step
dt = dx / c  # temporal step

lamb = 200


x_max = 300
y_max = 600
t_max = 800



Ez = np.zeros((x_max, y_max, 2), dtype= "float64")
Hx = np.zeros((x_max, y_max, 2), dtype= "float64")
Hy = np.zeros((x_max, y_max, 2), dtype= "float64")


    
def update_fields(Ez, Hx, Hy):

    """Yee"""
    for n in range(t_max):
        
        if n%2 == 0:
            
            Ez[1:-1,1:-1,1] = Ez[1:-1,1:-1,0] + dt/(2*epsilon_0)*((Hy[2:,1:-1,0]-Hy[0:-2,1:-1,0])/dx-(Hx[1:-1,2:,0]-Hx[1:-1,0:-2,0])/dy) #principal
  
            
            Ez[0,1:-1,1] = Ez[0,1:-1,0] + dt/(2*epsilon_0)*((Hy[1,1:-1,0]-Hy[-1,1:-1,0])/dx-(Hx[0,2:,0]-Hx[0,0:-2,0])/dy)
            Ez[-1,1:-1,1] = Ez[-1,1:-1,0] + dt/(2*epsilon_0)*((Hy[0,1:-1,0]-Hy[-2,1:-1,0])/dx-(Hx[-1,2:,0]-Hx[-1,0:-2,0])/dy)
            """Ez[1:-1,0,1] = Ez[1:-1,0,0] + dt/(2*epsilon_0)*((Hy[2:,0,0]-Hy[2:,0,0])/dx-(Hx[1:-1,1,0]-Hx[1:-1,-1,0])/dy)
            Ez[1:-1,-1,1] = Ez[1:-1,-1,0] + dt/(2*epsilon_0)*((Hy[2:,-1,0]-Hy[0:-2,-1,0])/dx-(Hx[1:-1,0,0]-Hx[1:-1,-2,0])/dy)
            """
            
            """
            Ez[0,0,1] = Ez[0,0,0] + dt/(2*epsilon_0)*((Hy[1,0,0]-Hy[-1,0,0])/dx-(Hx[0,1,0]-Hx[0,-1,0])/dy)
            Ez[-1,0,1] = Ez[-1,0,0] + dt/(2*epsilon_0)*((Hy[0,0,0]-Hy[-2,0,0])/dx-(Hx[-1,1,0]-Hx[-1,-1,0])/dy)
            Ez[0,-1,1] = Ez[0,-1,0] + dt/(2*epsilon_0)*((Hy[1,-1,0]-Hy[-1,-1,0])/dx-(Hx[0,0,0]-Hx[0,-2,0])/dy)
            Ez[-1,-1,1] = Ez[-1,-1,0] + dt/(2*epsilon_0)*((Hy[0,-1,0]-Hy[-2,-1,0])/dx-(Hx[-1,0,0]-Hx[-1,-2,0])/dy)
            """
            Ez[:,:,0] = Ez[:,:,1]
            
            
            
            Ez[:,0:2,0] = 100*np.sin(n*2*np.pi/lamb)       

                
            
        else:

                    
            Hx[1:-1,1:-1,1] = Hx[1:-1,1:-1,0] - dt/(2*mu_0*dy)*(Ez[1:-1,2:,0]-Ez[1:-1,0:-2,0]) #principal
            
            
            Hx[0,1:-1,1] = Hx[0,1:-1,0] - dt/(2*mu_0*dy)*(Ez[0,2:,0]-Ez[0,0:-2,0])  
            Hx[-1,1:-1,1] = Hx[-1,1:-1,0] - dt/(2*mu_0*dy)*(Ez[-1,2:,0]-Ez[-1,0:-2,0])
            """Hx[1:-1,0,1] = Hx[1:-1,0,0] - dt/(2*mu_0*dy)*(Ez[1:-1,1,0]-Ez[1:-1,-1,0])
            Hx[1:-1,-1,1] = Hx[1:-1,-1,0] - dt/(2*mu_0*dy)*(Ez[1:-1,0,0]-Ez[1:-1,-2,0])
            """
            
            """
            Hx[0,0,1] = Hx[0,0,0] - dt/(2*mu_0*dy)*(Ez[0,1,0]-Ez[0,-1,0])
            Hx[-1,0,1] = Hx[-1,0,0] - dt/(2*mu_0*dy)*(Ez[-1,1,0]-Ez[-1,-1,0])
            Hx[0,-1,1] = Hx[0,-1,0] - dt/(2*mu_0*dy)*(Ez[0,0,0]-Ez[0,-2,0])
            Hx[-1,-1,1] = Hx[-1,-1,0] - dt/(2*mu_0*dy)*(Ez[-1,0,0]-Ez[-1,-2,0])
            """
            Hx[:,:,0] = Hx[:,:,1]
            
            
            Hy[1:-1,1:-1,1] = Hy[1:-1,1:-1,0] + dt/(2*mu_0*dx)*(Ez[2:,1:-1,0]-Ez[0:-2,1:-1,0]) #principal
            
            
            Hy[0,1:-1,1] = Hy[0,1:-1,0] + dt/(2*mu_0*dx)*(Ez[1,1:-1,0]-Ez[-1,1:-1,0])
            Hy[-1,1:-1,1] = Hy[-1,1:-1,0] + dt/(2*mu_0*dx)*(Ez[0,1:-1,0]-Ez[-2,1:-1,0])
            """Hy[1:-1,0,1] = Hy[1:-1,0,0] + dt/(2*mu_0*dx)*(Ez[2:,0,0]-Ez[2:,0,0])
            Hy[1:-1,-1,1] = Hy[1:-1,-1,0] + dt/(2*mu_0*dx)*(Ez[2:,-1,0]-Ez[2:,-1,0])
            """
            
            """
            Hy[0,0,1] = Hy[0,0,0] + dt/(2*mu_0*dx)*(Ez[1,0,0]-Ez[-1,0,0])
            Hy[-1,0,1] = Hy[-1,0,0] + dt/(2*mu_0*dx)*(Ez[0,0,0]-Ez[-2,0,0])
            Hy[0,-1,1] = Hy[0,-1,0] + dt/(2*mu_0*dx)*(Ez[1,-1,0]-Ez[-1,-1,0])
            Hy[-1,-1,1] = Hy[-1,-1,0] + dt/(2*mu_0*dx)*(Ez[0,-1,0]-Ez[-2,-1,0])
            """
            Hy[:,:,0] = Hy[:,:,1]
                    
        
        

update_fields(Ez, Hx, Hy)



Energie = np.zeros((x_max,y_max))
Energie[:,:]=np.power(Ez[:,:,1],2)*epsilon_0/2




fig2 = plt.figure()
im = plt.imshow(Energie[:,:],norm=PowerNorm(gamma=0.4))
im.set_cmap('gray')
#cbar = plt.colorbar(im, shrink=0.6, anchor=0.1)
plt.show()


im = plt.imshow(Ez[:,:,1],norm=PowerNorm(gamma=1))
plt.colorbar()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(Ez.shape[0])
y = np.arange(Ez.shape[1])
X, Y = np.meshgrid(x, y)
ax.set_box_aspect(aspect = (1.5,2,0.7))
ax.plot_surface(X.T, Y.T, Ez[:,:,1], cmap='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Intensity')
ax.view_init(-150, 20)
plt.show()

"""
x=np.arange(2,400,1, dtype=int)
fig1 = plt.figure()
plt.plot(x,Ez[150,2:400,1])
"""

fft_data = np.fft.fft(Ez[150,2:400,1])
freqs = np.fft.fftfreq(len(Ez[150,2:400,1]))
idx = np.argmax(np.abs(fft_data))
period = 1/freqs[idx]

print("Period:", period)

