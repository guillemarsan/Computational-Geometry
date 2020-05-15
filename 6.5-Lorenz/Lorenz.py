# -*- coding: utf-8 -*-
"""
Coursework 6 Extra: Phase space of Lorenz Attractor
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Compute the derivative dq from an array of consecutive points q with initial 
# derivative dq0 with timestep d
def deriv(q,d=0.001):
   dq = (q[1:len(q),:]-q[0:(len(q)-1),:])/d 
   return dq

# Differential equation of Lorenz Attractor
def F(q):
    ddq1 = 10*(q[1]-q[0])
    ddq2 = 28*q[0] - q[1] -q[0]*q[2]
    ddq3 = q[0]*q[1] - (8/3)*q[2]
    return np.array([ddq1, ddq2, ddq3])

# Compute the orbit of the differential equation F with initial values q0 and dq0,
# n points and timestep d
def orb(n,q0, F, d=0.001):
    q = np.empty([n+1,3])
    q[0,:] = q0
    for i in np.arange(1,n+1):
        q[i,:] = q[i-1,:] + d*F(q[i-1,:])
    return q 

# Plot an orbit of the differential equation F with initial values q0 and dq0,
# n points and timestep d with color col and linestyle marker. Return orbits computed
def simplectica(q0,F,ax1,ax2,ax3,col = 0, d = 10**(-4),n = int(16/10**(-4)),marker='-'): 
    q = orb(n,q0=q0,F=F,d=d)
    dq = deriv(q,d=d)
    q = q[:-1,:]
    p = dq/2
    c=plt.get_cmap("inferno",125)(col)
    ax1.plot(q[:,0], p[:,0], linewidth = 0.25,  c=c )
    ax2.plot(q[:,1], p[:,1], linewidth = 0.25, c=c)
    ax3.plot(q[:,2], p[:,2], linewidth = 0.25, c=c)
    return q,p
    
# Compute for some n the estimation of the box-counting dimension for 
# a cover of epsilon = 2/(n-1) 6-squares
def box_count(points, n):
    lim = np.linspace(-1,1,n)
    H,_ = np.histogramdd(points, bins = (lim,lim,lim,lim,lim,lim))
    count = np.sum(H > 0)
    return count

    
#%%
"""
Exercise 2: Plot (q1,q2,q3) and (p1,p2,p3)
"""

# Compute orbit
d = 1e-3
q = orb(30000,[1,1,1],F)
dq = deriv(q)
p = dq/2

# Plot (q1,q2,q3)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(1,1,1,c='r')
ax.plot3D(q[:,0],q[:,1],q[:,2])

# Plot (p1,p2,p3)
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(p[0,0],p[1,1],p[2,2],c='r')
ax.plot3D(p[:,0],p[:,1],p[:,2])

#%%

"""
Exercise 3: Plot (q1,p1), (q2,p2) and (q3,p3)
"""

# Compute ticks of interval: C
seq_q0 = np.linspace(-1,1.,num=5)

# Initialise figures
col = np.linspace(0,1,125)
c = 0
fig = plt.figure()
ax1 = fig.add_subplot(1,1, 1)
fig = plt.figure()
ax2 = fig.add_subplot(1,1, 1)
fig = plt.figure()
ax3 = fig.add_subplot(1,1, 1)

# Store data for next exercise
allq = np.zeros((1,3))
allp = np.zeros((1,3))

# Plot orbit for initial conditions in C x C x C
for i in range(len(seq_q0)):
    for j in range(len(seq_q0)):
        for k in range(len(seq_q0)):
            q0 = [seq_q0[i], seq_q0[j], seq_q0[k]]
            q,p = simplectica(q0=q0,F=F,ax1=ax1,ax2=ax2,ax3=ax3,col=c,marker='ro',
                              d= 10**(-3),n=int(30/d))
            c = c + 1
            
            # Store data for next exercise
            allq = np.concatenate((allq,q),axis =0)
            allp = np.concatenate((allp,p),axis =0)
            
ax1.set_xlabel("q(t)", fontsize=12)
ax1.set_ylabel("p(t)", fontsize=12)
ax2.set_xlabel("q(t)", fontsize=12)
ax2.set_ylabel("p(t)", fontsize=12)
ax3.set_xlabel("q(t)", fontsize=12)
ax3.set_ylabel("p(t)", fontsize=12)
plt.show()

#%%
"""
Exercise 4: Hausdorff dimension
"""
narr = range(5,28)
points = np.array(list(zip(allq[:,0],allq[:,1],allq[:,2],allp[:,0],allp[:,1],allp[:,2])))

# Normalize
m = np.max(np.abs(points))
points = points/m

# Test dimension d
d = 2.75

Hd = []
He = []
for n in narr:
    # count boxes with side epsilon=2/(n-1)
    count = box_count(points,n)
    
    # diameter if each box
    diam = np.sqrt(24)/(n-1)
    Hd.append(count*(diam**d))
    
    eps = 2/(n-1)
    He.append(np.log(count)/np.log(1/eps))
    
# Plot evolution of H^d_{\epsilon}
plt.figure()
plt.plot(narr,Hd)
plt.xlabel('$n$')
plt.ylabel('$H^{d}_{\epsilon}(E)$')

# Plot evolution of dim_{box}(\epsilon)
plt.figure()
plt.plot(narr,He)
plt.xlabel('$n$')
plt.ylabel('$dim_{box}(\epsilon)$')