# -*- coding: utf-8 -*-
"""
Extra: Sierpinski Triangle
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate n points in Sierpinski Triangle
def generate_triangle(n):
    p0 = [0,0]
    p1 = [2,0]
    p2 = [1,np.sqrt(3)]
    p = np.array([p0, p1, p2])
    vi = 1/2 * (p[0] + p[1])
    
    varr = np.array([vi])
    for i in range(n):
        r = np.random.randint(3)
        vi = 1/2 * (vi + p[r])
        varr = np.append(varr,[vi], axis=0)
        
    return varr

# Generate n points in Sierpinski Carpet
def generate_carpet(n):
    p0 = [0,0]
    p1 = [1,0]
    p2 = [2,0]
    p3 = [2,1]
    p4 = [2,2]
    p5 = [1,2]
    p6 = [0,2]
    p7 = [0,1]
    p = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    vi = 1/3 * (p[0] + p[1])
    
    varr = np.array([vi])
    for i in range(n):
        r = np.random.randint(8)
        vi = 1/3 * (vi + p[r])
        varr = np.append(varr,[vi], axis=0)
        
    return varr

# Compute for some d and n the estimation of the Hausdorff d-Volume for 
# a cover of epsilon = 2/(n-1) squares
def volume_d(set, n, d):
    x = np.linspace(0,2,n)
    eps = 2/(n-1)
    count = 0
    for i in range(n-1):
        inx = (x[i] < set[:,0]) & (x[i+1] > set[:,0])
        for j in range(n-1):
            iny = (x[j] < set[:,1]) & (x[j+1] > set[:,1])
            filled = np.any(inx & iny)
            count = count + filled
    return count*((np.sqrt(2)*eps)**d)

#%%
"""
Generate Sierpinski Triangle
"""
set = generate_triangle(50000)
plt.scatter(set[:,0],set[:,1], s=0.1)
plt.axis([0, 2, 0, 2]);
plt.gca().set_aspect('equal', adjustable='box')

#%%
"""
Generate Sierpinski Carpet
"""

set = generate_carpet(50000)
plt.figure()
plt.scatter(set[:,0],set[:,1], s=0.1)
plt.axis([0, 1, 0, 1]);
plt.gca().set_aspect('equal', adjustable='box')

#%%
"""
Compute Sierpinski's Carpet Hausdorff d-Volume
"""
set = generate_carpet(50000)
narr = range(2,150)

d = 1.95
Hd = []
for n in narr:
    Hd.append(volume_d(set,n,d))
     
plt.figure()
plt.plot(narr,Hd)
plt.xlabel('$n$')
plt.ylabel('$H^{d}_{\epsilon}(E)$')

#%%
"""
Compute Sierpinski's Triangle Hausdorff d-Volume
"""
set = generate_triangle(50000)
narr = range(2,150)

d = 1.66
Hd = []
for n in narr:
    Hd.append(volume_d(set,n,d))
  
plt.figure()
plt.plot(narr,Hd)
plt.xlabel('$n$')
plt.ylabel('$H^{d}_{\epsilon}(E)$')