# -*- coding: utf-8 -*-
"""
Coursework 1: Logistic atractor
"""
import matplotlib.pyplot as plt
import numpy as np

# Compute the orbit of n points of the function f from x0
def orbit(f,x0,n):
    orb = np.array([x0])
    for i in range(n):
        orb = np.append(orb,f(orb[i]))
    return orb;

# Compute the period of and orbit orb with difference epsilon
def period(orb,epsilon):
    n = len(orb)-1
    for k in range(1,n):
        if (abs(orb[n] - orb[n-k]) < epsilon):
            return k
    return -1

# Compute a better estimation (V10 +/- error) of the atractor set 
# from an initial estimation V0 and the function f
def error(f,V0):
    Vant = V0
    Vant.sort()
    DeltaV = []
    for i in range(10):
        Vsig = f(Vant)
        Vsig.sort()
        DeltaV.append(max(abs(Vant - Vsig)))
        Vant = Vsig
    DeltaV.sort()
    return (Vant,DeltaV[8])
        
# Compute an estimation V0 of the atractor set of f
def find_V0(f,x0,n,epsilon):
    orb = orbit(f,x0,n)
    k = period(orb, epsilon)
    if (k == -1):
        return (orb,k,[-1])
    else:
        V0 = orb[-k:].copy()
        V0.sort()
        return (orb,k,V0)
    
#%%
""" 
Set parameters manually and plot
"""
r = 3.56
x0 = 0.5 

n = 100
epsilon = 0.001

logistic = lambda x: r * x * (1-x)

print("For r = :", r)
(orb,k,V0) = find_V0(logistic,x0,n,epsilon)
if (k == -1):
     print("- For x0 = {:.1f}".format(x0), "no period found smaller than",n)
else:
    (V10,err) = error(logistic,V0)
    print("- For x0 = {:.1f}".format(x0), "we get k =", k, "and V10 = ",
          [round(x,3) for x in V10], "+- {:.1g}".format(err))  
    
plt.plot(orb)
plt.axis([0, n, 0, 1]);


#%%
"""
Exercise 1
"""
arr = np.linspace(3,3.45,5)

arX0 = np.linspace(0.1,0.9,9)

n = 1000
epsilon = 1e-5

for r in arr:
    logistic = lambda x: r * x * (1-x)
    print("For r = {:.2f}".format(r))
    for x0 in arX0:
        (orb,k,V0) = find_V0(logistic,x0,n,epsilon)
        if (k == -1):
            print("- For x0 = {:.1f}".format(x0), 
                  "no period found smaller than",n)
        else:
           (V10,err) = error(logistic,V0)
           print("- For x0 = {:.1f}".format(x0), "we get k =", k, "and V10 = ",
                 [round(x,3) for x in V10], "+- {:.1g}".format(err))
#%%     
"""
Exercise 2
"""
arr = np.linspace(3.5,3.7,8) 
arr = np.append(arr,np.linspace(3.8,3.9,8))


x0 = 0.5

n = 1000
epsilon = 1e-5


for r in arr:
    logistic = lambda x: r * x * (1-x)
    (orb,k,V0) = find_V0(logistic,x0,n,epsilon)
    if (k == -1):
        print("- For r = {:.2f}".format(r), "no period found smaller than",n)
    else:
        (V10,err) = error(logistic,V0)
        print("- For x0 = {:.1f}".format(x0), "we get k =", k, "and V10 = ",
              [round(x,3) for x in V10], "+- {:.1g}".format(err))
#%%
"""
Error
"""
r = 3.569
x0 = 0.5 

narr = [100, 1000, 10000]
epsilon = 1e-5

logistic = lambda x: r * x * (1-x)

for n in narr:
    (orb,k,V0) = find_V0(logistic,x0,n,epsilon)
    if (k == -1):
        print("- For n =",n, "no period found")
    else:
        (V10,err) = error(logistic,V0)
        print("- For n =",n, "we get k =", k, "and error = {:.1g}".format(err))  

