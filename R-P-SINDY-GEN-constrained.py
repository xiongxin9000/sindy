import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt

model=ps.SINDy()
print(model)
########################
#pysindy.differentiation: X'
#ps.optimizers:coffecient
#ps.feature_library:basic function 
#########################
#import data from data.npy
data = np.loadtxt("pressure_100_101_0.31_15_VSM.dat")
data_lbm=data[:250,2]
R=1/data_lbm

R1 = R
R2 = model.differentiate(R1)
t = np.arange(0, 250, 1)
# print(len(t))
x = np.column_stack((R1, R2))
# print(x.shape)
opt = ps.STLSQ(threshold=0, fit_intercept=False)
# fourier_library=ps.FourierLibrary(n_frequencies=2)
# model=ps.SINDy(feature_library=fourier_library,feature_names=["R1","R2"],optimizer=opt)

model=ps.SINDy(feature_names=["R1","R2"],optimizer=opt)
model.fit(x=x,t=t)
print('polinomial')
model.print()
R_model=model.simulate(x[0,:],t)
fig,ax= plt.subplots(1,2,figsize=(12,6))
ax[0].plot(R,label="lbm")
ax[0].plot(R_model[:,0],'--',label="SINdy")
ax[0].set_title("polinomial library")
ax[0].set(xlabel="t", ylabel="R")
ax[0].legend()
# plt.show()

#customized library
#1/R1,R2/R1,R2^2,R1^2*R2^2,R2^2/R1,R1/ln(R1)
# library_functions=[lambda x:x,
#                    lambda x: 1.0/x,
#                    lambda x,y: y/x,
#                    lambda y:y**2,
#                    lambda x,y:x**2*y**2,
#                    lambda x,y:y**2/x,
#                    lambda x: x/np.log(x)]
# library_functions_names=[lambda x:x,
#                         lambda x: '1.0/'+x, 
#                         lambda x,y: y+'/'+x,
#                         lambda y:y+'^2',
#                         lambda x,y:x+'^2*'+y+'^2',
#                         lambda x,y:y+'^2/'+x,
#                         lambda x: x+'ln('+x+')']
#see each individual term

# library_functions=[lambda x:x,
#                    lambda x: 1.0/x,
#                    lambda x,y: y/x,
#                    lambda y:y**2,
#                    lambda x,y:x**2*y**2,
#                    lambda x:x*np.log(1.0/x),
#                    lambda x,y: y**2*np.log(1.0/x)]
# library_functions_names=[
#                         lambda x:x,
#                         lambda x: '1.0/'+x, 
#                         lambda x,y: y+'/'+x,
#                         lambda y:y+'^2',
#                         lambda x,y:x+'^2*'+y+'^2',
#                         lambda x:x+'^2/'+x,
#                         lambda x,y: x+'ln('+x+')']
# R-P
library_functions=[
                    lambda x:x,
                   lambda x: 1.0/(x*np.log(1.0/x)),
                   lambda x: 1.0/(x**2*np.log(1.0/x)),
                   lambda x,y:y/(x**2*np.log(1.0/x)),
                   lambda x,y:y**2/(x*np.log(1.0/x)),
                   lambda x,y:x*y**2/(np.log(1.0/x)),
                   lambda x,y: y**2/x
                   ]
library_functions_names=[
                        lambda x: x,
                        lambda x: '1/('+x+'*ln(1/'+x+'))',
                        lambda x: '1/('+x+'^2*ln('+'1/'+x+'))',
                        lambda x,y:y+'/('+x+'^2*ln('+'1/'+x+'))',
                        lambda x,y:y+'^2/('+x+'*ln('+'1/'+x+'))',
                        lambda x,y:x+'*'+y+'^2/(ln('+'1/'+x+'))',
                        lambda x,y: y+'^2/'+x]
#reproduce the polynomial
# library_functions=[lambda x:1,
#                    lambda x:x,
#                    lambda x,y: x*y,
#                    lambda y:y**2]
# library_functions_names=[lambda x:'',
#                         lambda x:x,
#                         lambda x,y: y+x,
#                         lambda y:y+'^2']
pol_lib=ps.PolynomialLibrary()

custom_lib=ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_functions_names,
)
####################################################
custom_lib.fit(x)
# n_features=custom_lib.n_features_in_
n_features=10
n_targets=x.shape[1]
constraint_rhs=np.asarray([0, 0])
constraint_lhs=np.zeros((n_targets, n_targets * n_features))

constraint_lhs[0,2]=1
constraint_lhs[0,1]=1

constraint_lhs[1,2]=1
constraint_lhs[1,1]=1

opt=ps.ConstrainedSR3(
    constraint_lhs=constraint_lhs,
    constraint_rhs=constraint_rhs,
    threshold=0.5,
    thresholder='l1'
)
#################################################
model=ps.SINDy(
    feature_names=["R1","R2"],
    feature_library=custom_lib,
    optimizer=opt)
model.fit(x=x,t=t)
print('customized')
model.print()
R_model=model.simulate(x[0,:],t)
# fig,ax= plt.subplots(1,2,figsize=(6,6))
ax[1].plot(R,label="lbm")
ax[1].plot(R_model[:,0],'--',label="SINdy")
ax[1].set(xlabel="t", ylabel="R")
ax[1].set_title("customized library")
ax[1].legend()
plt.show()