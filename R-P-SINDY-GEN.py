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

#R-P
library_functions=[
                    # lambda x:x,
                   lambda x: 1/(x*np.log(1/x)),
                   lambda x: 1/(x**2*np.log(1/x)),
                   lambda x,y:y/(x**2*np.log(1/x)),
                   lambda x,y:y**2/(x*np.log(1/x)),
                   lambda x,y:x*y**2/(np.log(1/x)),
                   lambda x,y: y**2/x
                   ]
library_functions_names=[
                        # lambda x: x,
                        lambda x: '1/('+x+'*ln(1/'+x+'))',
                        lambda x: '1/('+x+'^2*ln('+'1/'+x+'))',
                        lambda x,y:y+'/('+x+'^2*ln('+'1/'+x+'))',
                        lambda x,y:y+'^2/('+x+'*ln('+'1/'+x+'))',
                        lambda x,y:x+'*'+y+'^2/(ln('+'1/'+x+'))',
                        lambda x,y: y+'^2/'+x]
library_functions2=[
                    lambda x:x,
                   ]
library_functions_names2=[
                        lambda x:x]

# pol_lib=ps.PolynomialLibrary()
# library_functions3=[
#                     lambda x:1,
#                    ]
# library_functions_names3=[
#                         lambda x:'']
custom_lib=ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_functions_names,
)
custom_lib2=ps.CustomLibrary(
    library_functions=library_functions2,
    function_names=library_functions_names2,
)
# custom_lib3=ps.CustomLibrary(
#     library_functions=library_functions3,
#     function_names=library_functions_names3,
# )
inputs_temp = np.tile([0, 1], 2)
inputs_per_library=np.reshape(inputs_temp,(2,2))
inputs_per_library[0, 0] = 1
# inputs_per_library[2, 0] = 1
# inputs_per_library[2, 1] = 0
print(inputs_per_library)

# tensor_array=[[0,1,1],[1,0,1]]
gen_lib=ps.GeneralizedLibrary(
    [custom_lib2,custom_lib],
    # tensor_array=tensor_array,
    inputs_per_library=inputs_per_library
)
model=ps.SINDy(
    feature_names=["R1","R2"],
    feature_library=gen_lib,
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