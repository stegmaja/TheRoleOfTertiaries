import numpy as np
import h5py as h5
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from kick import apply_kick_vectors

ID_file = 2

binary_directory = "COMPAS_Output_8/"
tertiary_directory = "COMPAS_Output_5/"

binary_file = binary_directory + "Detailed_Output/BSE_Detailed_Output_" + str(ID_file) + ".h5"
tertiary_file = tertiary_directory + "Detailed_Output/SSE_Detailed_Output_" + str(ID_file) + ".h5"

binary_data = h5.File(binary_file, 'r')
tertiary_data = h5.File(tertiary_file, 'r')
binary_supernova_data = h5.File(binary_directory + "COMPAS_Output.h5", 'r')['BSE_Supernovae']
tertiary_supernova_data = h5.File(tertiary_directory + "COMPAS_Output.h5", 'r')['SSE_Supernovae']


# Set the initial outer orbital parameters
sep_out_0 = 90e3 # solar radii
ecc_out_0 = 0.1
jv = np.array([0,0,1])*np.sqrt(1-ecc_out_0**2)
ev = np.array([1,0,0])*ecc_out_0

# Read binary properties
binary_Time = np.array(binary_data['Time'])
binary_Time,unique_ind = np.unique(binary_Time,return_index=True)
binary_sep = np.array(binary_data['SemiMajorAxis'])[unique_ind]
binary_unbound = np.array(binary_data['Unbound'])[unique_ind]
binary_Mass1 = np.array(binary_data['Mass(1)'])[unique_ind]
binary_Mass2 = np.array(binary_data['Mass(2)'])[unique_ind]
binary_Stellar_Type1 = np.array(binary_data['Stellar_Type(1)'])[unique_ind]
binary_Stellar_Type2 = np.array(binary_data['Stellar_Type(2)'])[unique_ind]
binary_Mass1_0 = binary_Mass1[0]
binary_Mass2_0 = binary_Mass2[0]

# If evolution was not computed until t=13700 do so
if(max(binary_Time)<13700):
    binary_Time = np.append(binary_Time,13700)
    binary_sep = np.append(binary_sep,binary_sep[-1]) # not a problem that there could be a GW inspiral
    binary_unbound = np.append(binary_unbound,binary_unbound[-1])
    binary_Mass1 = np.append(binary_Mass1,binary_Mass1[-1])
    binary_Mass2 = np.append(binary_Mass2,binary_Mass2[-1])
    binary_Stellar_Type1 = np.append(binary_Stellar_Type1,binary_Stellar_Type1[-1])
    binary_Stellar_Type2 = np.append(binary_Stellar_Type2,binary_Stellar_Type2[-1])

# Interpolate the tertiary properties, at the times given by the binary file
tertiary_Time = np.array(tertiary_data['Time'])
tertiary_Mass = np.array(tertiary_data['Mass'])
tertiary_Radius = np.array(tertiary_data['Radius'])
tertiary_Stellar_Type = np.array(tertiary_data['Stellar_Type'])
if(tertiary_Time[0]!=0): # Account for the fact that sometimes detailed output does not start at t=0
    tertiary_Time = np.insert(tertiary_Time,0,0)
    tertiary_Mass = np.insert(tertiary_Mass,0,np.array(tertiary_data['Mass_0'])[0])
    tertiary_Radius = np.insert(tertiary_Radius,0,np.array(tertiary_data['Radius@ZAMS'])[0])
    tertiary_Stellar_Type = np.insert(tertiary_Stellar_Type,0,1)
tertiary_Mass = interp1d(tertiary_Time,tertiary_Mass,bounds_error=False,fill_value=tertiary_Mass[-1])
tertiary_Radius = interp1d(tertiary_Time,tertiary_Radius,bounds_error=False,fill_value=tertiary_Radius[-1])
tertiary_Stellar_Type = interp1d(tertiary_Time,tertiary_Stellar_Type,bounds_error=False,fill_value=tertiary_Stellar_Type[-1],kind='previous')
tertiary_Mass = tertiary_Mass(binary_Time)
tertiary_Radius = tertiary_Radius(binary_Time)
tertiary_Stellar_Type = tertiary_Stellar_Type(binary_Time)
tertiary_Mass_0 = tertiary_Mass[0]

triple_Mass = binary_Mass1+binary_Mass2+tertiary_Mass
triple_Mass_0 = binary_Mass1_0+binary_Mass2_0+tertiary_Mass_0

# Read supernova properties
binary_supernova_SEED = np.array(binary_supernova_data['SEED'])
tertiary_supernova_SEED = np.array(tertiary_supernova_data['SEED'])

binary_SEED = np.array(binary_data['SEED'])[0]
tertiary_SEED = np.array(tertiary_data['SEED'])[0]

binary_supernova_ids = np.where(binary_SEED==binary_supernova_SEED)[0]
tertiary_supernova_ids = np.where(tertiary_SEED==tertiary_supernova_SEED)[0]

binary_supernova_Time = np.array(binary_supernova_data['Time'])[binary_supernova_ids]
tertiary_supernova_Time = np.array(tertiary_supernova_data['Time'])[tertiary_supernova_ids]

binary_supernova_mass_cp = np.array(binary_supernova_data['Mass(CP)'])[binary_supernova_ids]
binary_supernova_mass_pre = np.array(binary_supernova_data['Mass_Total@CO(SN)'])[binary_supernova_ids]
binary_supernova_mass_post = np.array(binary_supernova_data['Mass(SN)'])[binary_supernova_ids]

tertiary_supernova_mass_cp = np.interp(tertiary_supernova_Time, binary_Time, binary_Mass1+binary_Mass2)
tertiary_supernova_mass_pre = np.array(tertiary_supernova_data['Mass_Total@CO'])[tertiary_supernova_ids]
tertiary_supernova_mass_post = np.array(tertiary_supernova_data['Mass'])[tertiary_supernova_ids]

binary_supernova_SystemicSpeed_X = np.array(binary_supernova_data['SystemicSpeed_Vector>SN_X'])[binary_supernova_ids]
binary_supernova_SystemicSpeed_Y = np.array(binary_supernova_data['SystemicSpeed_Vector>SN_Y'])[binary_supernova_ids]
binary_supernova_SystemicSpeed_Z = np.array(binary_supernova_data['SystemicSpeed_Vector>SN_Z'])[binary_supernova_ids]

if(len(binary_supernova_ids)==2): # Make it the change in systemic speed
    binary_supernova_SystemicSpeed_X[1] -= binary_supernova_SystemicSpeed_X[0]
    binary_supernova_SystemicSpeed_Y[1] -= binary_supernova_SystemicSpeed_Y[0]
    binary_supernova_SystemicSpeed_Z[1] -= binary_supernova_SystemicSpeed_Z[0]

tertiary_kick_direction = np.random.uniform(size=3)
tertiary_kick_direction /= np.linalg.norm(tertiary_kick_direction) # make it a unit vector
tertiary_supernova_SystemicSpeed = np.array(tertiary_supernova_data['Drawn_Kick_Magnitude'])[tertiary_supernova_ids]
tertiary_supernova_SystemicSpeed_X = tertiary_supernova_SystemicSpeed*tertiary_kick_direction[0]
tertiary_supernova_SystemicSpeed_Y = tertiary_supernova_SystemicSpeed*tertiary_kick_direction[1]
tertiary_supernova_SystemicSpeed_Z = tertiary_supernova_SystemicSpeed*tertiary_kick_direction[2]

binary_supernova_Type = np.zeros(len(binary_supernova_ids))
tertiary_supernova_Type = np.ones(len(tertiary_supernova_ids))

triple_supernova_Time = np.append(binary_supernova_Time,tertiary_supernova_Time)
triple_supernova_mass_cp = np.append(binary_supernova_mass_cp,tertiary_supernova_mass_cp)
triple_supernova_mass_pre = np.append(binary_supernova_mass_pre,tertiary_supernova_mass_pre)
triple_supernova_mass_post = np.append(binary_supernova_mass_post,tertiary_supernova_mass_post)
triple_supernova_SystemicSpeed_X = np.append(binary_supernova_SystemicSpeed_X,tertiary_supernova_SystemicSpeed_X)
triple_supernova_SystemicSpeed_Y = np.append(binary_supernova_SystemicSpeed_Y,tertiary_supernova_SystemicSpeed_Y)
triple_supernova_SystemicSpeed_Z = np.append(binary_supernova_SystemicSpeed_Z,tertiary_supernova_SystemicSpeed_Z)
triple_supernova_Type = np.append(binary_supernova_Type,tertiary_supernova_Type)

triple_supernova_Time_order = np.argsort(triple_supernova_Time)
triple_supernova_Time = triple_supernova_Time[triple_supernova_Time_order]
triple_supernova_mass_cp = triple_supernova_mass_cp[triple_supernova_Time_order]
triple_supernova_mass_pre = triple_supernova_mass_pre[triple_supernova_Time_order]
triple_supernova_mass_post = triple_supernova_mass_post[triple_supernova_Time_order]
triple_supernova_SystemicSpeed_X = triple_supernova_SystemicSpeed_X[triple_supernova_Time_order]
triple_supernova_SystemicSpeed_Y = triple_supernova_SystemicSpeed_Y[triple_supernova_Time_order]
triple_supernova_SystemicSpeed_Z = triple_supernova_SystemicSpeed_Z[triple_supernova_Time_order]
triple_supernova_Type = triple_supernova_Type[triple_supernova_Time_order]

if(len(binary_supernova_ids)==0):
    print("No supernova in the inner binary")
else:
    print(str(len(binary_supernova_ids)) + " supernova(e) in the inner binary at",binary_supernova_Time,"Myr")

if(len(tertiary_supernova_ids)==0):
    print("No tertiary supernova")
else:
    print("Tertiary supernova at",tertiary_supernova_Time,"Myr")

# Apply kicks
sep_out = np.full_like(binary_Time, np.nan, dtype=np.double)
ecc_out = np.full_like(binary_Time, np.nan, dtype=np.double)

if(len(triple_supernova_Time)==0):
    triple_Mass_0 = triple_Mass[0]
    sep_out = sep_out_0*triple_Mass_0/triple_Mass # Wind
    ecc_out = ecc_out_0*np.ones_like(ecc_out)

if(len(triple_supernova_Time)>=1):
    mask = (binary_Time<=triple_supernova_Time[0])
    triple_Mass_0 = triple_Mass[mask][0]
    sep_out[mask] = sep_out_0*triple_Mass_0/triple_Mass[mask] # Wind
    ecc_out[mask] = ecc_out_0*np.ones(np.sum(mask))
    vk = np.array([triple_supernova_SystemicSpeed_X[0],
                   triple_supernova_SystemicSpeed_Y[0],
                   triple_supernova_SystemicSpeed_Z[0]])
    sep_out_0,ev,jv = apply_kick_vectors(sep_out[mask][-1],ev,jv,vk,
                                   triple_supernova_mass_pre[0],
                                   triple_supernova_mass_post[0],
                                   triple_supernova_mass_cp[0])
    ecc_out_0 = np.linalg.norm(ev)
    if((ecc_out_0<0) or (ecc_out_0>=1) or (sep_out_0<=0)):
        print("Outer binary disrupted")
        exit()

if(len(triple_supernova_Time)>=2):
    mask = (binary_Time>triple_supernova_Time[0]) & (binary_Time<=triple_supernova_Time[1])
    triple_Mass_0 = triple_Mass[mask][0]
    sep_out[mask] = sep_out_0*triple_Mass_0/triple_Mass[mask] # Wind
    ecc_out[mask] = ecc_out_0*np.ones(np.sum(mask))
    vk = np.array([triple_supernova_SystemicSpeed_X[1],
                   triple_supernova_SystemicSpeed_Y[1],
                   triple_supernova_SystemicSpeed_Z[1]])
    sep_out_0,ev,jv = apply_kick_vectors(sep_out[mask][-1],ev,jv,vk,
                                   triple_supernova_mass_pre[1],
                                   triple_supernova_mass_post[1],
                                   triple_supernova_mass_cp[1])
    ecc_out_0 = np.linalg.norm(ev)
    if((ecc_out_0<0) or (ecc_out_0>=1) or (sep_out_0<=0)):
        print("Outer binary disrupted")
        exit()

if(len(triple_supernova_Time)>=3):
    mask = (binary_Time>triple_supernova_Time[1]) & (binary_Time<=triple_supernova_Time[2])
    print('mask',np.sum(mask))
    triple_Mass_0 = triple_Mass[mask][0]
    sep_out[mask] = sep_out_0*triple_Mass_0/triple_Mass[mask] # Wind
    ecc_out[mask] = ecc_out_0*np.ones(np.sum(mask))
    vk = np.array([triple_supernova_SystemicSpeed_X[2],
                   triple_supernova_SystemicSpeed_Y[2],
                   triple_supernova_SystemicSpeed_Z[2]])
    sep_out_0,ev,jv = apply_kick_vectors(sep_out[mask][-1],ev,jv,vk,
                                   triple_supernova_mass_pre[2],
                                   triple_supernova_mass_post[2],
                                   triple_supernova_mass_cp[2])
    ecc_out_0 = np.linalg.norm(ev)
    if((ecc_out_0<0) or (ecc_out_0>=1) or (sep_out_0<=0)):
        print("Outer binary disrupted")
        exit()

if(len(triple_supernova_Time)!=0):
    mask = np.isnan(sep_out)
    triple_Mass_0 = triple_Mass[mask][0]
    sep_out[mask] = sep_out_0*triple_Mass_0/triple_Mass[mask] # Wind
    ecc_out[mask] = ecc_out_0*np.ones(np.sum(mask))

# Triple parameters
q_out = tertiary_Mass/(binary_Mass1+binary_Mass2)
tertiary_RRLO = tertiary_Radius/(.49*q_out**(2/3)/(.6*q_out**(2/3)+np.log(1+q_out**(1/3))))/sep_out/(1-ecc_out)
Stability = sep_out*(1-ecc_out)/binary_sep/2.8/((1+tertiary_Mass/(binary_Mass1+binary_Mass2)*(1+ecc_out)/np.sqrt(1-ecc_out)))**(2/5)

# Check for tertiary RLO
tertiary_Detached = True
if(sum(tertiary_RRLO>1)>0):
    print("Tertiary RLO")
    tertiary_Detached = False

# Check for dyn. instability
Stable = True
if(sum(Stability<1)>0):
    print("System is not hierarchically stable")
    Stable = False

# Check if the inner binary remains bound
binary_bound = True
if(sum(binary_unbound==1)>0):
    print("Inner binary unbinds")
    binary_bound = False

# Check if the outer binary remains bound
tertiary_bound = True
if(sum((sep_out<0) | (ecc_out<0) | (ecc_out>=1))>0):
    print("Outer binary unbinds")
    tertiary_bound = False
