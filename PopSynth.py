from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
import numpy as np

BSEDict = {'xi': 1.0, 'bhflag': 2, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1, 'rtmsflag' : 0}

# Set the initial parameters
m10 = 85.543645
m20 = 84.99784
m30 = 5.543645
sep30 = 9e3 # solar radii
ecc3 = 0.

# Initiate inner binary and tertiary companion
single_binary   = InitialBinaryTable.InitialBinaries(m1=m10, m2=m20, porb=446.795757, ecc=0.448872, tphysf=13700.0, kstar1=1, kstar2=1, metallicity=0.002)
single_tertiary = InitialBinaryTable.InitialBinaries(m1=m30, m2=0.1, porb=1e12, ecc=0., tphysf=13700.0, kstar1=1, kstar2=14, metallicity=0.002)

# Evolve tertiary
bpp_tertiary, bcm_tertiary, initC_tertiary, kick_info_tertiary = Evolve.evolve(initialbinarytable=single_tertiary, BSEDict=BSEDict, dtp=1)

# Evolve inner binary
bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=single_binary, BSEDict=BSEDict, dtp=1)

# Set tertiary parameters
bcm['ecc3'] = ecc3
bcm['mass_3'] = bcm_tertiary['mass_1']
bcm['kstar_3'] = bcm_tertiary['kstar_1']
bcm['rad_3'] = bcm_tertiary['rad_1']
bcm['q_out'] = bcm['mass_3']/(bcm['mass_1']+bcm['mass_2'])
bcm['sep3'] = sep30*(m10+m20+m30)/(bcm['mass_1']+bcm['mass_2']+bcm['mass_3'])
bcm['RRLO_3'] = bcm['rad_3']/(.49*bcm['q_out']**(2/3)/(.6*bcm['q_out']**(2/3)+np.log(1+bcm['q_out']**(1/3))))/bcm['sep3']/(1-ecc3)
bcm['Stability'] = bcm['sep3']*(1-ecc3)/bcm['sep']/2.8/((1+bcm['mass_3']/(bcm['mass_1']+bcm['mass_2'])*(1+ecc3)/np.sqrt(1-ecc3)))**(2/5)

# Check for tertiary RLO
if(len(bcm[bcm['RRLO_3']>1])>0):
    print("Tertiary RLO")

# Check for dyn. instability
if(len(bcm[bcm['Stability']<1])>0):
    print("System is not hierarchically stable")

# Check for systems that form a stellar merger, isolated DCO, and inner DCO that remains stable and not tertiary RLO-filling
Stability = (len(bcm[bcm['Stability']<1])==0)
Detached = (len(bcm[bcm['RRLO_3']>1])==0)
LowMass = (len(bcm[bcm['kstar_3']>=13])==0)

SMe = bcm[(bcm['bin_state']==1) & (bcm['merger_type']!='1414') & (bcm['merger_type']!='1313') & 
          (bcm['merger_type']!='1314') & (bcm['merger_type']!='1413') & Stability & Detached]
DCO = bcm[(bcm['kstar_1']>=13) & (bcm['kstar_2']>=13) &
          (bcm['kstar_1']<=14) & (bcm['kstar_2']<=14) & (bcm['bin_state']==0)]
TCO = bcm[(bcm['kstar_1']>=13) & (bcm['kstar_2']>=13) & (bcm['kstar_3']>=13) &
          (bcm['kstar_1']<=14) & (bcm['kstar_2']<=14) & (bcm['kstar_3']<=14) & 
          Stability & Detached & (bcm['bin_state']==0)]
DCOLowMass = bcm[(bcm['kstar_1']>=13) & (bcm['kstar_2']>=13) &
          (bcm['kstar_1']<=14) & (bcm['kstar_2']<=14) & 
          LowMass & Stability & Detached & (bcm['bin_state']==0)]

if(len(SMe)>0):
    print("A stellar merger occurs:")
    print(SMe[['tphys','kstar_1','kstar_2','mass_1','mass_2','sep','ecc','bin_state']].iloc[0])

if(len(DCO)>0):
    print("A DCO would have been formed if it was in isolation:")
    print(DCO[['tphys','kstar_1','kstar_2','mass_1','mass_2','sep','ecc','bin_state']].iloc[0])

if(len(TCO)>0):
    print("Triple CO has been formed:")
    print(TCO[['tphys','kstar_1','kstar_2','kstar_3','mass_1','mass_2','mass_3','sep','sep3','ecc','ecc3','bin_state']].iloc[0])

if(len(DCOLowMass)>0):
    print("Inner DCO plus low mass companion have been formed:")
    print(DCOLowMass[['tphys','kstar_1','kstar_2','kstar_3','mass_1','mass_2','mass_3','sep','sep3','ecc','ecc3','bin_state']].iloc[0])

print(kick_info)

# Get the kick times, and their order
t1 = bpp[bpp['evol_type']==15]['tphys']
t2 = bpp[bpp['evol_type']==16]['tphys']
t3 = bpp_tertiary[bpp_tertiary['evol_type']==15]['tphys']
tSN = np.array([])
iSN = np.array([])
if(len(t1)>0):
    tSN = np.append(tSN,t1)
    iSN = np.append(iSN,1)
if(len(t2)>0):
    tSN = np.append(tSN,t2)
    iSN = np.append(iSN,2)
if(len(t3)>0):
    tSN = np.append(tSN,t3)
    iSN = np.append(iSN,3)
iSN = iSN[np.argsort(tSN)]
tSN = np.sort(tSN)

# Apply the kicks
for i in iSN:
    if(i<=2): # SN occured in the inner binary
        print("...")
