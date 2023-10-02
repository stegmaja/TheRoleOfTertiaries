from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
import numpy as np

BSEDict = {'xi': 1.0, 'bhflag': 0, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1, 'rtmsflag' : 0}

m10 = 85.543645
m20 = 84.99784
m30 = 85.543645

# Initiate inner binary and tertiary companion
single_binary   = InitialBinaryTable.InitialBinaries(m1=m10, m2=m20, porb=446.795757, ecc=0.448872, tphysf=13700.0, kstar1=1, kstar2=1, metallicity=0.002)
single_tertiary = InitialBinaryTable.InitialBinaries(m1=m30, m2=0.1, porb=1e12, ecc=0., tphysf=13700.0, kstar1=1, kstar2=14, metallicity=0.002)

# Set the initial outer orbital parameters
sep30 = 3e3 # solar radii
ecc3 = 0.

# Evolve tertiary
bpp_tertiary, bcm_tertiary, initC_tertiary, kick_info_tertiary = Evolve.evolve(initialbinarytable=single_tertiary, BSEDict=BSEDict, dtp=1.)
print(bcm_tertiary)

# Evolve inner binary
bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=single_binary, BSEDict=BSEDict, dtp=1.)


# Wind-mass loss
bcm['mass_3'] = bcm_tertiary['mass_1']
bcm['sep3'] = sep30*(m10+m20+m30)/(bcm['mass_1']+bcm['mass_2']+bcm['mass_3'])

# Set tertiary parameters
bcm['rad_3'] = bcm_tertiary['rad_1']
bcm['q_out'] = bcm['mass_3']/(bcm['mass_1']+bcm['mass_2'])
bcm['RRLO_3'] = bcm['rad_3']/(.49*bcm['q_out']**(2/3)/(.6*bcm['q_out']**(2/3)+np.log(1+bcm['q_out']**(1/3))))/bcm['sep3']/(1-ecc3)

# Check for tertiary RLO
if(len(bcm[bcm['RRLO_3']>1])==0):
    print("No tertiary RLO")
else:
    print(bcm[bcm['RRLO_3']>1])

# Check for dyn. instability
bcm['Stability'] = bcm['sep3']*(1-ecc3)/bcm['sep']/2.8/((1+bcm['mass_3']/(bcm['mass_1']+bcm['mass_2'])*(1+ecc3)/np.sqrt(1-ecc3)))**(2/5)

if(len(bcm[bcm['Stability']<1])>0):
    print("System is not hierarchically stable")
    print(bcm[bcm['Stability']<1])
else:
    print("System remains hierarchically stable")


print(kick_info)