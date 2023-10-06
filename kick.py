import numpy as np
import warnings
from scipy import optimize
import astropy.units as u
from astropy.constants import G

#
########################################################################
#
#     Supernova kicks with fallback in vectorial form
#
def apply_kick_vectors(a,ev,jv,vk,m1_pre,m1_post,m2):

    a *= u.AU
    vk *= u.km/u.s
    m1_pre *= u.Msun
    m1_post *= u.Msun
    m2 *= u.Msun

    e = np.linalg.norm(ev)
    print("%17s" % "Pre-SN:")
    print("%17s" % "Masses:","%8.4f" % float(m1_pre.value),"%8.4f" % float(m2.value),"Solar masses")
    print("%17s" % "SMA:","%8.4f" % float(a.value),"AU")
    print("%17s" % "Eccentricity:","%8.4f" % e)
    # Masses
    m12_pre = m1_pre+m2
    m12_post = m1_post+m2
    # Unit vectors
    u1 = ev/e
    u3 = jv/np.linalg.norm(jv)
    u2 = np.cross(u3,u1)
    # Anomaly
    M = np.random.uniform(0.,2.*np.pi)
    E = optimize.root_scalar(lambda E : E-e*np.sin(E)-M, x0=0., x1=np.pi).root
    r = a*(1.-e*np.cos(E))
    f = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    L = np.sqrt((G*m12_pre*a)*(1-e**2))
    f_dot = L/r**2
    # Semilatus rectum
    p = (1-e**2)*a
    # Relative distance
    r = p/(1+e*np.cos(f))
    # Relative vector
    rv = r*(u1*np.cos(f)+u2*np.sin(f))
    # Relative velocity
    vv = p/(1+e*np.cos(f))**2*f_dot*(-u1*np.sin(f)+u2*(np.cos(f)+e))
    print("%17s" % "Relative vel.:","%8.4f" % float(np.sqrt(np.dot(vv,vv)).to(u.km/u.s).value),"kms",end="\n\n")
    # Apply kick
    vv += vk
    # New eccentricity vector
    ev = np.cross(vv,np.cross(rv,vv))/(G*m12_post)-rv/r
    e = np.linalg.norm(ev)
    # New AM vector
    hv = np.cross(rv,vv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jv = np.sqrt(1-e**2)*hv/np.linalg.norm(hv) # binary got disrupted
    # New SMA
    a = np.dot(hv,hv)/(G*m12_post*(1-e**2))
    # New COM velocity
    vs = m1_post/m12_post*vk-(m1_pre-m1_post)*m2/m12_pre/m12_post*(vv-vk)
    print("%17s" % "Post-SN:")
    print("%17s" % "Masses:","%8.4f" % float(m1_post.to(u.Msun).value),"%8.4f" % float(m2.to(u.Msun).value),"Solar masses")
    print("%17s" % "SMA:","%8.4f" % float(a.to(u.AU).value),"AU")
    print("%17s" % "Eccentricity:","%8.4f" % e)
    print("%17s" % "COM velocity:","%8.4f" % float(np.sqrt(np.dot(vs,vs)).to(u.km/u.s).value),"kms",end="\n\n")
    return a.to(u.AU).value,ev,jv