import meep as mp
import math
import numpy as np
import sys

mp.verbosity(0)
dpml = 1.0
dair = 1.0
dLCs = np.arange(0.1,8.1,0.1)

center_wl = 0.52 # in um
resolution = 50 # points per um

# from doi/10.1002/jsid.1011
n_0 = 1.508
initial_angles = np.array([0,np.pi/2])
delta_n = 0.179
half_pitch_size = 0.162 #0.342
epsilon_diag = mp.Matrix(mp.Vector3(n_0**2,0,0),
                         mp.Vector3(0,n_0**2,0),
                         mp.Vector3(0,0,(n_0+delta_n)**2))

nfreq = 100
df = 1
add_angle = float(sys.argv[1])/180*np.pi # in deg
tilted_angle = float(sys.argv[2])/180*np.pi # in deg
effective_pitch = half_pitch_size/np.cos(tilted_angle)
R_tilt = mp.Matrix(mp.Vector3(math.cos(tilted_angle),math.sin(tilted_angle),0),
                   mp.Vector3(-math.sin(tilted_angle),math.cos(tilted_angle),0),
                   mp.Vector3(0,0,1),
                  )
R_add = mp.Matrix(mp.Vector3(1,0,0),
                  mp.Vector3(0,math.cos(add_angle),math.sin(add_angle)),
                  mp.Vector3(0,-math.sin(add_angle),math.cos(add_angle)))

def phi(p):
    xx = 0.5*d_cell + p.x - dpml # x distant to beginning
    return xx/half_pitch_size*math.pi + initial_angle

def tilted_phi(p):
    xx = 0.5*d_cell + p.x - dpml # x distant to beginning
    return xx/effective_pitch*math.pi + initial_angle

def CLC(p):
    # rotation matrix for rotation around x axis
    Rx = mp.Matrix(mp.Vector3(1,0,0),
                   mp.Vector3(0,math.cos(phi(p)),math.sin(phi(p))),
                   mp.Vector3(0,-math.sin(phi(p)),math.cos(phi(p))))
    lc_epsilon = Rx * epsilon_diag * Rx.transpose()
    lc_epsilon = R_tilt * lc_epsilon * R_tilt.transpose()
    lc_epsilon = R_add * lc_epsilon * R_add.transpose()
    
    lc_epsilon_diag = mp.Vector3(lc_epsilon[0].x,lc_epsilon[1].y,lc_epsilon[2].z)
    lc_epsilon_offdiag = mp.Vector3(lc_epsilon[1].x,lc_epsilon[2].x,lc_epsilon[2].y)
    return mp.Medium(epsilon_diag=lc_epsilon_diag,epsilon_offdiag=lc_epsilon_offdiag)

def get_S(Ey,Ez):
    result = np.zeros(4)
    result[0] = np.square(np.abs(Ey)) + np.square(np.abs(Ez))
    result[1] = np.square(np.abs(Ey)) - np.square(np.abs(Ez))
    Eyz = Ey*np.conjugate(Ez)
    result[2] = 2*Eyz.real
    result[3] = -2*Eyz.imag
    return result

pml_layers = [mp.PML(dpml)]
g_source = mp.GaussianSource(frequency=1/center_wl,fwidth=df)

all_T = np.zeros((dLCs.shape[0],nfreq))
all_R = np.zeros_like(all_T)
all_S_T = np.zeros((dLCs.shape[0],nfreq,4))
all_S_R = np.zeros_like(all_S_T)

for i in range(dLCs.shape[0]):
    print(i)
    dLC = dLCs[i]
    d_cell = dpml + dair + dLC + dair + dpml
    cell = mp.Vector3(d_cell, 0, 0)
    sources = [mp.Source(g_source,
                         component=mp.Ez,
                         center=mp.Vector3(-0.5*d_cell+dpml,0,0))]
    fluxregion_T = mp.FluxRegion(center=mp.Vector3(0.5*dLC+0.5*dair,0,0), size=mp.Vector3(0,0,0))
    fluxregion_R = mp.FluxRegion(center=mp.Vector3(dpml+0.5*dair-0.5*d_cell,0,0), size=mp.Vector3(0,0,0))
    # ------------------ get the blank flux reference ------------------
    geometry = [mp.Block(center=mp.Vector3(),size=mp.Vector3(dLC,mp.inf,mp.inf),material=mp.Medium(epsilon=1))]
    sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    force_all_components=True
)
    blank_fluxobj = sim.add_flux(1/center_wl, df, nfreq, fluxregion_R)

    sim.run(until=1000)
    blank_flux = mp.get_fluxes(blank_fluxobj)
    blank_flux_data = sim.get_flux_data(blank_fluxobj)

    if i==0:
        flux_freqs = mp.get_flux_freqs(blank_fluxobj)
        wl =[]
        for j in range(nfreq):
            wl = np.append(wl, 1/flux_freqs[j])

    sim.reset_meep()

    # ------------------- get LC flux ------------------
    for n in range(initial_angles.shape[0]):
        initial_angle = initial_angles[n]
        geometry = [mp.Block(center=mp.Vector3(),size=mp.Vector3(dLC,mp.inf,mp.inf),material=CLC)]
        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=resolution,
            force_all_components=True
        )


        T_fluxobj = sim.add_flux(1/center_wl, df, nfreq, fluxregion_T)
        R_fluxobj = sim.add_flux(1/center_wl, df, nfreq, fluxregion_R)
        sim.load_minus_flux_data(R_fluxobj, blank_flux_data)
        sim.run(until=1000)
        T_flux = mp.get_fluxes(T_fluxobj)
        R_flux = mp.get_fluxes(R_fluxobj)

        
        for j in range(nfreq):
            all_T[i,j] += T_flux[j]/blank_flux[j]
            all_R[i,j] += -R_flux[j]/blank_flux[j]
            T_Ey = sim.get_dft_array(T_fluxobj,mp.Ey,j)
            T_Ez = sim.get_dft_array(T_fluxobj,mp.Ez,j)
            all_S_T[i,j] += get_S(T_Ey,T_Ez)
            R_Ey = sim.get_dft_array(R_fluxobj,mp.Ey,j)
            R_Ez = sim.get_dft_array(R_fluxobj,mp.Ez,j)
            all_S_R[i,j] += get_S(R_Ey,R_Ez)
            

        sim.reset_meep()

all_T = all_T/initial_angles.shape[0]
all_R = all_R/initial_angles.shape[0]
all_S_T = all_S_T/initial_angles.shape[0]
all_S_R = all_S_R/initial_angles.shape[0]

np.savez_compressed("%.3fpi.npz"%(add_angle/np.pi),
                    all_T,all_R,all_S_T,all_S_R,wl,dLCs)
