import numpy as np
# This script takes input parameters to generate correctness test

#File Locations
coordfile = "./tests/correctness/testin4_coordinate.txt"
massfile = "./tests/correctness/testin4_mass.txt"
output_file = "./tests/correctness/testout4.txt"

#Test Parameters
num_of_bodies = 52

mass_max = 12
mass_min = 0

position_max = 100
position_min = -102

eps =  1e-9

#Generate Arrays
rng = np.random.default_rng()

positions_array = rng.uniform(low=position_min, high=position_max, size=(num_of_bodies, 3))
mass_array = rng.uniform(low=mass_min, high=mass_max, size=(num_of_bodies, 1))

#Generate Output
accel_array = []
for i, posi in enumerate(positions_array):
    accel = np.array([0, 0, 0], dtype=float)
    for j, posj in enumerate(positions_array):
        rij = posj - posi
        accel += (mass_array[j]*rij)/((np.dot(rij, rij) + eps**2)**(3/2))

    accel_array.append(accel)

#Save Test
np.savetxt(coordfile, positions_array, fmt='%lf')
np.savetxt(massfile, mass_array, fmt='%lf')
np.savetxt(output_file, np.array(accel_array), fmt='%lf')
