from ase.io.lammpsdata import read_lammps_data
from ase.calculators.lammpslib import LAMMPSlib
import my_functions as b
import numpy as np
import math

atoms = read_lammps_data('graphene.dat', atom_style='atomic')

cmds = [
    "pair_style airebo 3.0",
    "pair_coeff * * CH.airebo C"
]

lammps_calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps.log')
atoms.calc = lammps_calc

print(b.fc_matrix_is_symmetric())
b.plot_phonon_dispersion(atoms)



