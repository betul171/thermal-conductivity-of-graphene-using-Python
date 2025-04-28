from lammps import lammps

lmp = lammps()


command = f"""
units metal
dimension 3
boundary p p p 
atom_style atomic

dimension 3
boundary p p p 
atom_style atomic
region box block -15 30 0 25 -20 20
create_box 1 box
mass 1 12.011

lattice custom 1.42 a1 3 0 0 a2 0 1.732 0 a3 0 0 20 &
basis 0 0 0 &
basis 0.333 0 0 &
basis 0.5 0.5 0 &
basis 0.833 0.5 0

variable x_0 equal -1
variable x_f equal 15
variable y_0 equal ylo
variable y_f equal yhi
variable x equal ${{x_f}}-${{x_0}}
variable y equal ${{y_f}}-${{y_0}}
region graphene block ${{x_0}} ${{x_f}} ${{y_0}} ${{y_f}} -0.1 0.1 units box ## does z coordinates appropriate?
create_atoms 1 region graphene

pair_style airebo 3.0
pair_coeff * * CH.airebo C

write_data gnr.dat
"""

lmp.commands_string(command)
lmp.close()
