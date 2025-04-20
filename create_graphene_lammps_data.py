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
region box block - 50 0 100 -50 50
create_box 1 box
mass 1 12.011
lattice custom 1.42 a1 3 0 0 a2 0 1.732 0 a3 0 0 20 &
basis 0 0 0 &
basis 0.333 0 0 &
basis 0.5 0.5 0 &
basis 0.833 0.5 0
variable x_0 equal xlo
variable x_f equal xhi
variable y_0 equal ylo
variable y_f equal yhi
variable x equal xhi-xlo
variable y equal yhi-ylo
variable thickness equal 3.35
variable V equal $x*$y*${{thickness}}
region graphene block ${{x_0}} ${{x_f}} ${{y_0}} ${{y_f}} -10 10 units box 
create_atoms 1 region graphene
mass 1 12.011 

pair_style airebo 3.0
pair_coeff * * CH.airebo C

write_data graphene_custom.dat
"""

lmp.commands_string(command)
lmp.close()
