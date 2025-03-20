##### THE UNITS ARE PROBABLY WRONG!!!!!!

from ase.io.lammpsdata import read_lammps_data
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np
import math
import scipy.constants as const
from scipy.integrate import quad
import matplotlib.pyplot as plt
### IT'S TIME TO THINK ABOUT UNITS!!!!

def fc_matrix_is_symmetric():
    matrix, R_vectors = compute_force_constant_matrix(atoms)
    norm_diff = np.linalg.norm(matrix - matrix.T, 'fro') #Frobenius norm
    if norm_diff < 1e-6:
        print("The force constant matrix is symmetric!")
    else:
        print("The force constant matrix is not symmetric!")

def compute_force_constant_matrix(atoms):
    num_atoms = len(atoms)
    delta_u = 1e-6  # angstroms
    force_constants = np.zeros([num_atoms, 3, num_atoms, 3])  # 4D matrix
    initial_positions = atoms.get_positions().copy()

    R_vectors = np.zeros((num_atoms, num_atoms, 3))  # Initialize an (N, N, 3) array
    for i in range(num_atoms):
        for j in range(num_atoms):
            # The relative vector from atom i to atom j
            R_vectors[i, j, :] = atoms.get_distance(i, j, vector=True, mic=True) # mic (min image condition) implies periodic boundary condition

    # Disturb j,beta calculate the response of i,alpha (force)
    for j in range(num_atoms):
        for beta in range(3):
            atoms.positions[j, beta] += delta_u
            forces_plus = atoms.get_forces()
            atoms.positions[j, beta] -= 2 * delta_u
            forces_minus = atoms.get_forces()

            for i in range(num_atoms):
                for alpha in range(3):
                    force_constants[j, beta, i, alpha] = - (forces_plus[i, alpha] - forces_minus[i, alpha]) / (
                                2 * delta_u)

            atoms.positions[:, :] = initial_positions

    force_constants = force_constants.reshape(3 * num_atoms, 3 * num_atoms) # 3N * 3N matrix
    return force_constants, R_vectors

def dynamical_matrix_with_R(q, fc_matrix, R_vectors, atoms):
    masses = atoms.get_masses()  # array of length N
    num_atoms = len(atoms)
    Dq = np.zeros((3 * num_atoms, 3 * num_atoms), dtype=complex)

    # force on atom i and displacement on atom j.
    for i in range(num_atoms):
        for j in range(num_atoms):
            R_ij = R_vectors[i, j]  # r_j - r_i
            phase = np.exp(-1j * np.dot(q, R_ij))
            phi_ij = fc_matrix[3*j : 3*j+3, 3*i : 3*i+3]
            #Dq[3*j : 3*j+3, 3*i : 3*i +3] = (phi_ij / np.sqrt(masses[i] * masses[j])) * phase
            Dq[3*i : 3*i+3, 3*j : 3*j+3] = (phi_ij / np.sqrt(masses[i] * masses[j])) * phase ## I did not understand the indexing here.
    return Dq

def plot_phonon_dispersion(atoms):
    # Define a q-path.
    Gamma = np.array([0.0, 0.0, 0.0])
    K = np.array([4 * math.pi / (3 * math.sqrt(3)), 0.0, 0.0])
    #M = np.array([math.pi / math.sqrt(3), math.pi / 3, 0.0])

    # Create a linear interpolation between these points.
    num_q = 200
    q_path = [Gamma + (K - Gamma) * t for t in np.linspace(0, 1, num_q)]
    force_constants, R_vectors = compute_force_constant_matrix(atoms)
    # Compute frequencies along the q_path.
    frequencies = []
    for q in q_path:
        Dq = dynamical_matrix_with_R(q, force_constants, R_vectors, atoms)
        eigvals = np.linalg.eigvals(Dq)
        eigvals = np.maximum(eigvals.real, 0)  ## Convert negative values to O. ???
        f = np.sort(np.sqrt(eigvals))
        frequencies.append(f)
    frequencies = np.array(frequencies)
    num_branch = frequencies.shape[1] #3N
    q_points = np.linspace(0, 1, num_q)

    plt.figure()
    for branch in range(num_branch):
        plt.plot(q_points, frequencies[:, branch], 'b-', linewidth=1)
    plt.xlabel('q')
    plt.ylabel('Phonon Frequency')  ## UNITS????
    plt.title('Phonon Dispersion')
    plt.grid(True)
    plt.savefig("phonon_dispersion_plot.png")
    plt.show()
    return frequencies, q_points

def find_wave_vectors():
    # lattice vectors, should I multiply by 1.42?
    a1 = np.array([3, 0, 0]) * 1.42
    a2 = np.array([0, 1.732, 0]) * 1.42
    a3 = np.array([0, 0, 20]) * 1.42
    lattice_vectors = [a1, a2, a3]
    # find reciprocal lattice vectors
    b1 = ( 2 * math.pi * (np.cross(lattice_vectors[1], lattice_vectors[2])) ) / (np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
    b2 = ( 2 * math.pi * (np.cross(lattice_vectors[2], lattice_vectors[0])) ) / (np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
    b3 = ( 2 * math.pi * (np.cross(lattice_vectors[0], lattice_vectors[1])) ) / (np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
    # use Monkhorst-Pack grid to find wave vector k.
    q = [] ## wave vectors
    Nq = 10 # adjust this.
    for n1 in range(Nq):
        for n2 in range(Nq):
            for n3 in range(Nq):
                # create fractional ccordinates
                c1 = (n1 - Nq / 2) / Nq
                c2 = (n2 - Nq / 2) / Nq
                c3 = (n3 - Nq / 2) / Nq
                q_vec = c1 * b1 + c2 * b2 + c3 * b3
                q.append(q_vec)
    return q

def compute_phonon_relaxation_time(phonon_frequency): # By Mathisen's rule
    # some specific values taken from "the article".
    a = 1.42 #angstrom ## lattice constant
    p = 0.9 # surface roughness
    l = a * math.sqrt(3) / 2 # the smallest dimension of sheet
    v = 184 # 184 angstrom/ps ## phonon velocity along LA
    tau_boundary = (1 + p * l) / (1 - p * v)
    ############################################################################
    A_d = 4.5e-4 # defect concentration factor
    debye_freq = 2.66e14 #Hz ## along LA
    tau_defect = debye_freq ** 2 / (A_d * 2 * math.pi * phonon_frequency ** 3)
    #############################################################################
    gruneisen_parameter = 1.8 #along LA
    T = 300 # K
    m = 12.011
    kB = const.k
    tau_umklapp = (m * v**2 * debye_freq) / (2 * gruneisen_parameter**2 * kB * T * phonon_frequency**2)
    inv_tau = (1 / tau_boundary) + (1 / tau_defect) + (1 / tau_umklapp)
    tau = 1 / inv_tau
    return tau

def compute_phonon_relaxation_time_list(atoms):
    phonon_freq_list = compute_phonon_frequencies(atoms)
    phonon_relaxation_times = []
    for i in range(len(phonon_freq_list)):
        tau = compute_phonon_relaxation_time(phonon_freq_list[i])
        phonon_relaxation_times.append(tau)
    return phonon_relaxation_times

def compute_thermal_conductivity(atoms):
    mass = 12.011
    group_velocity = 184 #angstrom/ps ## LA
    phase_velocity = 184 #LA
    omega_max = 2.66e14 #Hz  # debye frequency ##LA
    L_z = 3.35 #angstroms ## The article took this parameter differently!!!
    kB = const.k
    hbar = const.hbar
    gruneisen_parameter = 1.8  # along LA
    debye_freq = 2.66e14
    T = 300
    beta = 1 / (kB * T)
    coeff = 1 / (4 * math.pi * L_z * kB * T**2)
    phonon_frequencies = compute_phonon_frequencies(atoms)
    phonon_relaxation_times = compute_phonon_relaxation_time_list(atoms)

    sum = 0
    for i in range(len(phonon_frequencies)):
        integrand = ( ( (hbar * phonon_frequencies[i])**2 * (math.e ** (hbar * phonon_frequencies[i] * beta)) ) / ( (math.e ** (hbar* phonon_frequencies[i] * beta) - 1) ** 2)  ) * \
                    phonon_relaxation_times[i] * (group_velocity/phase_velocity) * phonon_frequencies[i]
        ## Klemen's derivation
        omega_min = (group_velocity**3 * mass * debye_freq) / (2 * gruneisen_parameter**2 * kB * T * phonon_frequencies[i]**2)
        result = quad(integrand, omega_min, omega_max)
        sum += result
    kappa = coeff * sum
    return kappa

atoms = read_lammps_data('graphene.dat', atom_style='atomic')

cmds = [
    "pair_style airebo 3.0",
    "pair_coeff * * CH.airebo C"
]

lammps_calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps.log')
atoms.calc = lammps_calc


























