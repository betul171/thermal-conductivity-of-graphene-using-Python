import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Number of q-points
num_q = 100

# Parameters in metal units:
a = 1.42  # lattice constant
p = 0.9   # surface roughness
A_d = 4.5e-4  # defect concentration factor
hbar = 6.582119569e-4  # eV*ps
kB = 8.617333262e-5    # eV/K
T = 300  # K
beta = 1 / (kB * T)
l_x = 100e4  # characteristic length
l_z = 3.35   # thickness in angstrom
mass = 12.011  # in amu
# Conversion factor for mass: 1 amu = 1.03643e-4 (eV*ps^2/angstrom^2)
amu_to_eV_ps2_per_A2 = 1.03643e-4
mass = mass * amu_to_eV_ps2_per_A2

gruneisen_param_dict = {
    'LA': 1.8,
    'TA': 1.6,
    'ZA': -1.2  # negative Grüneisen. Use its absolute value when needed.
}

debye_freq_dict = {
    'LA': 266,
    'TA': 238,
    'ZA': 132
}

# Global Fourier harmonic parameter
N_FOURIER = 10

# --- Global caches for expensive computations ---
_fc_matrix_cache = None
_R_vectors_cache = None
_dispersion_data_cache = None
_arc_length_cache = None
_acoustic_indices_cache = None

def get_force_constant_data(atoms):

    global _fc_matrix_cache, _R_vectors_cache
    if _fc_matrix_cache is None or _R_vectors_cache is None:
        _fc_matrix_cache, _R_vectors_cache = compute_force_constant_matrix(atoms)
    return _fc_matrix_cache, _R_vectors_cache

def get_dispersion_data(atoms):

    global _dispersion_data_cache
    if _dispersion_data_cache is None:
        _dispersion_data_cache = plot_phonon_dispersion(atoms)
    return _dispersion_data_cache

def get_q_arc_length(q_path):

    global _arc_length_cache
    if _arc_length_cache is None:
        _arc_length_cache = compute_q_arc_length(q_path)
    return _arc_length_cache

def fc_matrix_is_symmetric(atoms):
    matrix, R_vectors = compute_force_constant_matrix(atoms)
    norm_diff = np.linalg.norm(matrix - matrix.T, 'fro')
    if norm_diff < 1e-6:
        print("The force constant matrix is symmetric!")
    else:
        print("The force constant matrix is not symmetric!")

def compute_force_constant_matrix(atoms):
    num_atoms = len(atoms)
    delta_u = 1e-6  # displacement in angstrom
    force_constants = np.zeros((num_atoms, 3, num_atoms, 3))
    initial_positions = atoms.get_positions().copy()  # in angstrom
    R_vectors = np.zeros((num_atoms, num_atoms, 3))
    for i in range(num_atoms):
        for j in range(num_atoms):
            R_vectors[i, j, :] = atoms.get_distance(i, j, vector=True, mic=True)
    for j in range(num_atoms):
        for beta in range(3):
            atoms.positions[j, beta] += delta_u
            forces_plus = atoms.get_forces()  # expected in eV/angstrom
            atoms.positions[j, beta] -= 2 * delta_u
            forces_minus = atoms.get_forces()
            for i in range(num_atoms):
                for alpha in range(3):
                    force_constants[j, beta, i, alpha] = - (forces_plus[i, alpha] - forces_minus[i, alpha]) / (2 * delta_u)
            atoms.positions[:, :] = initial_positions
    fc_matrix = force_constants.reshape(3 * num_atoms, 3 * num_atoms)
    return fc_matrix, R_vectors

def dynamical_matrix_with_R(q, fc_matrix, R_vectors, atoms):
    masses = atoms.get_masses() * amu_to_eV_ps2_per_A2
    num_atoms = len(atoms)
    Dq = np.zeros((3 * num_atoms, 3 * num_atoms), dtype=complex)
    for i in range(num_atoms):
        for j in range(num_atoms):
            R_ij = R_vectors[i, j]
            phase = np.exp(-1j * np.dot(q, R_ij))
            phi_ij = fc_matrix[3*j : 3*j+3, 3*i : 3*i+3]
            Dq[3*i : 3*i+3, 3*j : 3*j+3] = (phi_ij / np.sqrt(masses[i] * masses[j])) * phase
    return Dq

def plot_phonon_dispersion(atoms):
    Gamma = np.array([0.0, 0.0, 0.0])
    K = np.array([4 * math.pi / (3 * math.sqrt(3)), 0.0, 0.0])
    q_points_norm = np.linspace(0, 1, num_q)
    q_path = [Gamma + (K - Gamma) * t for t in q_points_norm]
    q_path = np.array(q_path)
    fc_matrix, R_vectors = get_force_constant_data(atoms)
    frequencies = []
    for q in q_path:
        Dq = dynamical_matrix_with_R(q, fc_matrix, R_vectors, atoms)
        eigvals = np.linalg.eigvals(Dq)
        eigvals = np.maximum(eigvals.real, 0)
        f = np.sort(np.sqrt(eigvals))
        frequencies.append(f)
    frequencies = np.array(frequencies)
    return frequencies, q_points_norm, q_path

def compute_q_arc_length(q_path):
    arc = np.zeros(len(q_path))
    for i in range(1, len(q_path)):
        arc[i] = arc[i-1] + np.linalg.norm(q_path[i] - q_path[i-1])
    return arc

# Make this function computationally less expensive
def identify_indices_of_acoustic_branches(atoms, num_fit_points=3):

    global _acoustic_indices_cache
    if _acoustic_indices_cache is not None:
        return _acoustic_indices_cache

    frequencies, q_points, _ = get_dispersion_data(atoms)
    tol = 1e-3
    acoustic = np.where(frequencies[0, :] < tol)[0]
    if len(acoustic) < 3:
        print("Warning: fewer than three acoustic branches identified.")
        return {}
    slopes = {}
    for i in acoustic:
        pts = min(num_fit_points, len(q_points))
        q_fit = q_points[:pts]
        f_fit = frequencies[:pts, i]
        slope, _ = np.polyfit(q_fit, f_fit, 1)
        slopes[i] = slope
    sorted_modes = sorted(slopes, key=lambda i: abs(slopes[i]), reverse=True)
    mode_labels = {
        "LA": sorted_modes[0],
        "TA": sorted_modes[1],
        "ZA": min(sorted_modes[2:], key=lambda i: abs(slopes[i]))
    }
    _acoustic_indices_cache = mode_labels
    return mode_labels

def fourier_smooth_dispersion(q_points, frequencies_branch, N=N_FOURIER):
    num_q = len(q_points)
    X = np.ones((num_q, 2 * N + 1))
    for n in range(1, N + 1):
        X[:, 2 * n - 1] = np.cos(2 * np.pi * n * q_points)
        X[:, 2 * n] = np.sin(2 * np.pi * n * q_points)
    coeffs, _, _, _ = np.linalg.lstsq(X, frequencies_branch, rcond=None)
    freq_fit = X.dot(coeffs)
    X_deriv = np.zeros((num_q, 2 * N + 1))
    for n in range(1, N + 1):
        X_deriv[:, 2 * n - 1] = -2 * np.pi * n * np.sin(2 * np.pi * n * q_points)
        X_deriv[:, 2 * n] = 2 * np.pi * n * np.cos(2 * np.pi * n * q_points)
    freq_deriv = X_deriv.dot(coeffs)
    return freq_fit, freq_deriv

def compute_group_and_phase_velocities_fourier(atoms, N=N_FOURIER):
    frequencies, q_points_norm, q_path = get_dispersion_data(atoms)
    mode_labels = identify_indices_of_acoustic_branches(atoms)
    acoustic_labels = ["LA", "TA", "ZA"]
    q_actual = get_q_arc_length(q_path)
    v_group_dict = {}
    v_phase_dict = {}
    for label in acoustic_labels:
        branch_indx = mode_labels[label]
        freq_raw = frequencies[:, branch_indx]
        freq_smooth, group_velocity = fourier_smooth_dispersion(q_points_norm, freq_raw, N)
        phase_velocity = np.empty_like(q_actual)
        phase_velocity[1:] = freq_smooth[1:] / q_actual[1:]
        phase_velocity[0] = phase_velocity[1]
        v_group_dict[label] = group_velocity
        v_phase_dict[label] = phase_velocity
    return v_group_dict, v_phase_dict

def plot_group_velocities_fourier(atoms, N=N_FOURIER):
    v_group_dict, _ = compute_group_and_phase_velocities_fourier(atoms, N)
    _, q_points_norm, q_path = get_dispersion_data(atoms)
    q_actual = get_q_arc_length(q_path)
    plt.figure(figsize=(8, 6))
    for label in ["LA", "TA", "ZA"]:
        plt.plot(q_actual, v_group_dict[label], marker='o', label=f'{label} branch')
    plt.xlabel('q (1/angstrom) [arc-length]')
    plt.ylabel('Group Velocity (metal units)')
    plt.title('Phonon Group Velocities via Fourier Expansion')
    plt.legend()
    plt.grid(True)
    plt.savefig('group_velocities_fourier.png')
    plt.show()

def plot_dispersion_validation(atoms, branch_label, N=N_FOURIER):
    frequencies, q_points_norm, q_path = get_dispersion_data(atoms)
    mode_labels = identify_indices_of_acoustic_branches(atoms)
    branch_index = mode_labels[branch_label]
    raw_freq = frequencies[:, branch_index]
    freq_fit, freq_deriv = fourier_smooth_dispersion(q_points_norm, raw_freq, N)
    q_actual = get_q_arc_length(q_path)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(q_actual, raw_freq, 'o', label='Raw Data')
    plt.plot(q_actual, freq_fit, '-', label='Fourier Fit')
    plt.xlabel('q (1/angstrom) [arc-length]')
    plt.ylabel('Frequency (1/ps)')
    plt.title(f'Phonon Dispersion: {branch_label} Branch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(q_actual, freq_deriv, '-', label='Group Velocity')
    plt.xlabel('q (1/angstrom) [arc-length]')
    plt.ylabel('dω/dq (1/ps per 1/angstrom)')
    plt.title(f'Group Velocity: {branch_label} Branch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_dispersion_fit(atoms, branch_label, N=N_FOURIER):
    frequencies, q_points_norm, q_path = get_dispersion_data(atoms)
    mode_labels = identify_indices_of_acoustic_branches(atoms)
    branch_index = mode_labels[branch_label]
    raw_freq = frequencies[:, branch_index]
    freq_fit, freq_deriv = fourier_smooth_dispersion(q_points_norm, raw_freq, N)
    q_actual = get_q_arc_length(q_path)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(q_actual, raw_freq, 'o', label='Raw Data')
    plt.plot(q_actual, freq_fit, '-', label='Fourier Fit')
    plt.xlabel('q (1/angstrom) [arc-length]')
    plt.ylabel('Frequency (1/ps)')
    plt.title(f'Phonon Dispersion: {branch_label} Branch')
    plt.legend()
    plt.grid(True)
    rmse = np.sqrt(np.mean((raw_freq - freq_fit) ** 2))
    plt.subplot(1, 2, 2)
    plt.plot(q_actual, np.abs(raw_freq - freq_fit), 'r-', label='Absolute Error')
    plt.xlabel('q (1/angstrom) [arc-length]')
    plt.ylabel('Absolute Error (1/ps)')
    plt.title(f'RMSE = {rmse:.3e}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"RMSE for {branch_label} branch: {rmse:.3e} 1/ps")

def tau(omega, group_velocity, label):
    l = a * math.sqrt(3)  # angstrom
    inv_tau_boundary = (1 - p) / (1 + p) * abs(group_velocity) / l_x
    inv_tau_defect = (A_d * 2 * math.pi * omega**3) / (debye_freq_dict[label]**2)
    inv_tau_umklapp = (2 * (gruneisen_param_dict[label]**2) * kB * T * omega**2) / (mass * group_velocity**2 * debye_freq_dict[label])
    inv_tau = inv_tau_boundary + inv_tau_defect + inv_tau_umklapp
    epsilon = 1e-12
    inv_tau += epsilon
    return 1 / inv_tau

def integrand(omega, group_velocity, phase_velocity, label):
    x = hbar * omega * beta
    threshold = 350
    if x > threshold:
        fraction = (hbar * omega) ** 2 * np.exp(-x)
    else:
        exp_factor = np.exp(x)
        denom = (exp_factor - 1) ** 2
        fraction = (hbar * omega) ** 2 * exp_factor / (denom + 1e-30)
    abs_vg = abs(group_velocity)
    abs_vp = abs(phase_velocity)
    return fraction * tau(omega, abs_vg, label) * (abs_vg / abs_vp) * omega

def compute_thermal_conductivity(atoms):
    coeff = 1 / (4 * np.pi * l_z * kB * T**2)
    v_group_dict, v_phase_dict = compute_group_and_phase_velocities_fourier(atoms)
    acoustic_labels = ["LA", "TA", "ZA"]
    total_integral = 0
    for label in acoustic_labels:
        group_velocity = v_group_dict[label][1]
        phase_velocity = v_phase_dict[label][1]
        omega_max = debye_freq_dict[label]
        omega_min = (abs(group_velocity) / abs(gruneisen_param_dict[label])) * \
                    np.sqrt((mass * abs(group_velocity) * debye_freq_dict[label]) / (2 * kB * T * l_x))
        result_of_integral = quad(lambda omega: integrand(omega, group_velocity, phase_velocity, label),
                                  omega_min, omega_max, epsabs=1e-8, epsrel=1e-8, limit=1000)
        total_integral += result_of_integral[0]
    therm_cond = coeff * total_integral
    conversion_factor = 1.60218e3
    therm_cond_in_si = therm_cond * conversion_factor
    print("Thermal conductivity in SI:", therm_cond_in_si, "W/(m*K)")
    return therm_cond_in_si

##############################
# Example usage with ASE setup:
##############################
from ase.io.lammpsdata import read_lammps_data
from ase.calculators.lammpslib import LAMMPSlib

atoms = read_lammps_data('graphene.dat', atom_style='atomic')
cmds = [
    "pair_style airebo 3.0",
    "pair_coeff * * CH.airebo C"
]
lammps_calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps.log')
atoms.calc = lammps_calc

# Compute thermal conductivity.
compute_thermal_conductivity(atoms)
