import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar, h, e
from scipy import optimize
import sympy as sp
from scipy.linalg import eigh
import scipy


def _generate_qubit_params(L, LJ, CJ):
    EJ = (hbar / 2 / e) ** 2 / (LJ * h) / 1e9
    EL = (hbar / 2 / e) ** 2 / (L * h) / 1e9
    EC = e**2 / (2 * CJ * h) / 1e9

    return EL, EJ, EC


def generate_potential(phi, phi_g, L, LJ, CJ):
    EL, EJ, EC = _generate_qubit_params(L, LJ, CJ)
    return -EJ * np.cos(phiplot_values + pi) + 0.5 * EL * phiplot_values**2


# quantum operators
def create(n):
    return np.diag(np.sqrt(np.arange(1, n)), k=-1)


def destroy(n):
    return np.diag(np.sqrt(np.arange(1, n)), k=1)


##### Below this is double well analysis #####

# input parameters
# LJ = 6e-9
# L = 15.208e-9
# CJ = 100e-15
phig = pi
numdivisions = 1000000
phiplot_values = np.linspace(-6, 6, numdivisions)

# EJ = (hbar / 2 / e) ** 2 / (LJ * h) / 1e9
# EL = (hbar / 2 / e) ** 2 / (L * h) / 1e9
# EC = e**2 / (2 * CJ * h) / 1e9


def _finding_posminima(EJ, EL, EC, phig):
    phi_posmin = np.linspace(0, 6, numdivisions)

    def Upos(EJ, EL, phig):
        return -EJ * np.cos(phi_posmin + phig) + 0.5 * EL * phi_posmin**2

    posmin = np.argmin(Upos(EJ, EL, phig))
    posphi_star = phi_posmin[posmin]

    return posphi_star


def _finding_negminima(EJ, EL, EC, phig):
    phi_negmin = np.linspace(-6, 0, numdivisions)

    def Uneg(EJ, EL, phig):
        return -EJ * np.cos(phi_negmin + phig) + 0.5 * EL * phi_negmin**2

    negmin = np.argmin(Uneg(EJ, EL, phig))
    negphi_star = phi_negmin[negmin]

    return negphi_star


def find_minima(L, LJ, CJ, phig):
    EJ, EL, EC = _generate_qubit_params(L, LJ, CJ)
    posphi_star = _finding_posminima(EJ, EL, EC, phig)
    negphi_star = _finding_negminima(EJ, EL, EC, phig)

    return posphi_star, negphi_star


# posphi_star = finding_posminima(EJ, EL, EC, phig)
# negphi_star = finding_negminima(EJ, EL, EC, phig)


def compute_derivatives(phi, LJ, L, CJ, phi_g):
    U = generate_potential(phi, phi_g, L, LJ, CJ)
    der1 = U(phi, phi_g).diff(phi)
    der2 = der1.diff(phi)
    der3 = der2.diff(phi)
    der4 = der3.diff(phi)
    der5 = der4.diff(phi)
    der6 = der5.diff(phi)

    return der1, der2, der3, der4, der5, der6


# def evaluate_derivatives_at_min(derivates, min):
#     Adash = sp.lambdify(phi, der2)
#     A = Adash(posphi_star)
#     Bdash = sp.lambdify(phi, der3)
#     B = Bdash(posphi_star)
#     Cdash = sp.lambdify(phi, der4)
#     C = Cdash(posphi_star)
#     Ddash = sp.lambdify(phi, der5)
#     D = Ddash(posphi_star)
#     Edash = sp.lambdify(phi, der6)
#     E = Edash(posphi_star)


def finding_energy_levels_of_wells(LJ, L, CJ, phi_g, levels):
    # levels = 20
    # EJ = (hbar / 2 / e) ** 2 / (LJ * h) / 1e9
    # EL = (hbar / 2 / e) ** 2 / (L * h) / 1e9
    # EC = e**2 / (2 * CJ * h) / 1e9

    EJ, EL, EC = _generate_qubit_params(L, LJ, CJ)

    phi = sp.Symbol("phi")

    # def U(phi, phi_g):
    #     return -EJ * sp.cos(phi + phi_g) + 1 / 2 * EL * phi**2

    # finding derivatives of the potential

    # der1 = U(phi, phi_g).diff(phi)
    # der2 = der1.diff(phi)
    # der3 = der2.diff(phi)
    # der4 = der3.diff(phi)
    # der5 = der4.diff(phi)
    # der6 = der5.diff(phi)

    # computing higher order derivatives at minimum
    Adash = sp.lambdify(phi, der2)
    A = Adash(posphi_star)
    Bdash = sp.lambdify(phi, der3)
    B = Bdash(posphi_star)
    Cdash = sp.lambdify(phi, der4)
    C = Cdash(posphi_star)
    Ddash = sp.lambdify(phi, der5)
    D = Ddash(posphi_star)
    Edash = sp.lambdify(phi, der6)
    E = Edash(posphi_star)

    Adash1 = sp.lambdify(phi, der2)
    A1 = Adash1(negphi_star)
    Bdash1 = sp.lambdify(phi, der3)
    B1 = Bdash1(negphi_star)
    Cdash1 = sp.lambdify(phi, der4)
    C1 = Cdash1(negphi_star)
    Ddash1 = sp.lambdify(phi, der5)
    D1 = Ddash1(negphi_star)
    Edash1 = sp.lambdify(phi, der6)
    E1 = Edash1(negphi_star)

    # Taylor series around minima
    def UTaylorpositive(phi, phi_g):
        return (
            -EJ * sp.cos(posphi_star + phi_g)
            + 0.5 * EL * posphi_star**2
            + 0.5 * A * (phi - posphi_star) ** 2
            + 1 / 6 * B * (phi - posphi_star) ** 3
            + 1 / 24 * C * (phi - posphi_star) ** 4
            + 1 / 120 * D * (phi - posphi_star) ** 5
            + 1 / 720 * E * (phi - posphi_star) ** 6
        )

    def UTaylornegative(phi, phi_g):
        return (
            -EJ * sp.cos(negphi_star + phi_g)
            + 0.5 * EL * negphi_star**2
            + 0.5 * A1 * (phi - negphi_star) ** 2
            + 1 / 6 * B1 * (phi - negphi_star) ** 3
            + 1 / 24 * C1 * (phi - negphi_star) ** 4
            + 1 / 120 * D1 * (phi - negphi_star) ** 5
            + 1 / 720 * E1 * (phi - negphi_star) ** 6
        )

    phi_zpf1 = (2 * EC / A) ** (1 / 4)
    omega_01 = np.sqrt(8 * A * EC)

    phi_zpf2 = (2 * EC / A) ** (1 / 4)
    omega_02 = np.sqrt(8 * A * EC)

    # quantum operators
    a = destroy(levels)
    a_dag = create(levels)
    n = np.diag(np.arange(levels))

    id_matrix = np.eye(levels)

    # Qunatum phase operator
    Phi1 = phi_zpf1 * (a + a_dag)
    Phi2 = phi_zpf2 * (a + a_dag)

    # value of potential at minima
    posUminima = U(posphi_star, phig)
    negUminima = U(negphi_star, phig)

    Umin = np.float64(posUminima)
    Umin1 = np.float64(negUminima)

    halfs = np.full(20, 1 / 2)
    diag_matrix = np.diag(halfs)

    H1 = (
        omega_01 * (n + diag_matrix)
        + 1 / 6 * B * (Phi1 @ Phi1 @ Phi1)
        + 1 / 24 * C * (Phi1 @ Phi1 @ Phi1 @ Phi1)
        + 1 / 120 * D * (Phi1 @ Phi1 @ Phi1 @ Phi1 @ Phi1)
        + 1 / 720 * (Phi1 @ Phi1 @ Phi1 @ Phi1 @ Phi1 @ Phi1)
    )

    H1 += Umin * np.eye(H1.shape[0])

    H2 = (
        omega_01 * (n + diag_matrix)
        + 1 / 6 * B * (Phi2 @ Phi2 @ Phi2)
        + 1 / 24 * C * (Phi2 @ Phi2 @ Phi2 @ Phi2)
        + 1 / 120 * D * (Phi2 @ Phi2 @ Phi2 @ Phi2 @ Phi2)
        + 1 / 720 * (Phi2 @ Phi2 @ Phi2 @ Phi2 @ Phi2 @ Phi2)
    )

    H2 += Umin * np.eye(H2.shape[0])

    l1, v1 = eigh(H1)
    omega01 = l1[1] - l1[0]
    Anh1 = l1[2] - 2 * l1[1] + l1[0]

    l2, v2 = eigh(H2)
    omega02 = l2[1] - l2[0]
    Anh2 = l2[2] - 2 * l2[1] + l2[0]

    return (
        l1,
        l2,
        U,
        UTaylorpositive,
        UTaylornegative,
        # phi_zpf1,
        # A1,
        # der2,
        Anh1,
        Anh2,
        # omega_01,
        # omega_02,
        Umin,
    )


# finding classical turning point
def finding_intersections(EJ, EL, L, eigenergies1):
    phi = sp.Symbol("phi")

    potentialcheck = -EJ * np.cos(phiplot_values + pi) + 0.5 * EL * phiplot_values**2

    index_intersections = list()
    Energy_level = eigenergies1[L]

    for i in range(0, len(potentialcheck) - 2):
        if potentialcheck[i] > Energy_level and potentialcheck[i + 1] < Energy_level:
            index_intersections.append(i)

        elif potentialcheck[i] < Energy_level and potentialcheck[i + 1] > Energy_level:
            index_intersections.append(i)

        elif potentialcheck[i] == Energy_level:
            if (
                potentialcheck[i - 1] > Energy_level
                and potentialcheck[i + 1] < Energy_level
            ):
                index_intersections.append(i)

    intersection = index_intersections[2]

    classical_turning_point = phiplot_values[intersection]

    return classical_turning_point


def integration(EJin, ELin, X1, Ein):
    Ej = EJin
    El = ELin
    phi = sp.Symbol("phi")
    E = Ein

    def integrate(Ej, El, E):
        return (-Ej * sp.cos(phi + phig) + 0.5 * El * phi**2 - E) ** 0.5

    value = sp.integrate(integrate(EJ, EL, E), (phi, 0, X1))

    return value


######### Below this is single well analysis #########


# def finding_derivatives(phi_g):
#     phi = sp.Symbol("phi")
#     der1 = U(phi, phi_g).diff(phi)
#     der2 = der1.diff(phi)
#     der3 = der2.diff(phi)
#     der4 = der3.diff(phi)

#     return (
#         der1,
#         der2,
#         der3,
#         der4,
#     )


# # from scipy.linalg import eigh

# LJ = 10e-9
# L = 6.596e-9
# CJ = 100.131e-15
# phi_g = 0.6 * pi

# levels = 20
# EJ = (hbar / 2 / e) ** 2 / (LJ * h) / 1e9
# EL = (hbar / 2 / e) ** 2 / (L * h) / 1e9
# EC = e**2 / (2 * CJ * h) / 1e9


# def U(phi, phi_g):
#     return EJ * sp.cos(phi + phi_g) + 1 / 2 * EL * phi**2


# # # quantum operators
# # def create(n):
# #     return np.diag(np.sqrt(np.arange(1, n)), k=-1)


# # def destroy(n):
# #     return np.diag(np.sqrt(np.arange(1, n)), k=1)


# def finding_Qubit(phi_g):
#     levels = 20
#     phi_values = np.arange(-pi, pi, 0.0001 * pi)

#     der1, der2, der3, der4 = finding_derivatives(phi_g)
#     phi = sp.Symbol("phi")
#     der1_expr = sp.lambdify(phi, der1)
#     der1_values = der1_expr(phi_values)

#     minimum = np.argmin(np.abs(der1_values))
#     phi_star = phi_values[minimum]

#     # Taylor potential parameters
#     Adash = sp.lambdify(phi, der2)
#     A = Adash(phi_star)
#     Bdash = sp.lambdify(phi, der3)
#     B = Bdash(phi_star)
#     Cdash = sp.lambdify(phi, der4)
#     C = Cdash(phi_star)

#     # defining Taylor series of function
#     def UTaylor(phi_in):
#         return (
#             U(phi, phi_g).subs(phi, phi_star)
#             + 1 / 2 * der2.subs(phi, phi_star) * (phi_in - phi_star) ** 2
#             + 1 / 6 * der3.subs(phi, phi_star) * (phi_in - phi_star) ** 3
#             + 1 / 24 * der4.subs(phi, phi_star) * (phi_in - phi_star) ** 4
#         )

#     phi_zpf = (2 * EC / A) ** (1 / 4)
#     omega_0 = np.sqrt(8 * A * EC)

#     # quantum operators
#     a = destroy(levels)
#     a_dag = create(levels)
#     n = a_dag @ a
#     id_matrix = np.eye(levels)

#     # Qunatum phase operator
#     Phi = phi_zpf * (a + a_dag)

#     # getting eigenenergies and anharmonicities

#     H = omega_0 * (n + 1 / 2) + 1 / 6 * B * Phi**3 + 1 / 24 * C * Phi**4

#     l, v = eigh(H)
#     omega01 = l[1] - l[0]
#     Anh = l[2] - 2 * l[1] + l[0]

#     return l, Anh, omega_0, UTaylor, phi_zpf, phi_star
