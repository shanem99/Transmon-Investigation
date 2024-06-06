import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar, h, e
from scipy import optimize
import sympy as sp
from scipy.linalg import eigh
from abc import ABC, abstractmethod
from scipy.optimize import root
from scipy.misc import derivative

numdivisions = 1000


class BaseQubit(ABC):

    def __init__(self, L, LJ, CJ, phi_g):
        self.EJ = (hbar / 2 / e) ** 2 / (LJ * h) / 1e9
        self.EC = e**2 / (2 * CJ * h) / 1e9
        self.EL = (hbar / 2 / e) ** 2 / (L * h) / 1e9
        self.phi = sp.Symbol("phi")
        self.phi_g = phi_g
        self.derivatives = []
        self.minima = []

    def taylor_series(self, var, point, order=6):
        return self.potential.series(var, point, n=order).removeO()

    def compute_derivatives(self, order=6):

        if not self.derivatives == []:
            self.derivatives = []

        for i in range(order):
            if len(self.derivatives) == 0:
                self.derivatives.append({"1_derivative": self.potential.diff(self.phi)})
            else:
                self.derivatives.append(
                    {
                        f"{i+1}_derivative": self.derivatives[i - 1][
                            f"{i}_derivative"
                        ].diff(self.phi)
                    }
                )

    # quantum operators
    def create(self, n):
        return np.diag(np.sqrt(np.arange(1, n)), k=-1)

    def destroy(self, n):
        return np.diag(np.sqrt(np.arange(1, n)), k=1)


class Double_Well_qubit(BaseQubit):

    def __init__(self, L, LJ, CJ, phi_g):
        super().__init__(L, LJ, CJ, phi_g)
        self.potential = (
            -self.EJ * sp.cos(self.phi + self.phi_g) + 0.5 * self.EL * self.phi**2
        )

    def _lowest_wells(self, minima_points):
        U = sp.lambdify(self.phi, self.potential)
        wells = [(minimum, U(minimum)) for minimum in minima_points]

        wells.sort(key=lambda x: x[1])

        if self.minima == []:
            self.minima.append(wells[0][0])
            self.minima.append(wells[1][0])
        else:
            self.minima[0] = wells[0][0]
            self.minima[1] = wells[1][0]

    def find_minima(self, start, end, numdivisions=100000):

        if self.derivatives == []:
            self.compute_derivatives()

        first_derivative = sp.lambdify(self.phi, self.derivatives[0]["1_derivative"])
        second_derivative = sp.lambdify(self.phi, self.derivatives[1]["2_derivative"])

        x_vals = np.linspace(start, end, numdivisions)
        roots = [x for x in x_vals if np.isclose(first_derivative(x), 0, atol=1e-3)]

        minima = [r for r in roots if second_derivative(r) > 0]

        return self._lowest_wells(minima)

    def compute_derivates_at_minima(self):
        if self.minima == []:
            self.find_minima(-6, 6)
        derivatative_values = {"negative_minima": {}, "positive_minima": {}}
        for i in range(len(self.derivatives) - 1):
            derivatative_values["negative_minima"][f"{i+2}_derivative"] = (
                self.derivatives[i + 1][f"{i+2}_derivative"].subs(
                    self.phi, self.minima[0]
                )
            )
            derivatative_values["positive_minima"][f"{i+2}_derivative"] = (
                self.derivatives[i + 1][f"{i+2}_derivative"].subs(
                    self.phi, self.minima[1]
                )
            )

        return derivatative_values

    def find_energy_levels(self, levels=10):

        self.compute_derivatives()

        derivative_values_at_minima = self.compute_derivates_at_minima()

        phi_zpf1 = (
            2 * self.EC / derivative_values_at_minima["positive_minima"]["2_derivative"]
        ) ** (1 / 4)

        omega_01 = np.sqrt(
            8
            * float(derivative_values_at_minima["positive_minima"]["2_derivative"])
            * self.EC
        )

        phi_zpf2 = (
            2 * self.EC / derivative_values_at_minima["negative_minima"]["2_derivative"]
        ) ** (1 / 4)

        omega_02 = np.sqrt(
            8
            * float(derivative_values_at_minima["negative_minima"]["2_derivative"])
            * self.EC
        )

        a = self.create(levels)
        a_dag = self.destroy(levels)

        Phi1 = phi_zpf1 * (a + a_dag)
        Phi2 = phi_zpf2 * (a + a_dag)

        if self.minima == []:
            self.find_minima

        U = sp.lambdify(self.phi, self.potential)

        Umin_pos = U(self.minima[1])
        Umin_neg = U(self.minima[0])

        n = np.diag(np.arange(levels))

        halfs = np.full(levels, 1 / 2)
        diag_matrix = np.diag(halfs)

        H1 = (
            omega_01 * (n + diag_matrix)
            + 1
            / 6
            * float(derivative_values_at_minima["positive_minima"]["3_derivative"])
            * (Phi1 @ Phi1 @ Phi1)
            + 1
            / 24
            * float(derivative_values_at_minima["positive_minima"]["4_derivative"])
            * (Phi1 @ Phi1 @ Phi1 @ Phi1)
            + 1
            / 120
            * float(derivative_values_at_minima["positive_minima"]["5_derivative"])
            * (Phi1 @ Phi1 @ Phi1 @ Phi1 @ Phi1)
            + 1 / 720 * (Phi1 @ Phi1 @ Phi1 @ Phi1 @ Phi1 @ Phi1)
        )

        H1 += Umin_pos * np.eye(H1.shape[0])

        h1_star = H1.astype(float)

        H2 = (
            omega_02 * (n + diag_matrix)
            + 1
            / 6
            * float(derivative_values_at_minima["negative_minima"]["3_derivative"])
            * (Phi2 @ Phi2 @ Phi2)
            + 1
            / 24
            * float(derivative_values_at_minima["negative_minima"]["4_derivative"])
            * (Phi2 @ Phi2 @ Phi2 @ Phi2)
            + 1
            / 120
            * float(derivative_values_at_minima["negative_minima"]["5_derivative"])
            * (Phi2 @ Phi2 @ Phi2 @ Phi2 @ Phi2)
            + 1 / 720 * (Phi2 @ Phi2 @ Phi2 @ Phi2 @ Phi2 @ Phi2)
        )

        H2 += Umin_neg * np.eye(H2.shape[0])

        h2_star = H2.astype(float)

        l1, v1 = eigh(h1_star)
        omega01 = l1[1] - l1[0]
        Anh1 = l1[2] - 2 * l1[1] + l1[0]

        l2, v2 = eigh(h2_star)
        omega02 = l2[1] - l2[0]
        Anh2 = l2[2] - 2 * l2[1] + l2[0]

        return l1, l2, omega02, omega01


class Single_Well_qubit(BaseQubit):
    def __init__(self, L, LJ, CJ, phi_g):
        super().__init__(L, LJ, CJ, phi_g)
        self.potential = (
            -self.EJ * sp.cos(self.phi + self.phi_g) + 0.5 * self.EL * self.phi**2
        )

    def find_minima(self, start, end, numdivisions=100000):

        if self.derivatives == []:
            self.compute_derivatives()

        first_derivative = sp.lambdify(self.phi, self.derivatives[0]["1_derivative"])
        second_derivative = sp.lambdify(self.phi, self.derivatives[1]["2_derivative"])

        x_vals = np.linspace(start, end, numdivisions)
        roots = [x for x in x_vals if np.isclose(first_derivative(x), 0, atol=1e-3)]

        minima = [r for r in roots if second_derivative(r) > 0]

        return self._lowest_well(minima)

    def _lowest_well(self, minima_points):
        U = sp.lambdify(self.phi, self.potential)
        wells = [(minimum, U(minimum)) for minimum in minima_points]

        wells.sort(key=lambda x: x[1])

        if self.minima == []:
            self.minima.append(wells[0][0])
        else:
            self.minima[0] = wells[0][0]

    def compute_derivates_at_minima(self):
        if self.minima == []:
            self.find_minima(-3, 3)
        derivatative_values = {}
        for i in range(len(self.derivatives) - 1):
            derivatative_values[f"{i+2}_derivative"] = self.derivatives[i + 1][
                f"{i+2}_derivative"
            ].subs(self.phi, self.minima[0])

        return derivatative_values

    def find_energy_levels(self, levels=10):

        self.compute_derivatives()

        derivative_values_at_minima = self.compute_derivates_at_minima()

        Phi = sp.Symbol("phi")

        phi_zpf = (
            2 * self.EC / float(derivative_values_at_minima["2_derivative"])
        ) ** (1 / 4)
        omega_0 = np.sqrt(
            8 * float(derivative_values_at_minima["2_derivative"]) * self.EC
        )

        # quantum operators
        a = self.destroy(levels)
        a_dag = self.create(levels)
        n = a_dag @ a

        # Qunatum phase operator
        Phi = phi_zpf * (a + a_dag)

        # getting eigenenergies and anharmonicities

        H = (
            omega_0 * (n + 1 / 2)
            + 1 / 6 * float(derivative_values_at_minima["3_derivative"]) * Phi**3
            + 1 / 24 * float(derivative_values_at_minima["4_derivative"]) * Phi**4
        )

        l, v = eigh(H)
        omega01 = l[1] - l[0]
        Anh = l[2] - 2 * l[1] + l[0]

        return l, Anh
