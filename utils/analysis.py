import numpy as np
import math
import scipy as sc
import sympy as sp
from scipy.constants import pi, hbar, h, e
from scipy import signal


def get_resonance_frequencies(matrix):
    num_rows = int(math.floor(len(matrix) / 6))
    aggr_rows = np.zeros((3001, 133))

    for i in range(0, 133):
        aggr_rows[:, i] = (
            matrix[:, 6 * i]
            + matrix[:, 6 * i + 1]
            + matrix[:, 6 * i + 2]
            + matrix[:, 6 * i + 3]
            + matrix[:, 6 * i + 4]
            + matrix[:, 6 * i + 5]
        )

    num_columns = int(math.floor(len(aggr_rows) / 6))
    aggr_rows_columns = np.zeros((500, 133))

    for k in range(0, 133):
        for l in range(0, 500):
            aggr_rows_columns[l, k] = (
                aggr_rows[6 * l, k]
                + aggr_rows[6 * l + 1, k]
                + aggr_rows[6 * l + 2, k]
                + aggr_rows[6 * l + 3, k]
                + aggr_rows[6 * l + 4, k]
                + aggr_rows[6 * l + 5, k]
            )

    resonant_point_index = np.zeros(len(aggr_rows_columns[0]))

    for l in range(0, len(aggr_rows_columns[0])):
        average = np.average(aggr_rows_columns[:, l])
        add_indicies_max = np.argmax(aggr_rows_columns[:, l])
        add_indicies_min = np.argmin(aggr_rows_columns[:, l])

        test_max = np.abs(aggr_rows_columns[add_indicies_max, l]) - np.abs(average)

        test_min = np.abs(aggr_rows_columns[add_indicies_min, l]) - np.abs(average)

        if np.abs(test_max) > np.abs(test_min):
            resonant_point_index[l] = add_indicies_max

        else:
            resonant_point_index[l] = add_indicies_min

    frequencydata = np.zeros(133)
    frequency02 = np.linspace(4.5, 7.5, 500)

    for i in range(0, 133):
        index = int(resonant_point_index[i])
        value = frequency02[index]
        frequencydata[i] = value

    return frequencydata


def RefineResults(data):
    datainput = data

    for i in range(0, len(datainput) - 1):
        if abs(datainput[i] - datainput[i + 1]) > 0.3:
            datainput[i] = 0
        else:
            datainput[i] = datainput[i]

    refinedData = datainput[datainput != 0]

    return refinedData


def median_fit(data):
    datahat = sc.ndimage.filters.median_filter(data, size=3)

    return datahat


def finding_Qubit(EJ, EL, EC, phi_g):
    levels = 20
    phi_values = np.arange(-pi, pi, 0.0001 * pi)

    def U(phi, phi_g):
        return EJ * sp.cos(phi + phi_g) + 1 / 2 * EL * phi**2

    def finding_derivatives(phi_g):
        phi = sp.Symbol("phi")
        der1 = U(phi, phi_g).diff(phi)
        der2 = der1.diff(phi)

        return der1, der2

    der1, der2 = finding_derivatives(phi_g)
    phi = sp.Symbol("phi")
    der1_expr = sp.lambdify(phi, der1)
    der1_values = der1_expr(phi_values)

    minimum = np.argmin(np.abs(der1_values))
    phi_star = phi_values[minimum]

    # Taylor potential parameters
    Adash = sp.lambdify(phi, der2)
    A = Adash(phi_star)

    phi_zpf = (2 * EC / A) ** (1 / 4)
    omega_0 = np.sqrt(8 * A * EC)

    return omega_0
