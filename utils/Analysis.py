import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import sympy as sp
import scipy as sc
import math


class CWS_analysis:
    def __init__(self, imaginary_data, real_data):
        self.imaginary_data = imaginary_data
        self.real_data = real_data
        self.__flipped_real = np.flip(self.real_data, axis=0)
        self.__flipped_imaginary = np.flip(self.imaginary_data, axis=0)
        self.__magnitude_data = (
            self.__flipped_real**2 + self.__flipped_imaginary**2
        ) ** 0.5
        self.__phase_data = np.arctan(self.__flipped_imaginary / self.__flipped_real)
        self.aggr_data = {
            "real": None,
            "imaginary": None,
            "magnitude": None,
            "phase": None,
        }
        self.frequency_data = {
            "real": None,
            "imaginary": None,
            "magnitude": None,
            "phase": None,
        }
        self.optimal_parameters = {
            "EJ": None,
            "EL": None,
            "EC": None,
        }

    def plot_raw_data(self, results_file=None):
        plt.rcParams["figure.figsize"] = (15, 15)
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(self.__flipped_real, cmap="Blues", aspect="auto")
        axs[0, 0].set_title("Real", fontsize=15)
        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 0].get_yaxis().set_visible(False)
        axs[0, 1].imshow(self.__flipped_imaginary, cmap="Reds", aspect="auto")
        axs[0, 1].set_title("Imaginary", fontsize=15)
        axs[0, 1].get_xaxis().set_visible(False)
        axs[0, 1].get_yaxis().set_visible(False)
        axs[1, 0].imshow(self.__magnitude_data, cmap="Greens", aspect="auto")
        axs[1, 0].set_title("Magnitude", fontsize=15)
        axs[1, 0].get_xaxis().set_visible(False)
        axs[1, 0].get_yaxis().set_visible(False)
        axs[1, 1].imshow(self.__phase_data, cmap="Oranges", aspect="auto")
        axs[1, 1].set_title("Phase", fontsize=15)
        axs[1, 1].get_xaxis().set_visible(False)
        axs[1, 1].get_yaxis().set_visible(False)

        if results_file is not None:
            plt.savefig(results_file)

        plt.show()

    def plot_raw_filterdata(self, data_type="real", voltage_range=[-12, 10]):
        voltage = np.linspace(voltage_range[0], voltage_range[1], 133)
        plt.rcParams["figure.figsize"] = (15, 8)
        fig, axs = plt.subplots(1, 2)
        filtered_resonance_frequencies, non_filtered_reseonance_frequencies = (
            self.get_resonance_frequencies(data_type)
        )
        axs[0].scatter(voltage, non_filtered_reseonance_frequencies)
        axs[0].set_title("Raw Data", fontsize=20)
        axs[0].set_ylabel("frequency (GHz)")
        axs[0].set_xlabel("voltage (V)")
        axs[1].scatter(voltage, filtered_resonance_frequencies)
        axs[1].set_title("Meadian Fitted Data", fontsize=20)
        axs[1].set_ylabel("frequency (GHz)")
        axs[1].set_xlabel("voltage (V)")

    def get_resonance_frequencies(self, data_type="real"):

        self.get_aggregate_data(data_type)

        resonant_point_index = np.zeros(self.aggr_data[data_type].shape[1])

        for l in range(0, len(self.aggr_data[data_type][0])):
            average = np.average(self.aggr_data[data_type][:, l])
            add_indicies_max = np.argmax(self.aggr_data[data_type][:, l])
            add_indicies_min = np.argmin(self.aggr_data[data_type][:, l])

            test_max = np.abs(self.aggr_data[data_type][add_indicies_max, l]) - np.abs(
                average
            )

            test_min = np.abs(self.aggr_data[data_type][add_indicies_min, l]) - np.abs(
                average
            )

            if np.abs(test_max) > np.abs(test_min):
                resonant_point_index[l] = add_indicies_max

            else:
                resonant_point_index[l] = add_indicies_min

        frequencydata = np.zeros(self.aggr_data[data_type].shape[1])
        frequency02 = np.linspace(4.5, 7.5, self.aggr_data[data_type].shape[0])

        for i in range(0, 133):
            index = int(resonant_point_index[i])
            value = frequency02[index]
            frequencydata[i] = value

        frequencydata_filtered = self._median_filter(frequencydata)

        self.frequency_data[data_type] = frequencydata_filtered
        return frequencydata_filtered, frequencydata

    def _median_filter(self, data, size=3):
        return sc.ndimage.filters.median_filter(data, size=size)

    def finding_qubit(self, EJ, EL, EC, phi_g):
        # levels = 20
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

        # phi_zpf = (2 * EC / A) ** (1 / 4)
        omega_0 = np.sqrt(8 * A * EC)

        return omega_0

    def _get_data(self, data_type):
        if data_type == "real":
            return self.real_data
        elif data_type == "imaginary":
            return self.__flipped_imaginary
        elif data_type == "magnitude":
            return self.__magnitude_data
        elif data_type == "phase":
            return self.__phase_data
        else:
            raise ValueError("Invalid data type")

    def get_aggregate_data(self, data_type, rows_to_aggr=6, columns_to_aggr=6):

        data = self._get_data(data_type)

        num_columns = data.shape[1]
        num_rows = data.shape[0]
        aggr_num_rows = int(math.floor(num_rows / rows_to_aggr))
        aggr_num_columns = int(math.floor(num_columns / columns_to_aggr))

        aggr_column_data = np.zeros((num_rows, aggr_num_columns))

        for i in range(0, aggr_num_columns):
            aggr_column_data[:, i] = (
                data[:, 6 * i]
                + data[:, 6 * i + 1]
                + data[:, 6 * i + 2]
                + data[:, 6 * i + 3]
                + data[:, 6 * i + 4]
                + data[:, 6 * i + 5]
            )

        aggr_row_and_column_data = np.zeros((aggr_num_rows, aggr_num_columns))

        for k in range(0, aggr_num_columns):
            for l in range(0, aggr_num_rows):
                if 6 * (l + 1) - 1 < num_rows:
                    aggr_row_and_column_data[l, k] = (
                        aggr_column_data[6 * l, k]
                        + aggr_column_data[6 * l + 1, k]
                        + aggr_column_data[6 * l + 2, k]
                        + aggr_column_data[6 * l + 3, k]
                        + aggr_column_data[6 * l + 4, k]
                        + aggr_column_data[6 * l + 5, k]
                    )

        self.aggr_data[data_type] = aggr_row_and_column_data
        return aggr_row_and_column_data

    def _generate_phi_values(self, data_type):
        minpoint = np.argmin(self.frequency_data[data_type])
        maxpoint = np.argmax(self.frequency_data[data_type])

        part = pi / np.abs(maxpoint - minpoint)
        starthere = 0 - (minpoint + 1) * part
        endpoint = 0 + (np.size(self.frequency_data[data_type]) - (minpoint + 1)) * part

        return np.arange(
            starthere,
            endpoint,
            (endpoint - starthere) / (np.size(self.frequency_data[data_type]) - 0.5),
        )

    def optimise_parameters(self, data_type, EC, EJ_range, EL_range, number_of_steps):

        EC = EC  ###enter expected EC###
        EJvalues = np.linspace(
            EJ_range[0], EJ_range[1], number_of_steps
        )  ###pick range of EJ values wanted scanned###
        ELvalues = np.linspace(
            EL_range[0], EL_range[1], number_of_steps
        )  ###pick range of EL values wanted scanned###

        self.R2Data = np.zeros([number_of_steps, number_of_steps])

        phi_g_sweep = self._generate_phi_values(data_type)

        for i in range(0, 10):

            EJ = EJvalues[i]

            for j in range(0, 10):

                EL = ELvalues[j]

                omega_0_ar = np.zeros(np.size(phi_g_sweep))

                for k in range(0, np.size(omega_0_ar)):

                    phig = phi_g_sweep[k]

                    omega_0_ar[k] = self.finding_qubit(EJ, EL, EC, phig)

                residuals = self.frequency_data[data_type] - omega_0_ar
                sq = np.dot(residuals, residuals)

                self.R2Data[i, j] = sq

        indexvalues = np.unravel_index(self.R2Data.argmin(), self.R2Data.shape)

        EJopt = EJvalues[indexvalues[0]]
        ELopt = ELvalues[indexvalues[1]]

        self.optimal_parameters["EJ"] = EJopt
        self.optimal_parameters["EL"] = ELopt

        return EJopt, ELopt

    def plot_parameter_colour_map(self, EJ_range, EL_range, number_of_steps):
        plt.imshow(self.R2Data)
        plt.colorbar()
        plt.rcParams["figure.figsize"] = (8, 8)

    def optimise_EC(self, EC_range, data_type="real"):
        ECvalues = np.linspace(EC_range[0], EC_range[1], 30)
        phi_g_sweep = self._generate_phi_values(data_type)

        ECData = np.zeros(25)
        for i in range(0, 25):
            EC = ECvalues[i]

            omega_0_EC = np.zeros(np.size(self.frequency_data[data_type]))

            for j in range(0, np.size(omega_0_EC)):

                phigin = phi_g_sweep[j]

                omega_0_EC[j] = self.finding_qubit(
                    self.optimal_parameters["EJ"],
                    self.optimal_parameters["EL"],
                    EC,
                    phigin,
                )

            resd = np.abs(self.frequency_data[data_type] - omega_0_EC)
            squared = np.dot(resd, resd)
            ECData[i] = squared

        index = np.argmin(ECData)
        ECopt = ECvalues[index]

        self.optimal_parameters["EC"] = ECopt

        return ECopt

    def plot_optimsied_resonance_curve(self, data_type="real"):
        phi_g_sweep = self._generate_phi_values(data_type)

        omega_0_opt1 = []

        for phi_g in phi_g_sweep:
            omega_opt = self.finding_qubit(
                self.optimal_parameters["EJ"],
                self.optimal_parameters["EL"],
                self.optimal_parameters["EC"],
                phi_g,
            )

            omega_0_opt1.append(omega_opt)

        plt.rcParams["figure.figsize"] = (15, 8)

        fig, ax = plt.subplots(1, 2)
        ax.scatter(phi_g_sweep, self.frequency_data[data_type])
        ax.plot(phi_g_sweep, omega_0_opt1, "r")
        ax.set_ylabel("frequency (GHz)")
        ax.set_xlabel("phi")
        ax.set_title("Optimal Fit")

        plt.savefig("analyse.jpg")
        plt.show()
