# General python library imports
# Argument parsing
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from pylab import *

# rcParams["text.usetex"] = True

# Use these lines for thesis-style plots, else presentation-style plots
# rcParams["xtick.labelsize"] = 20
# rcParams["ytick.labelsize"] = 20
# rcParams["axes.labelsize"] = 20
# rcParams["axes.titlesize"] = 20
# rcParams["legend.fontsize"] = 17

# Pretty plots and a colour-blind friendly colour scheme
plt.style.use("classic")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "FreeSerif"

colors = {
    "blue": "#4477aa",
    "cyan": "#66ccee",
    "green": "#228833",
    "yellow": "#ccbb44",
    "red": "#ee6677",
    "purple": "#aa3377",
    "grey": "#bbbbbb",
}


# Function to read cooling function file and return data
def read_file_cool(file_name):
    print("Reading file...")
    data = np.genfromtxt(file_name)
    av = data[:, 2]
    cii = data[:, 3]
    ci = data[:, 4]
    oi = data[:, 5]
    co = data[:, 6]
    return av, cii, ci, oi, co


# Function to plot the cooling function for all species as a function of visual extinction
def plot_cooling_func(data_files, fig_title, savefig_file_name):
    print("Plotting cooling function...")
    av, cii, ci, oi, co = data_files
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.loglog(av, cii, color=colors["blue"], label="CII", linewidth=2)
    ax.loglog(av, ci, color=colors["red"], label="CI", linewidth=2)
    ax.loglog(av, oi, color=colors["green"], label="OI", linewidth=2)
    ax.loglog(av, co, color=colors["yellow"], label="CO", linewidth=2)
    ax.set_xlim([1e-6, 20])
    ax.legend(loc="best")
    ax.set_xlabel(r"$A_v$ [mag]", fontsize=15)
    ax.set_ylabel("Cooling function", fontsize=15)
    ax.set_title(fig_title, fontsize=15, fontweight="bold")
    ax.grid()
    fig.savefig(savefig_file_name, bbox_inches="tight")


# Run the cooling functions given a prefix for all file names
def run_cooling_funcs(prefix):
    av, cii, ci, oi, co = read_file_cool(prefix + ".cool.fin")
    data = [av, cii, ci, oi, co]
    plot_cooling_func(
        data_files=data, fig_title="Test run", savefig_file_name=prefix + "_cool.png"
    )


# Function to read heating function file and return data
def read_file_heat(file_name):
    print("Reading files...")
    data = np.genfromtxt(file_name)
    av = data[:, 2]
    hr02 = data[:, 4]
    hr04 = data[:, 6]
    hr05 = data[:, 7]
    hr06 = data[:, 8]
    hr07 = data[:, 9]
    hr08 = data[:, 10]
    hr09 = data[:, 11]
    hr10 = data[:, 12]
    hr12 = data[:, 14]
    return av, hr02, hr04, hr05, hr06, hr07, hr08, hr09, hr10, hr12


# Function to plot the heating function for all species as a function of visual extinction
def plot_heating_func(data_files, fig_title, savefig_file_name):
    print("Plotting heating function...")
    av, hr02, hr04, hr05, hr06, hr07, hr08, hr09, hr10, hr12 = data_files
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.loglog(av, hr02, label="Photoelectric", linewidth=2)
    ax.loglog(av, hr04, label="Carbon ionization", linewidth=2)
    ax.loglog(av, hr05, label=r"H$_2$ formation", linewidth=2)
    ax.loglog(av, hr06, label=r"H$_2$ photodiss.", linewidth=2)
    ax.loglog(av, hr07, label="FUV pumbing", linewidth=2)
    ax.loglog(av, hr08, label="Cosmic rays", linewidth=2)
    ax.loglog(av, hr09, label="Microturbulent", linewidth=2)
    ax.loglog(av, hr10, "--", label="Chemical", linewidth=2)
    ax.loglog(av, hr12, "--", label="Gas-grain", linewidth=2)
    ax.set_ylim([1e-42, 1e-19])
    ax.set_xlim([1e-6, 20])
    ax.legend(loc="best")
    ax.set_xlabel(r"$A_v$ [mag]", fontsize=15)
    ax.set_ylabel("Heating function", fontsize=15)
    ax.set_title(fig_title, fontsize=15, fontweight="bold")
    ax.grid()
    fig.savefig(savefig_file_name, bbox_inches="tight")


# Run the heating functions given a prefix for all file names
def run_heating_funcs(prefix):
    av, hr02, hr04, hr05, hr06, hr07, hr08, hr09, hr10, hr12 = read_file_heat(
        prefix + ".heat.fin"
    )
    data = [av, hr02, hr04, hr05, hr06, hr07, hr08, hr09, hr10, hr12]
    plot_heating_func(
        data_files=data, fig_title="Test run", savefig_file_name=prefix + "_heat.png"
    )


# Run the line emissivities file and return data
def read_linefin_files(file_name):
    print("Reading files...")
    data = np.genfromtxt(file_name)
    av = data[:, 2]
    l01 = data[:, 3]  # CII
    l02 = data[:, 4]  # CI 1-0
    l03 = data[:, 5]  # CI 2-0
    l04 = data[:, 6]  # CI 2-1
    l05 = data[:, 7]  # OI 1-0
    l06 = data[:, 8]  # OI 2-0
    l07 = data[:, 9]  # OI 2-1
    l08 = data[:, 10]  # CO (1-0)
    l09 = data[:, 11]  # CO (2-1)
    l10 = data[:, 12]  # CO (3-2)
    l11 = data[:, 13]  # CO (4-3)
    l12 = data[:, 14]  # CO (5-4)
    l13 = data[:, 15]  # CO (6-5)
    l14 = data[:, 16]  # CO (7-6)
    l15 = data[:, 17]  # CO (8-7)
    l16 = data[:, 18]  # CO (9-8)
    l17 = data[:, 19]  # CO (10-9)
    return (
        av,
        l01,
        l02,
        l03,
        l04,
        l05,
        l06,
        l07,
        l08,
        l09,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15,
        l16,
        l17,
    )


# Function to plot fine structure and ladder line emissivities
def plot_line_emissivity(data_files, fig_titles, savefig_file_names):
    print("Plotting line emissivities...")
    (
        av,
        l01,
        l02,
        l03,
        l04,
        l05,
        l06,
        l07,
        l08,
        l09,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15,
        l16,
        l17,
    ) = data_files
    fine_struct_title, ladder_title = fig_titles
    fine_struct_savefig, ladder_savefig = savefig_file_names

    # Fine structure plot
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.loglog(av, l01, label=r"CII 158$\mu$m", linewidth=2)
    ax1.loglog(av, l02, label=r"CI 609$\mu$m", linewidth=2)
    ax1.loglog(av, l04, label=r"CI 320$\mu$m", linewidth=2)
    ax1.loglog(av, l05, label=r"OI 64$\mu$m", linewidth=2)
    ax1.loglog(av, l07, label=r"OI 146$\mu$m", linewidth=2)
    ax1.set_xlim([1e-6, 20])
    ax1.legend(loc="best")
    ax1.set_xlabel(r"$A_v$ [mag]", fontsize=15)
    ax1.set_ylabel(r"Line emissivity [erg/cm$^3$/s]", fontsize=15)
    ax1.set_title(fine_struct_title, fontsize=15, fontweight="bold")
    ax1.grid()
    fig1.savefig(fine_struct_savefig, bbox_inches="tight")

    # CO ladder plot
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    ax2.loglog(av, l08, label=r"CO(1-0)", linewidth=2)
    ax2.loglog(av, l09, label=r"CO(2-1)", linewidth=2)
    ax2.loglog(av, l10, label=r"CO(3-2)", linewidth=2)
    ax2.loglog(av, l11, label=r"CO(4-3)", linewidth=2)
    ax2.loglog(av, l12, label=r"CO(5-4)", linewidth=2)
    ax2.loglog(av, l13, label=r"CO(6-5)", linewidth=2)
    ax2.loglog(av, l14, label=r"CO(7-6)", linewidth=2)
    ax2.loglog(av, l15, "--", label=r"CO(8-7)", linewidth=2)
    ax2.loglog(av, l16, "--", label=r"CO(9-8)", linewidth=2)
    ax2.loglog(av, l17, "--", label=r"CO(10-9)", linewidth=2)
    ax2.set_xlim([1e-6, 20])
    ax2.legend(loc="best")
    ax2.set_xlabel(r"$A_v$ [mag]", fontsize=15)
    ax2.set_ylabel(r"Line emissivity [erg/cm$^3$/s]", fontsize=15)
    ax2.set_title(ladder_title, fontsize=15, fontweight="bold")
    ax2.grid()
    fig2.savefig(ladder_savefig, bbox_inches="tight")


# Run the heating functions given a prefix for all file names
def run_line_emissivity_funcs(prefix):
    (
        av,
        l01,
        l02,
        l03,
        l04,
        l05,
        l06,
        l07,
        l08,
        l09,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15,
        l16,
        l17,
    ) = read_linefin_files(prefix + ".line.fin")
    data = [
        av,
        l01,
        l02,
        l03,
        l04,
        l05,
        l06,
        l07,
        l08,
        l09,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15,
        l16,
        l17,
    ]
    fig_titles = ["Fine structure lines", "CO ladder"]
    savefig_file_names = [prefix + "_fine_structure.png", prefix + "_CO_ladder.png"]
    plot_line_emissivity(
        data_files=data, fig_titles=fig_titles, savefig_file_names=savefig_file_names
    )


# Read the PDR properties file and return data
def read_pdr_file(file_name, start_index=None, end_index=None):
    # indices:
    # av - 2, tgas - 3, tdust - 4
    # reduced network: HI - 39, H2 - 38, CII - 18, CI - 32, C0 - 35
    # full network: HI - 221, H2 - 220, CII - 179, CI - 217, CO - 218

    data = np.genfromtxt(file_name)
    # print(data[:,2])
    return [
        data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 221],
        data[:, 220],
        data[:, 179],
        data[:, 217],
        data[:, 218],
    ]


# Plotting necessary plots from the PDR properties data
def plot_pdr(data_files, prefix, savefig_file_names):
    print("Plotting some properties...")
    av, tgas, tdust, HI, H2, CII, CI, CO = data_files
    # tgas_title, tdust_title, HI_title, H2_title, CII_title, CI_title, CO_title = fig_titles
    temp_savefig, abundances_savefig = savefig_file_names

    # Gas and dust temperatures
    fig_temp, ax_temp = plt.subplots(nrows=1, ncols=1)
    ax_temp.semilogx(av, tgas, color=colors["blue"], label=r"$T_{gas}$", linewidth=2)
    ax_temp.semilogx(
        av, tdust, color=colors["yellow"], label=r"$T_{dust}$", linewidth=2
    )
    ax_temp.set_xlim([1e-6, 20])
    ax_temp.legend(loc="best")
    ax_temp.set_xlabel(r"$A_v$ [mag]", fontsize=15)
    ax_temp.set_ylabel("Gas and dust temperature [K]", fontsize=15)
    ax_temp.set_title(prefix, fontsize=15)
    ax_temp.grid()
    fig_temp.savefig(temp_savefig, bbox_inches="tight")

    # Abundances of species
    fig_abund, ax_abund = plt.subplots(nrows=1, ncols=1)
    ax_abund.loglog(av, HI, color=colors["blue"], label="HI", linewidth=2)
    ax_abund.loglog(av, H2, color=colors["cyan"], label="H2", linewidth=2)
    ax_abund.loglog(av, CII, color=colors["green"], label="CII", linewidth=2)
    ax_abund.loglog(av, CI, color=colors["yellow"], label="CI", linewidth=2)
    ax_abund.loglog(av, CO, color=colors["red"], label="CO", linewidth=2)
    # ax_abund.set_xlim([1e-6, 20])
    ax_abund.legend(loc="best")
    ax_abund.set_xlabel(r"$A_v$ [mag]", fontsize=15)
    ax_abund.set_ylabel("Abundances of species", fontsize=15)
    ax_abund.set_title(prefix)
    ax_abund.grid()
    fig_abund.savefig(abundances_savefig, bbox_inches="tight")


# Run the heating functions given a prefix for all file names
def run_pdr(prefix):
    av, tgas, tdust, HI, H2, CII, CI, CO = read_pdr_file(file_name=prefix + ".pdr.fin")
    data = [av, tgas, tdust, HI, H2, CII, CI, CO]
    savefig_file_names = [prefix + "_temp.png", prefix + "_abundances.png"]
    plot_pdr(data_files=data, prefix=prefix, savefig_file_names=savefig_file_names)


if __name__ in "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model prefix")
    args = parser.parse_args()
    run_prefix = args.model_name + "/" + args.model_name

    # Running all functions
    # run_prefix = "model_900/model_900"
    run_cooling_funcs(prefix=run_prefix)
    run_heating_funcs(prefix=run_prefix)
    run_line_emissivity_funcs(prefix=run_prefix)
    run_pdr(prefix=run_prefix)
