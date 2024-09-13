import numpy as np
import matplotlib.pyplot as plt
from plot_figures import read_pdr_file, plot_pdr

plt.rcParams["text.usetex"] = True
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 15

# Function to get trends of temperature and abundances of species by varying certain parameters
def get_trend_plots(varying_qty, prefixes, linestyles, colors):
    if varying_qty == "g_uv":
        qtys = np.array([10**4, 10**3, 10**2, 10**1]) 
        latex_string = "G_{{UV}}"
    if varying_qty == "CR_zeta":
        qtys = np.array([10**-17, 10**-16, 10**-15])
        latex_string = "\zeta_{{CR}}"
    if varying_qty == "n_H":
        qtys = np.array([10**2, 10**3, 10**4, 10**5, 10**6])
        latex_string = "n_{{H}}"
    temp_ls, abund_ls = linestyles
    
    # Generating names for saving files
    temperature_savefig = f"temperatures_{varying_qty}.png"
    abundances_savefig = f"abundances_{varying_qty}.png"
    
    # Defining the figures
    fig_temp, ax_temp = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8))
    fig_abund, ax_abund = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8))
    
    # Plotting the figures
    for i in range(len(qtys)):
        prefix = prefixes[i]
        print(f"Plotting {prefix}...")
        qty = qtys[i]
        c = colors[i]
        av, tgas, tdust, HI, H2, CII, CI, CO = read_pdr_file(prefix + ".pdr.fin")
        print("Max av: ", np.max(av))
        print("Min av: ", np.min(av))
        print(np.shape(av))
        ax_temp.loglog(av, tgas, linestyle = temp_ls[0], color = c, label = rf"$T_{{gas}}, {latex_string} = {qty}$")
        ax_temp.loglog(av, tdust, linestyle = temp_ls[1], color = c, label = rf"$T_{{dust}}, {latex_string} = {qty}$")
        ax_abund.loglog(av, HI, linestyle = abund_ls[0], color = c, label = rf"[HI], ${latex_string} = {qty}$")
        ax_abund.loglog(av, H2, linestyle = abund_ls[1], color = c, label = rf"[H2], ${latex_string} = {qty}$")
        # ax_abund.loglog(av, CII, linestyle = abund_ls[2], color = c, label = rf"[CII], ${latex_string} = {qty}$")
        # ax_abund.loglog(av, CI, linestyle = abund_ls[3], color = c, label = rf"[CI], ${latex_string} = {qty}$")
        ax_abund.loglog(av, CO, linestyle = abund_ls[4], color = c, label = rf"[CO], ${latex_string} = {qty}$")
    
    # Saving the temperatures figure
    ax_temp.set_xlabel(r"$A_v$")
    ax_temp.set_ylabel("Temperature [K]")
    ax_temp.grid()
    ax_temp.legend(loc = "best")
    fig_temp.savefig(temperature_savefig, dpi = 1000)
    
    # Saving the abundances figure
    ax_abund.set_xlabel(r"$A_v$")
    ax_abund.set_xlim(1e-2, 1e2)
    ax_abund.set_ylabel("Abundances")
    ax_abund.legend(loc = "best")
    ax_abund.grid()
    fig_abund.savefig(abundances_savefig, dpi = 1000)
    
    print("DONE!!!!")
    
if __name__ in "__main__":
    prefixes_g_uv = ["OB_fiducial/OB_fiducial", "ob_uv_3/ob_uv_3", "ob_uv_2/ob_uv_2", 
                     "ob_uv_1/ob_uv_1"]
    prefixes_n_H = ["ob_nH_2/ob_nH_2", "ob_nH_3/ob_nH_3", "OB_fiducial/OB_fiducial", 
                    "ob_nH_5/ob_nH_5", "ob_nH_6/ob_nH_6"]
    
    temp_ls = ["solid", "dotted"]
    abund_ls = ["solid", "dotted", "dashed", "dashdot", (0, (1,10))]
    colors = ["blue", "green", "red", "yellow", "purple", "grey"]
    
    get_trend_plots(varying_qty = "g_uv", prefixes = prefixes_g_uv, linestyles = (temp_ls, abund_ls), 
                    colors = colors)
    get_trend_plots(varying_qty = "n_H", prefixes = prefixes_n_H, linestyles = (temp_ls, abund_ls),
                    colors = colors)
    # # Defining the plots
    # fig_temp, ax_temp = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7))
    # fig_abund, ax_abund = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7))
    
    # for i in range(4):
    #     prefix = prefixes[i]
    #     print(f"Plotting {prefix}...")
    #     g_uv = g_uvs[i]
    #     c = colors[i]
    #     av, tgas, tdust, HI, H2, CII, CI, CO = read_pdr_file(prefix + ".pdr.fin")
    #     ax_temp.semilogx(av, tgas, linestyle = temp_ls[0], color = c, label = r"$T_{gas}, G_{UV} = $" + f"{g_uv}")
    #     ax_temp.semilogx(av, tdust, linestyle = temp_ls[1], color = c, label = r"$T_{dust}, G_{UV} = $" + f"{g_uv}")
    #     ax_abund.loglog(av, HI, linestyle = abund_ls[0], color = c, label = r"[HI], $G_{UV} = $" + f"{g_uv}")
    #     ax_abund.loglog(av, H2, linestyle = abund_ls[1], color = c, label = r"[H2], $G_{UV} = $" + f"{g_uv}")
    #     ax_abund.loglog(av, CII, linestyle = abund_ls[2], color = c, label = r"[CII], $G_{UV} = $" + f"{g_uv}")
    #     ax_abund.loglog(av, CI, linestyle = abund_ls[3], color = c, label = r"[CI], $G_{UV} = $" + f"{g_uv}")
    #     ax_abund.loglog(av, CO, linestyle = abund_ls[4], color = c, label = r"[HI], $G_{UV} = $" + f"{g_uv}")
        
    # ax_temp.set_xlabel(r"$A_v$")
    # ax_temp.set_ylabel("Temperature [K]")
    # ax_temp.grid()
    # ax_temp.legend(loc = "best")
    # fig_temp.savefig(savefig_file_names[0], dpi = 1000)
    
    # ax_abund.set_xlabel(r"$A_v$")
    # ax_abund.set_ylabel("Abundances")
    # ax_abund.legend(loc = "best")
    # ax_abund.grid()
    # fig_abund.savefig(savefig_file_names[1], dpi = 1000)
    
    # print("DONE!!!")
    
                

