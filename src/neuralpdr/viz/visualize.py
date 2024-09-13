# Script to visualize NeuralODE runs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import time
from subprocess import call
from itertools import repeat
import matplotlib as mpl

# Pretty plots and a colour-blind friendly colour scheme
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

colors = {"blue":"#4477aa", "cyan":"#66ccee", "green":"#228833", "yellow":"#ccbb44",
          "red":"#ee6677", "purple":"#aa3377", "grey":"#bbbbbb"}
labels = ["HI", "CII", "CI", "CO"]

def parameters(model_index):
    """model_index is an integer
    """
    df = pd.read_csv("samples.csv")
    return df.iloc[model_index]

def display_loss(train_loss, val_loss, savefig_path):
    """Display training and validation loss functions
    savefig_path is the full path where the plot is saved.
    """
    epochs = len(train_loss)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True)
    ax.plot(np.arange(1, epochs + 1), train_loss, color = "red", label = "Train loss")
    ax.plot(np.arange(1, epochs + 1), val_loss, color = "green", label = "Val loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.set_ylabel("MSE Loss")
    ax.grid()
    fig.legend()
    fig.savefig(savefig_path, dpi = 300)
    
def display_predictions(pred, true, val_ind, fracs_array, path):
    """Display predictions and true values of time series in the validation set.
    val_ind is the list of model_indices of the validation set.
    """
    frac = fracs_array[-1]
    labels = ["HI", "CO"]
    for i in tqdm(range(len(val_ind))):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        val_index = val_ind[i]
        params = parameters(val_index).to_numpy()
        std_params = np.array([1e4, 1e4, 1.3e-17])
        standardized_params = params[1:]/std_params
        rounded_params = [np.round(standardized_params[j], 3) for j in range(len(params[1:]))]
        true_data, pred_data = true[i].T, pred[i].T
        for true_, pred_, label, color_key in zip(true_data, pred_data, labels, colors.keys()):
            length = len(val_av[i])
            ax.plot(val_av[i][:int(frac*length)], true_, label = label, color = colors[color_key], linestyle = "-", linewidth = 1.0)
            ax.plot(val_av[i][:int(frac*length)], pred_, color = colors[color_key], linestyle = "--", linewidth = 1.0)
        for f in fracs_array[:-1]:
            ax.axvline(x = val_av[i][int(f*length)], color = "red", linestyle = "--", linewidth = 0.5)
        ax.set_xlabel(r"$\overline{\log_{10}{A_v}}$")
        ax.set_ylabel(r"$\overline{\log_{10}{X}}$")
        ax.set_title(rf"$G_{{UV}} = {rounded_params[0]}G_{{UV_{{0}}}}, n_{{H}} = {rounded_params[1]}n_{{H_{{0}}}}, \zeta_{{CR}} = {rounded_params[2]}\zeta_{{CR_{{0}}}}$")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.grid()
        ax.legend(loc = "best")
        fig.savefig(f"{path}/model_{val_ind[i]}_pred_abundances.png", dpi = 300) 
        plt.close()
        
def plot_epoch_prediction(true_data, pred_data, epoch, path):
    true_data = np.swapaxes(true_data, 0, 1)
    pred_data = np.swapaxes(pred_data, 0, 1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    index = np.arange(np.shape(true_data)[1])
    for true_, pred_, label, color_key in zip(true_data, pred_data, labels, colors.keys()):
        ax.plot(index, true_, label = label, color = colors[color_key], linestyle = "-")
        ax.plot(index, pred_, color = colors[color_key], linestyle = "--")
    ax.set_xlabel(r"$\overline{\log_{10}{A_v}}$")
    ax.set_ylabel(r"$\overline{\log_{10}{X}}$")
    ax.set_title(f"Model 0: Epoch = {epoch}")
    fig.savefig(f"{path}/train_predictions/epoch_{epoch}.png", dpi = 300)
    plt.close()        

def display_predictions_all(model_00_true, model_00_pred, path):
    """Display model predictions as a function of epoch to see how the model trains.
    set_of_predictions contains predictions for all epochs.
    set_of_true_timeseries contains the true timeseries for all epochs.
    path is the folder where the plots are saved.
    """
    # print(np.shape(set_of_predictions), np.shape(set_of_true_timeseries))
    print("Plotting predictions obtained during training...")
    epochs = np.arange(1, len(model_00_pred) + 1, 1)
    # Defining the multiprocessing pool
    p = mp.Pool(processes = 8)
    start = time.time()
    p.starmap(plot_epoch_prediction, zip(repeat(model_00_true), model_00_pred, epochs,
                                         repeat(path)))
    end = time.time()
    print("Total time: ", end - start)
    
def visualize_weights(all_weights):
    """Display weights of different layers as a function of epoch. 
    """
    for layer, weight in enumerate(all_weights, start = 1):
        epoch, l1, l2 = np.shape(weight)
        weight = np.reshape(weight, (epoch, l1*l2))
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        img = ax.imshow(weight, cmap = "viridis", aspect = "auto")
        fig.colorbar(img)
        fig.savefig(f"weights/weight_{layer}.png", dpi = 300)
    
if __name__ in "__main__":
    
    path_numbers = np.arange(1,6)
    paths = [f"predictions_{number}" for number in path_numbers]
    df = pd.read_csv("samples.csv")
    df = df.rename(columns = {"Unnamed: 0": "model_index"})
    
    for path in paths:
        
        call(f"mkdir {path}/train_predictions", shell = True)
        val_ind = np.load(f"{path}/val_ind.npy")
        val_av = np.load(f"{path}/val_av.npy")
        print(np.shape(val_av))
        
        loss_functions = np.loadtxt(f"{path}/loss_function.csv", delimiter = ",")
        train_loss, val_loss = loss_functions[:, 0], loss_functions[:, 1]
        pred_data = np.load(f"{path}/eval_data_pred.npy")
        true_data = np.load(f"{path}/val_data_true.npy")
        loss_av = np.load(f"{path}/loss_av.npy")
        
        print("Plotting the train and validation loss functions as a function of epoch...")
        display_loss(train_loss, val_loss, savefig_path = f"{path}/loss_function.png")
        
        print("Making predictions on the validation set...")
        display_predictions(pred_data, true_data, val_ind, np.array([0.2, 0.4, 0.6, 0.8, 1.0]), path)
        
        print("Visualizing predictions while training and making a movie...")
        # set_of_predictions = np.swapaxes(set_of_predictions, 1, 2)
        # set_of_true_timeseries = np.swapaxes(set_of_true_timeseries, 1, 2)
        model_00_pred = np.load(f"{path}/model_00_pred.npy")
        model_00_true = np.load(f"{path}/model_00_true.npy")
        display_predictions_all(model_00_true, model_00_pred, path)
        call(f"ffmpeg -framerate 8 -i {path}/train_predictions/epoch_%d.png -r 30 {path}/predictions_all.mp4", shell = True)
        
        print("Plotting final loss as a function of visual extinction...")
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        index = np.arange(len(loss_av))
        ax.plot(index, loss_av)
        ax.set_xlabel("Index (scaled $A_v$)")
        ax.set_ylabel("MSE loss")
        ax.set_title("Loss vs scaled $A_v$")
        ax.grid()
        fig.savefig(f"{path}/loss_vs_av.png", dpi = 300)
    


