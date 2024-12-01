### Multi-expoential file with MuFLE version 2
# using MuFLE ver 2 (with l variable as the individual lifetimes)

import numpy as np
import pandas as pd

from TCSPC_fileload import FileLoad, TCSPCtime
from preprocessing import HistPreprocess, Processed_hist

from single_pixel_funcs import SE_single_pixel_residuals, single_pixel_fitting_func, SE_initial_guess

from spline_funcs import Basisfunc_Knots_IdealCoef, LS_fittingfunc

import matplotlib.pyplot as plt

""" File loading """
# IRF data loading
IRF_file_num = np.arange(1, 4)

IRF_all = []

for f in IRF_file_num:
    file_dict, tech_dict = FileLoad(FolderPath='/Users/alexadams/Documents/Flamingo/JupyterNotebooks/new_structure/all_EPTRFS_Data',
                                    FilePath='240417',
                                    FileType=f'/histogram_50ps_Q_RB_{f}_1.mat',
                                   WaveFilePath='/calibration_TRFS_system/lambdaMap.mat',
                                   TimeFilePath='/calibration_TRFS_system/TDCres.mat')


    Hist = file_dict['HistData']
    IRF_hist = Hist[0]
    IRF_all.append(np.flip(IRF_hist))

IRF_TCSPC_hist = np.sum(IRF_all, axis=0)
# IRF_TCSPC_hist = np.flip(IRF_hist)

# Hist data loading
# loading all files in the folder
file_num = np.arange(1, 4)
loc_num = np.arange(1, 7)

for l in loc_num:

    TCSPC_all = []

    for f in file_num:
        file_dict4, tech_dict4 = FileLoad(FolderPath='/Users/alexadams/Documents/Flamingo/JupyterNotebooks/new_structure/all_EPTRFS_Data',
                                   FilePath='240417',
                                   FileType=f'/histogram_50ps_T240417_ctrl_W1B_loc{l}_{f}_1.mat',
                                   WaveFilePath='/calibration_TRFS_system/lambdaMap.mat',
                                   TimeFilePath='/calibration_TRFS_system/TDCres.mat')

        Hist = file_dict4['HistData']
        TCSPC_hist1 = Hist[0]
        TCSPC_all.append(np.flip(TCSPC_hist1))
    # TCSPC_hist = (np.flip(TCSPC_hist1))

    TCSPC_time = TCSPCtime(TechDict=tech_dict4)

    corrected_wave1 = []
    for i in np.arange(1, 513, 1):
        corrected_wave1.append(0.51 * i + 474)

    corrected_wave2 = [0.51 * i + 474 for i in np.arange(1, 513, 1)]

    TCSPC_wave = np.reshape(np.repeat(corrected_wave1, 1200), (512, 1200), order='C')

    TCSPC_hist = np.sum(TCSPC_all, axis=0)

    """ Histogram preprocess """

    # pixel number: how many pixels to include in the final histogram, filter, a variable in HistPreprocess function
    # allows you to change where the histogram is cropped from at the beginning ie at 500nm or 520nm -- default is 552 nm
    # A filter at 152 and 150 pixel numbers means the preprocessed histogram goes from 552 nm - 628nm
    PixelNumber = 160

    Hist_pro, Time_pro, Wave_pro, IRF_pro, Wave_col = HistPreprocess(IRFRaw=IRF_TCSPC_hist, Hist=TCSPC_hist,
                                                                     Wave=TCSPC_wave, Time=TCSPC_time, Crop=PixelNumber, Filter=100)

    HistPeak = np.unravel_index(np.argmax(Hist_pro), np.shape(Hist_pro))

    Data = {'Hist': Hist_pro.flatten(), 'IRF': IRF_pro.flatten(), 'Time': Time_pro.flatten()}
    Data_File = pd.DataFrame(data=Data)

    # plotting the preprocessed histogram
    Observed_histogram = Processed_hist(Wave=Wave_pro, Time=Time_pro, Hist=Hist_pro)

    """ Single Pixel Fit Section """

    # Single pixel fit
    Single_pixel_all = []

    for Pixel, PixelIRF, decayTime, Wavei in zip(Hist_pro, IRF_pro, Time_pro, Wave_pro.T[1]):
        Params = SE_initial_guess(Int_initial_guess=1, Lt_initial_guess=4, Bias_initial_guess=-4)
        FitDecay, Title, Value, STD = single_pixel_fitting_func(params=Params, Residuals=SE_single_pixel_residuals, Time=decayTime, Data=Pixel, IRF=PixelIRF)
        WaveNM = []

        WaveNM.append(Wavei)

        Single_pixel_all.append(Value + STD + WaveNM)

    SE_FinalDF = pd.DataFrame(Single_pixel_all, columns=['Intensity', 'Lifetime', 'Bias', 'Intensity Variance', 'LifeTime Variance', 'Bias Variance', 'Wavelength'])

    """ The Splines Section """

    # Fits the knots and Basis functions are calculated from scipy with a given internal knot number and degree
    IntCoef, IntKnots, BfuncInt = Basisfunc_Knots_IdealCoef(Wave_axis=Wave_col, Decay_feature=SE_FinalDF['Intensity'], knot_num=3, degree=3)
    LtCoef, LtKnots, BfuncLt = Basisfunc_Knots_IdealCoef(Wave_axis=Wave_col, Decay_feature=SE_FinalDF['Lifetime'], knot_num=3, degree=3)

    # we just need the cubic function from this:
    cubic_coef, cubic_knots, cubic_func = Basisfunc_Knots_IdealCoef(Wave_axis=Wave_col, Decay_feature=SE_FinalDF['Lifetime'], knot_num=3, degree=3)

    # Starting parameters
    # prepping this for l exponentials:
    rng = np.random.default_rng(123145)

    l_run = 3
    p = len(Hist_pro)

    # iterating l times to create a list that is length l of random starting
    # parameters for tau and gamma
    rand_gamma = [rng.uniform(0, 1) for _ in range(l_run)]
    rand_tau = [rng.uniform(1, 4) for _ in range(l_run)]

    # Then repeating the l length list to be the size of the coefficients
    gamma_starting = [gamma for gamma in rand_gamma for _ in range(len(cubic_func))]

    # in the initial assessment we are using fixed lifetime for l:
    tau_starting = [tau for tau in rand_tau for _ in range(len(cubic_func))]

    # put them all together in one big starting parameter list
    initial_guess = gamma_starting + rand_tau + [-4] * PixelNumber

    # fitting functions
    SIntCoef, SLtCoef, SE_Spline_Bias, results = LS_fittingfunc(InitialGuess=initial_guess, Time=Time_pro, IRF=IRF_pro, Cubic_func=cubic_func,
                                                                Data=Hist_pro, l=l_run)

    """ Plotting """

    # spline results
    if l_run > 1:
        for l_results in range(l_run):
            gamma_coef = np.reshape(SIntCoef, (l_run, len(cubic_func)))
            gamma_Coef_results = [g @ BfuncInt for g in gamma_coef]

            tau_results = [[tau] * p for tau in SLtCoef]

            SE_FinalDF[f'Spline_int_{l_results}'] = gamma_Coef_results[l_results]
            SE_FinalDF[f'Spline_lifetime_{l_results}'] = tau_results[l_results]

    else:
        gamma_Coef_results = SIntCoef @ cubic_func
        tau_results = [SLtCoef] * p

        SE_FinalDF['Spline_int'] = gamma_Coef_results
        SE_FinalDF['Spline_lifetime'] = tau_results


    """Initial plotting"""

    Multi_color = ["#1F77B4", "#33A02C",  "#FF7F00", "#6A3D9A", "#008080"]

    if l_run > 1:
        for l_results, results_color in zip(range(l_run), Multi_color):
            plt.plot(SE_FinalDF['Wavelength'], tau_results[l_results], color=results_color, label='MuFLE', linewidth=4)
            plt.plot(SE_FinalDF['Wavelength'], tau_results[l_results], color=results_color, label='MuFLE', linewidth=4)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
        # plt.ylim([0, 4.0])
        plt.show()
        plt.close()

        for l_results, results_color in zip(range(l_run), Multi_color):
            plt.plot(SE_FinalDF['Wavelength'], gamma_Coef_results[l_results], color=results_color, label='MuFLE', linewidth=4)
        plt.plot(SE_FinalDF['Wavelength'], gamma_Coef_results[l_results], color=results_color, label='MuFLE', linewidth=4)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Intensity (a.u.)', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
        plt.ylim([0, 1.1])
        plt.show()
        plt.close()
    else:
        plt.plot(SE_FinalDF['Wavelength'], tau_results, color=Multi_color[-1], label='MuFLE', linewidth=4)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
        plt.ylim([0, 4.0])
        plt.show()
        plt.close()

        plt.plot(SE_FinalDF['Wavelength'], gamma_Coef_results, color=Multi_color[-1], label='MuFLE', linewidth=4)
        plt.xlabel('Wavelength (nm)', fontsize=18)
        plt.ylabel('Intensity (a.u.)', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
        plt.ylim([0, 1.1])
        plt.show()
        plt.close()

    SE_FinalDF.to_csv(f'/Users/alexadams/Desktop/240417_anal/multi_ctrl_W1B_loc{l}_Spline.csv',
                       header=True)