# Preprocessing and plotting functions
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def IRFBack(IRFHist):
    """ Function to remove the dark count rates of the IRF to 0 -- setting a 0 bias
    :param IRFHist: The 2D IRF histogram measured from the quenched Rose-bengal solution
    This function calculated the mean of the first 2 time bins (containing no signal)
    The function then tiles the average dark count rate and subtracts it from the IRF histogram
    :return: The IRF with the dark counts removed
    """
    DarkCount = np.mean(IRFHist[:, 0:2], axis=1)
    DarkCountRep = np.transpose(np.stack([DarkCount] * len(IRFHist.T)))
    Sub = np.array(np.subtract(IRFHist, DarkCountRep))
    Sub[Sub < 0] = 0
    return Sub

def SumNorm(Decay):
    """ Function to normalise a decay using the summation normalisation method
    :param Decay: One decay from the original histogram
    :return: The normalised decay
    """
    a = Decay - np.min(Decay)
    return(a/np.sum(a))

def MaxNorm(Decay):
    """ Function to normalise a decay or histogram using the maximum normalisation method
    :param: One decay (or the entire histogram) from the original histogram
    :return: The normalised decay (or histogram)
    """
    a = Decay - np.min(Decay)
    b = np.max(Decay) - np.min(Decay)
    return a / b

def HistPreprocess(IRFRaw, Hist, Wave, Time, Crop, Filter):
    """ Function that preprocesses the IRF, Time, Wave and histogram, preparing them for further analysis
    :param IRFRaw: The measured IRF histogram
    :param Hist: The measured data histogram
    :param Wave: The total wavelength variable
    :param Time: The total time variable
    :param Crop: The higher wavelength range to crop the histogram at-- also dictates the number of channels
    the histogram is cropped into
    :param Filter: Which lower wavelength range to crop this histogram -- removing the area where no
    photons are collected due to the band-pass filter but is adjustable for fluorophores like
    Fluorescien -- traditionally a filter of 152 channels is used removing the first 152 decays
    :return: Cropped and normalised IRF, histogram, equilivent time and wavelengths. In addition
    to the total knots used in the splines analysis
    """
    # IRF background removed
    IRF = IRFBack(IRFRaw)

    # Removing the last 10 time bins
    # Crop the histogram and wavelength so only the decay is present
    # Remove the time bins where no spectrum is either (ie before the peak)
    Crop = Filter + Crop
    Wavecrop, Timecrop, Histcrop, IRFcrop = Wave[Filter:Crop, 700:-10], Time[Filter:Crop, :490], Hist[Filter:Crop, 700:-10], IRF[Filter:Crop, 700:-10]

    # Normalise the IRF using the summed norm method
    NormIRF = []

    for pix in IRFcrop:
        NormIRF.append(SumNorm(Decay=pix))

    NormIRF = np.array(NormIRF)

    # Normalize the histogram
    NormHist = MaxNorm(Histcrop)

    Conv = []

    for row in range(NormHist.shape[0]):
        ModelConv = convolve(NormIRF[row], NormHist[row], mode='full', method='direct')
        Conv.append(ModelConv[:len(NormHist.T)])

    Conv = np.array(Conv)

    IRFcropFinal = NormIRF/np.max(Conv)
    IRFNew = IRFcropFinal

    # Splines Wavedummy
    WaveDummy = np.reshape(np.repeat(np.linspace(0.01, 0.99, num=len(NormHist)), len(NormHist.T)), (len(NormHist), len(NormHist.T)), order='C')

    # Joint Estimation
    for i in WaveDummy.T:
        WaveCol = i

    return NormHist, Timecrop, Wavecrop, IRFNew, WaveCol

# plotting functions
def Processed_hist(Wave, Time, Hist):
    """ Function to surfave 3D plot a histogram
    :param Wave: Wavelength histogram
    :param Time: Time histogram
    :param Hist: Data histogram
    :return: The 3D surface plot figure
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(Wave, Time, Hist, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Time (ns)', fontsize=14)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Intensity (a.u.)', fontsize=14, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    ax.view_init(30, 300)
    # plt.savefig('LottoC_ep1_hist.png', format='png', dpi=600)
    plt.show()

# single exponential
def Int_Muf(Single_channel, Hist, Spline_int, Int_CI, Muf_colour):
    """ Function to plot the MuFLE intensity on one y axis and the plater reader intensity on another
    :param Single_channel: Single channel fit data frame
    :param Hist: Pre-processed histogram data
    :param Spline_int: Splines intensity result
    :param Int_CI: Splines intensity confidence interval
    :param Muf_colour: Mufle colour
    :return: The plotted figure
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(Single_channel['Wavelength'], MaxNorm(np.sum(Hist, axis=1)), color='#636363', alpha=0.6, linewidth=2)
    ax2.plot(Single_channel['Wavelength'], Spline_int, color=Muf_colour, label='MuFLE', linewidth=4)
    ax2.fill_between(Single_channel['Wavelength'], (Spline_int - Int_CI), (Spline_int + Int_CI),
                     color=Muf_colour, alpha=.1)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax1.set_ylabel('Observed Intensity (a.u.)', fontsize=18, color='#636363')
    ax2.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)
    # plt.legend(fontsize=15)
    ax1.tick_params(axis='y', width=3, colors='#636363', labelsize='large')
    ax2.tick_params(axis='y', width=3, colors=Muf_colour, labelsize='large')
    plt.xticks(fontsize=18)
    plt.tight_layout()
    ax1.set_ylim([(np.min(MaxNorm(np.sum(Hist, axis=1)))), (np.max(MaxNorm(np.sum(Hist, axis=1)) * 1.21))])
    ax2.set_ylim([(np.min(Spline_int) * 0.9), (np.max(Spline_int) * 1.16)])
    plt.show()
    plt.close()

def Int_Muf_PR(PlateReader, Sample, bluerange, Single_channel, Hist, Spline_int, Int_CI, Muf_colour, PR_colour):
    """ Function to plot the MuFLE intensity on one y axis and the plater reader intensity on another
    :param PlateReader: Plate reader data file with both wavelength and fluorophore column
    :param Sample: What fluorophore it was named on the plate reader
    :param bluerange: The blue wavelegnth range to crop the plate reader at -- RhB = 40, Fluorescein = 14
    :param Single_channel: Single channel fit data frame
    :param Hist: Pre-processed histogram data
    :param Spline_int: Splines intensity result
    :param Int_CI: Splines intensity confidence interval
    :param Muf_colour: Mufle colour
    :param PR_colour: Plate reader colour
    :return: The plotted figure
    """

    ymin1 = 0
    ymax1 = np.max(MaxNorm(np.sum(Hist, axis=1)) * 0.9)

    ymin2 = np.min(Spline_int) * 0.3
    ymax2 = np.max(Spline_int) * 1.3

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(PlateReader['Wavelength'][bluerange:116], MaxNorm(PlateReader[Sample][bluerange:116]),
             color=PR_colour, label='Plate Reader', linewidth=4)
    ax2.scatter(Single_channel['Wavelength'], Single_channel['SP Intensity'], color='#636363', label='Single Pixel Fit',
                alpha=0.2)
    ax1.plot(Single_channel['Wavelength'], Spline_int, color=Muf_colour, label='MuFLE', linewidth=4)
    ax1.fill_between(Single_channel['Wavelength'], (Spline_int - Int_CI), (Spline_int + Int_CI),
                     color=Muf_colour, alpha=.1)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax2.set_ylabel('Plate Reader Intensity (a.u.)', fontsize=18, color=PR_colour)
    ax1.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)
    ax2.tick_params(axis='y', width=2, colors=PR_colour, labelsize='large')
    ax1.tick_params(axis='y', width=2, colors=Muf_colour, labelsize='large')

    ax2.set_ylim([0, 1.2])
    ax1.set_ylim([0.25, 1.0])
    ax1.legend(fontsize=15, loc='upper left')
    ax2.legend(fontsize=15, loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()

def Lt_Muf_SC(Single_channel, Spline_lt, Lt_CI, Muf_colour):
    """ Function to plot the mufle lifetime result compared to the single channel -- with no y axis limit
    :param Single_channel: Single channel results
    :param Spline_lt: Spline lifetime results
    :param Lt_CI: Spline lifetime confidence intervals
    :param Muf_colour: MuFLE colour
    :return: The plotted figure
    """

    plt.scatter(Single_channel['Wavelength'], Single_channel['Lifetime'], color='#636363', s=50, label='Single Pixel Fit',
                alpha=0.3)
    plt.plot(Single_channel['Wavelength'], Spline_lt, color=Muf_colour, label='MuFLE', linewidth=4)
    plt.fill_between(Single_channel['Wavelength'], (Spline_lt - Lt_CI), (Spline_lt + Lt_CI),
                     color=Muf_colour, alpha=.4)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    plt.ylim([0, 4])
    # plt.savefig('LottoC_Lt.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def Lt_Muf_SC_literature(Single_channel, Spline_lt, Lt_CI, Lit_val, Muf_colour):
    """ Function to plot the lifetime from single channel, splines, compared to the literature value
    :param Single_channel: Single channel results
    :param Spline_lt: Splines lifetime result
    :param Lt_CI: Splines lifetime confidence intervals
    :param Lit_val: Array with the literature values to be plotted
    :return: The plotted figure
    """
    fig, ax = plt.subplots()
    ax.scatter(Single_channel['Wavelength'], Single_channel['SP Lifetime'], color='#636363', s=50, label='Single Pixel Fit',
                alpha=0.3)
    ax.plot(Single_channel['Wavelength'], Spline_lt, color=Muf_colour, label='MuFLE', linewidth=4)
    ax.fill_between(Single_channel['Wavelength'], (Spline_lt - Lt_CI), (Spline_lt + Lt_CI),
                     color=Muf_colour, alpha=.4)

    arrow1 = patches.FancyArrowPatch((631,  Lit_val[0]), (634, Lit_val[0]), arrowstyle='<|-', lw=3, color='#6baed6', mutation_scale=20,
                                         clip_on=False)
    arrow2 = patches.FancyArrowPatch((631, Lit_val[1]), (634, Lit_val[1]), arrowstyle='<|-', lw=3, color='#6baed6', mutation_scale=20,
                                         clip_on=False)
    arrow3 = patches.FancyArrowPatch((631, Lit_val[2]), (634, Lit_val[2]), arrowstyle='<|-', lw=3, color='#6baed6', mutation_scale=20,
                                         clip_on=False)
    plt.ylim([0, 4.52])
    plt.xlim([550, 630])
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    # plt.ylim([0, 4.52])
    # plt.savefig('FINALFLUO3_LT.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def Decay(Spline_int, Spline_lt, m, IRF, Obs_hist, bias):
    """ Function to plot a single channel decay
    :param Spline_int: A single channel splines intensity result
    :param Spline_lt: A single channel splines lifetime result
    :param m: A single channel time axis
    :param IRF: A single channel IRF
    :param Obs_hist: The single channel observed decay
    :param bias: The single channel bias term
    :return: The plotted figure
    """

    Decay = Spline_int * np.exp(-m / Spline_lt)
    Conv = convolve(Decay, IRF, mode='full', method='direct')

    plt.plot(m, Obs_hist, color='#377eb8')
    plt.plot(m, Conv[:len(m)] + np.exp(bias), color='#e41a1c')
    plt.xlabel('Time (ns)', fontsize=15)
    plt.ylabel('Intensity (a.u.)', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.savefig('RhBWater_Fit577nm.png', format='png', dpi=1200)
    plt.show()

def Decay_IRF(Spline_int, Spline_lt, m, IRF, Obs_hist, bias):
    """ Function to plot a single channel decay
    :param Spline_int: A single channel splines intensity result
    :param Spline_lt: A single channel splines lifetime result
    :param m: A single channel time axis
    :param IRF: A single channel IRF
    :param Obs_hist: The single channel observed decay
    :param bias: The single channel bias term
    :return: The plotted figure
    """

    Decay = Spline_int * np.exp(-m / Spline_lt)
    Conv = convolve(Decay, IRF, mode='full', method='direct')

    plt.plot(m, Obs_hist, linewidth=3, color='#377eb8')
    plt.plot(m, IRF, linewidth=3, color='#ab8b04')
    plt.plot(m, Conv[:len(m)] + np.exp(bias), linewidth=3,color='#e41a1c')
    plt.xlabel('Time (ns)', fontsize=15)
    plt.ylabel('Intensity (a.u.)', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.savefig('RhBWater_Fit577nm.png', format='png', dpi=1200)
    plt.show()

def Bias(Single_channel, bias):
    """ Function to plot the Splines bias values
    :param Single_channel: Single channel data frame results
    :param bias: Splines bias values
    :return: The plotted figure
    """
    plt.scatter(Single_channel['Wavelength'], np.exp(bias), color='#66c2a4', label='MuFLE', s=40)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Bias")
    plt.tight_layout()
    # plt.savefig('Fluo_Bias.png', format='png', dpi=1200)
    plt.show()

def Lt_Muf_SCNoCI(Single_channel, Spline_lt, Muf_colour):
    """ Function to plot the mufle lifetime result compared to the single channel -- with no y axis limit
    :param Single_channel: Single channel results
    :param Spline_lt: Spline lifetime results
    :param Muf_colour: MuFLE colour
    :return: The plotted figure
    """

    plt.scatter(Single_channel['Wavelength'], Single_channel['Lifetime'], color='#636363', s=50, label='Single Pixel Fit',
                alpha=0.3)
    plt.plot(Single_channel['Wavelength'], Spline_lt, color=Muf_colour, label='MuFLE', linewidth=4)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0, 4])
    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    # plt.savefig('LottoC_Lt.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def Int_MufNoCI(Single_channel, Spline_int, Muf_colour):
    """ Function to plot the MuFLE intensity on one y axis and the plater reader intensity on another
    :param Single_channel: Single channel fit data frame
    :param Hist: Pre-processed histogram data
    :param Spline_int: Splines intensity result
    :param Muf_colour: Mufle colour
    :return: The plotted figure
    """
    plt.plot(Single_channel['Wavelength'], Single_channel['Intensity'], color='#636363', alpha=0.6, linewidth=2)
    plt.plot(Single_channel['Wavelength'], Spline_int, color=Muf_colour, label='MuFLE', linewidth=4)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.close()

def IntHist_MufNoCI(Single_channel, Hist, Spline_int, Muf_colour):
    """ Function to plot the MuFLE intensity on one y axis and the plater reader intensity on another
    :param Single_channel: Single channel fit data frame
    :param Hist: Pre-processed histogram data
    :param Spline_int: Splines intensity result
    :param Int_CI: Splines intensity confidence interval
    :param Muf_colour: Mufle colour
    :return: The plotted figure
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(Single_channel['Wavelength'], MaxNorm(np.sum(Hist, axis=1)), color='#636363', alpha=0.6, linewidth=2)
    ax2.plot(Single_channel['Wavelength'], Spline_int, color=Muf_colour, label='MuFLE', linewidth=4)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax1.set_ylabel('Observed Intensity (a.u.)', fontsize=18, color='#636363')
    ax2.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)
    # plt.legend(fontsize=15)
    plt.ylabel('Intensity (a.u)')
    ax1.tick_params(axis='y', width=3, colors='#636363', labelsize='large')
    ax2.tick_params(axis='y', width=3, colors=Muf_colour, labelsize='large')
    plt.xticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.close()

# subplots
def Int_lt_subplot(PlateReader, Sample, bluerange, Single_channel, Hist, Spline_int, Int_CI, Muf_colour, PR_colour, Spline_lt, Lt_CI, Lit_val):

    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_figheight(8)
    fig.set_figwidth(6)
    ax3 = ax1.twinx()

    ax3.plot(PlateReader['Wavelength'][bluerange:116], MaxNorm(PlateReader[Sample][bluerange:116]),
             color=PR_colour, label='Plate Reader', linewidth=4)
    # ax3.plot(Single_channel['Wavelength'], MaxNorm(np.sum(Hist, axis=1)), color='#636363', label='Single Pixel Fit',alpha=0.2)
    ax3.scatter(Single_channel['Wavelength'], Single_channel['Intensity'], color='#636363', label='Single Pixel Fit', alpha=0.2, s=50)
    ax1.plot(Single_channel['Wavelength'], Spline_int, color=Muf_colour, label='MuFLE', linewidth=4)
    ax1.fill_between(Single_channel['Wavelength'], (Spline_int - Int_CI), (Spline_int + Int_CI),
                     color=Muf_colour, alpha=.1)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax3.set_ylabel('Plate Reader Intensity (a.u.)', fontsize=18, color=PR_colour)
    ax1.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)
    ax3.tick_params(axis='y', width=2, colors=PR_colour, labelsize='large')
    ax1.tick_params(axis='y', width=2, colors=Muf_colour, labelsize='large')
    # ax1.set_xticks(ticks=np.arange(550, 640, 10))
    ax1.tick_params(axis='both', which='major', labelsize=15)
    # ax3.legend(loc='upper right', fontsize=13)
    # ax1.legend(loc='upper left', fontsize=13)
    ax3.set_ylim([0, 1.2])
    ax1.set_ylim([0.201, 1.0])
    ax1.set_xlim([515, 622])


    ax2.scatter(Single_channel['Wavelength'], Single_channel['Lifetime'], color='#636363', s=50,
               label='Single Pixel Fit',
               alpha=0.3)
    ax2.plot(Single_channel['Wavelength'], Spline_lt, color=Muf_colour, label='MuFLE', linewidth=4)
    ax2.fill_between(Single_channel['Wavelength'], (Spline_lt - Lt_CI), (Spline_lt + Lt_CI),
                    color=Muf_colour, alpha=.4)

    arrow1 = patches.FancyArrowPatch((622, Lit_val[0]), (625, Lit_val[0]), arrowstyle='<|-', lw=3, color='#6DCAA2',
                                     mutation_scale=20,
                                     clip_on=False)
    arrow2 = patches.FancyArrowPatch((622, Lit_val[1]), (625, Lit_val[1]), arrowstyle='<|-', lw=3, color='#6DCAA2',
                                     mutation_scale=20,
                                     clip_on=False)
    arrow3 = patches.FancyArrowPatch((622, Lit_val[2]), (625, Lit_val[2]), arrowstyle='<|-', lw=3, color='#6DCAA2',
                                     mutation_scale=20,
                                     clip_on=False)
    ax2.set_ylim([0, 4.2])
    ax2.set_xlim([515, 622])
    ax2.add_patch(arrow1)
    ax2.add_patch(arrow2)
    ax2.add_patch(arrow3)
    ax2.set_xlabel('Wavelength (nm)', fontsize=18)
    # ax2.set_xticks(ticks=np.arange(550, 640, 10))
    ax2.set_ylabel('Fluorescence lifetime (ns)', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    # plt.savefig('FluoSingle.png', format='png', dpi=1200)
    plt.show()


# double exponential
def DE_lt_Muf_SC(Single_channel, Spline_lt1, Spline_lt2, Lt_CI1, Lt_CI2, Muf_colour):
    """ Function to show double exponential lifetime vs single channel fit
    :param Single_channel: Single channel result
    :param Spline_lt1: Spline lifetime 1
    :param Spline_lt2: Spline lifetime 2
    :param Lt_CI1: Spline lifetime 1 confidence interval
    :param Lt_CI2: Spline lifetime 2 confidence interval
    :return: The plotted figure
    """
    plt.scatter(Single_channel['Wavelength'], Single_channel['Lifetime1'], color='#636363', s=50, alpha=0.4)
    plt.scatter(Single_channel['Wavelength'], Single_channel['Lifetime2'], color='#636363', s=50, label='Single Channel Fit',
                alpha=0.4)
    plt.plot(Single_channel['Wavelength'], Spline_lt1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    plt.plot(Single_channel['Wavelength'], Spline_lt2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')
    plt.fill_between(Single_channel['Wavelength'], (Spline_lt1 - Lt_CI1), (Spline_lt1 + Lt_CI1),
                     color=Muf_colour, alpha=.1)
    plt.fill_between(Single_channel['Wavelength'], (Spline_lt2 - Lt_CI2), (Spline_lt2 + Lt_CI2),
                     color=Muf_colour, alpha=.1)
    plt.xlabel('Wavelength (nm)', fontsize=15)
    plt.ylabel('Fluorescence Lifetime (ns)', fontsize=15)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    plt.ylim([0, 4.5])
    # plt.savefig('Lotto2_Lt.png', format='png', dpi=1200)
    # plt.figure(figsize=(10.5, 8))
    plt.show()
    plt.close()

def Int_Muf_PRInt_Muf_PR(PlateReader1, PlateReader2, Sample1, Sample2, Single_channel, Spline_int1, Spline_int2, Int_CI1, Int_CI2, pOfs, Muf_colour, PR_colour):
    """ Function to plot mufle double exponential results vs the plate reader for the individual samples
    :param PlateReader1: Plate reader results from one of the fluorophores
    :param PlateReader2: Plate reader results from the other fluorophore
    :param Sample1: Sample 1 ie 'RhB Methanol' the title of the channels from the plate reader
    :param Sample2: Sample 1 ie 'Fluo' the title of the channels from the plate reader
    :param Single_channel: The single channel fit results
    :param Spline_int1: Splines results from intensity 1
    :param Spline_int2: Splines results from intensity 2
    :param Int_CI1: Splines intensity 1 confidence intervals
    :param Int_CI2: Splines intensity 2 confidence intervals
    :param pOfs: What ratio of the distance between max and min points you want
    :param Muf_colour: Mufle Colour
    :param PR_colour: Plate reader colour
    :return: The plotted figure
    """

    ymin1 = ((1 - pOfs) * min(MaxNorm(PlateReader1[Sample1][40:116])) - pOfs * max(MaxNorm(PlateReader1[Sample1][40:116]))) / (1 - 2 * pOfs)
    ymax1 = ((1 - pOfs) * max(MaxNorm(PlateReader2[Sample2][40:116])) - pOfs * min(MaxNorm(PlateReader2[Sample2][40:116]))) / (1 - 2 * pOfs)

    ymin2 = ((1 - pOfs) * min(Spline_int1) - pOfs * max(Spline_int1)) / (1 - 2 * pOfs)
    ymax2 = ((1 - pOfs) * max(Spline_int2) - pOfs * min(Spline_int2)) / (1 - 2 * pOfs)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(PlateReader1['Wavelength'][40:120], MaxNorm(PlateReader1[Sample1][40:120]), color=PR_colour, label='Plate Reader', linewidth=4, linestyle='--')
    ax1.plot(PlateReader2['Wavelength'][40:120], MaxNorm(PlateReader2[Sample2][40:120]), color=PR_colour, label='Plate Reader', linewidth=4, linestyle=':')

    ax2.plot(Single_channel['Wavelength'], Spline_int1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    ax2.plot(Single_channel['Wavelength'], Spline_int2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')
    ax2.fill_between(Single_channel['Wavelength'], (Spline_int1 - Int_CI1), (Spline_int1 + Int_CI1),
                     color=Muf_colour, alpha=.1)
    ax2.fill_between(Single_channel['Wavelength'], (Spline_int2 - Int_CI2), (Spline_int2 + Int_CI2),
                     color=Muf_colour, alpha=.1)

    ax1.set_xlabel('Wavelength (nm)', fontsize=15)
    ax1.set_ylabel('Plate Reader Intensity (a.u.)', fontsize=18, color='#238b45')
    ax2.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)

    ax1.tick_params(axis='y', width=2, colors=PR_colour, labelsize='large')
    ax2.tick_params(axis='y', width=2, colors=Muf_colour, labelsize='large')

    plt.tight_layout()
    ax1.set_ylim([ymin1, ymax1])
    ax2.set_ylim([ymin2, ymax2])
    # plt.savefig('LottoC_int.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def Int_Muf_DE(Single_channel, Spline_int1, Spline_int2, Int_CI1, Int_CI2, Muf_colour):
    """ Function to plot mufle double exponential results vs the plate reader for the individual samples
    :param Single_channel: The single channel fit results
    :param Spline_int1: Splines results from intensity 1
    :param Spline_int2: Splines results from intensity 2
    :param Int_CI1: Splines intensity 1 confidence intervals
    :param Int_CI2: Splines intensity 2 confidence intervals
    :param Muf_colour: Mufle Colour
    :return: The plotted figure
    """

    plt.plot(Single_channel['Wavelength'], Spline_int1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    plt.plot(Single_channel['Wavelength'], Spline_int2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')
    plt.fill_between(Single_channel['Wavelength'], (Spline_int1 - Int_CI1), (Spline_int1 + Int_CI1),
                     color=Muf_colour, alpha=.1)
    plt.fill_between(Single_channel['Wavelength'], (Spline_int2 - Int_CI2), (Spline_int2 + Int_CI2),
                     color=Muf_colour, alpha=.1)

    plt.xlabel('Wavelength (nm)', fontsize=15)
    plt.ylabel('Fitted Intensity (a.u.)', fontsize=18)

    plt.tight_layout()
    # plt.savefig('LottoC_int.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def DE_Decay(Spline_int1, Spline_lt1, Spline_int2, Spline_lt2, m, IRF, Obs_hist, bias):
    """ Function to plot a single channel decay
    :param Spline_int1: A single channel splines intensity 1 result
    :param Spline_lt1: A single channel splines lifetime 1 result
    :param Spline_int2: A single channel splines intensity 2 result
    :param Spline_lt2: A single channel splines lifetime 2 result
    :param m: A single channel time axis
    :param IRF: A single channel IRF
    :param Obs_hist: The single channel observed decay
    :param bias: The single channel bias term
    :return: The plotted figure
    """

    Decay = Spline_int1 * np.exp(-m / Spline_lt1) + Spline_int2 * np.exp(-m / Spline_lt2)
    Conv = convolve(Decay, IRF, mode='full', method='direct')

    plt.plot(m, Obs_hist, color='#377eb8', linewidth=2)
    plt.plot(m, Conv[:len(m)] + np.exp(bias), color='#e41a1c', linewidth=2)
    plt.xlabel('Time (ns)', fontsize=15)
    plt.ylabel('Intensity (a.u.)', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.savefig('RhBWater_Fit577nm.png', format='png', dpi=1200)
    plt.show()

def DE_lt_MufNoCI(Single_channel, Spline_lt1, Spline_lt2, Muf_colour):
    """ Function to show double exponential lifetime vs single channel fit
    :param Single_channel: Single channel result
    :param Spline_lt1: Spline lifetime 1
    :param Spline_lt2: Spline lifetime 2
    :return: The plotted figure
    """
    plt.scatter(Single_channel['Wavelength'], Single_channel['Lifetime1'], color='#636363', s=50, alpha=0.4)
    plt.scatter(Single_channel['Wavelength'], Single_channel['Lifetime2'], color='#636363', s=50, label='Single Channel Fit',
                alpha=0.4)
    plt.plot(Single_channel['Wavelength'], Spline_lt1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    plt.plot(Single_channel['Wavelength'], Spline_lt2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')
    plt.xlabel('Wavelength (nm)', fontsize=15)
    plt.ylabel('Fluorescence Lifetime (ns)', fontsize=15)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    plt.ylim([0, 4.5])
    # plt.savefig('Lotto2_Lt.png', format='png', dpi=1200)
    # plt.figure(figsize=(10.5, 8))
    plt.show()
    plt.close()

def DE_Int_MufNoCI(Single_channel, Spline_int1, Spline_int2, Muf_colour):
    """ Function to plot mufle double exponential results vs the plate reader for the individual samples
    :param Single_channel: The single channel fit results
    :param Spline_int1: Splines results from intensity 1
    :param Spline_int2: Splines results from intensity 2
    :param Int_CI1: Splines intensity 1 confidence intervals
    :param Int_CI2: Splines intensity 2 confidence intervals
    :param Muf_colour: Mufle Colour
    :return: The plotted figure
    """

    plt.plot(Single_channel['Wavelength'], Spline_int1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    plt.plot(Single_channel['Wavelength'], Spline_int2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')

    plt.xlabel('Wavelength (nm)', fontsize=15)
    plt.ylabel('Fitted Intensity (a.u.)', fontsize=18)

    plt.tight_layout()
    # plt.savefig('LottoC_int.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def Multi_File_Int(File_List, File_name):

    for label, File_DF in enumerate(File_List):
        plt.plot(File_DF['Wavelength'], File_DF['Spline Int'], color=f'#FF{int(label * 3)}733', linewidth=4, label=f'P{int(label) + 1}')
        plt.fill_between(File_DF['Wavelength'], (File_DF['Spline Int'] - File_DF['Spline Int CI']), (File_DF['Spline Int'] + File_DF['Spline Int CI']), color=f'#FF{int(label * 3)}733', alpha=.4)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fitted Intensity (a.u.)', fontsize=18)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.ylim([0, 1.1])
    plt.savefig(f'Int_{File_name}.png', format='png', dpi=1200)
    plt.show()

def Multi_File_Lt(File_List, File_name):

    for label, File_DF in enumerate(File_List):
        plt.plot(File_DF['Wavelength'], File_DF['Spline Lifetime'], color=f'#FF{int(label * 3)}733', linewidth=4, label=f'P{int(label) + 1}')
        plt.fill_between(File_DF['Wavelength'], (File_DF['Spline Lifetime'] - File_DF['Spline Lifetime CI']), (File_DF['Spline Lifetime'] + File_DF['Spline Lifetime CI']), color=f'#FF{int(label * 3)}733', alpha=.4)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'Lt_{File_name}.png', format='png', dpi=1200)
    plt.show()

# subplots
# the first one is comparing the single expo Int vs double expo Int
def Sing_doub_int(SE_Single_channel, SE_Spline_Int, SE_SP_Int, PlateReader1, PlateReader2, Sample1, Sample2, DE_Single_channel, DE_Spline_int1, DE_Spline_int2, DE_Int_CI1, DE_Int_CI2, pOfs, Muf_colour, PR_colour):

    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_figheight(8)
    fig.set_figwidth(6)
    ax3 = ax2.twinx()

    ymin1 = ((1 - pOfs) * min(MaxNorm(PlateReader1[Sample1][40:116])) - pOfs * max(
        MaxNorm(PlateReader1[Sample1][40:116]))) / (1 - 2 * pOfs)
    ymax1 = ((1 - pOfs) * max(MaxNorm(PlateReader2[Sample2][40:116])) - pOfs * min(
        MaxNorm(PlateReader2[Sample2][40:116]))) / (1 - 2 * pOfs)

    ymin2 = ((1 - pOfs) * min(DE_Spline_int1) - pOfs * max(DE_Spline_int1)) / (1 - 2 * pOfs)
    ymax2 = ((1 - pOfs) * max(DE_Spline_int2) - pOfs * min(DE_Spline_int2)) / (1 - 2 * pOfs)

    ax1.scatter(SE_Single_channel['Wavelength'], SE_SP_Int, color='#636363', label='Single Pixel Fit', alpha=0.2, s=50)
    ax1.plot(SE_Single_channel['Wavelength'], SE_Spline_Int, color=Muf_colour, linewidth=4)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=18)
    # ax2.set_xticks(ticks=np.arange(550, 640, 10))
    ax1.set_xlim([550, 630])


    ax2.plot(PlateReader1['Wavelength'][40:120], MaxNorm(PlateReader1[Sample1][40:120]), color=PR_colour,
             label='Plate Reader', linewidth=4, linestyle='--')
    ax2.plot(PlateReader2['Wavelength'][40:120], MaxNorm(PlateReader2[Sample2][40:120]), color=PR_colour,
             label='Plate Reader', linewidth=4, linestyle=':')

    ax3.plot(DE_Single_channel['Wavelength'], DE_Spline_int1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    ax3.plot(DE_Single_channel['Wavelength'], DE_Spline_int2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')
    ax3.fill_between(DE_Single_channel['Wavelength'], (DE_Spline_int1 - DE_Int_CI1), (DE_Spline_int1 + DE_Int_CI1),
                     color=Muf_colour, alpha=.1)
    ax3.fill_between(DE_Single_channel['Wavelength'], (DE_Spline_int2 - DE_Int_CI2), (DE_Spline_int2 + DE_Int_CI2),
                     color=Muf_colour, alpha=.1)

    ax2.set_xlabel('Wavelength (nm)', fontsize=18)
    ax2.set_ylabel('Plate Reader Intensity (a.u.)', fontsize=18, color=PR_colour)
    ax3.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)

    ax2.tick_params(axis='y', width=2, colors=PR_colour, labelsize='large')
    ax3.tick_params(axis='y', width=2, colors=Muf_colour, labelsize='large')

    ax2.tick_params(axis='both', which='major', labelsize=15)
    # ax2.set_xticks(ticks=np.arange(550, 640, 10))
    ax2.set_xlim([550, 630])

    ax2.set_ylim([ymin1, ymax1])
    ax3.set_ylim([ymin2, ymax2])

    plt.tight_layout()
    # plt.savefig('RHBWaterFluo_IntDoub.png', format='png', dpi=1200)

    plt.show()
    plt.close()

def Sing_doub_lt(SE_Single_channel, DE_Single_channel, Spline_lt, Lt_CI, Muf_colour, Spline_lt1, Spline_lt2, Lt_CI1, Lt_CI2):

    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_figheight(8)
    fig.set_figwidth(5)

    ax1.scatter(SE_Single_channel['Wavelength'], SE_Single_channel['Lifetime'], color='#636363', s=50,
                label='Single Pixel Fit',
                alpha=0.3)
    ax1.plot(SE_Single_channel['Wavelength'], Spline_lt, color=Muf_colour, label='MuFLE', linewidth=4)
    ax1.fill_between(SE_Single_channel['Wavelength'], (Spline_lt - Lt_CI), (Spline_lt + Lt_CI),
                     color=Muf_colour, alpha=.4)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax1.set_ylabel('Fluorescence lifetime (ns)', fontsize=18)
    # plt.legend(fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_ylim([0, 4.5])


    ax2.scatter(DE_Single_channel['Wavelength'], DE_Single_channel['Lifetime1'], color='#636363', s=50, alpha=0.4)
    ax2.scatter(DE_Single_channel['Wavelength'], DE_Single_channel['Lifetime2'], color='#636363', s=50,
                label='Single Channel Fit',
                alpha=0.4)
    ax2.plot(DE_Single_channel['Wavelength'], Spline_lt1, color=Muf_colour, label='MuFLE 1', linewidth=4, linestyle='--')
    ax2.plot(DE_Single_channel['Wavelength'], Spline_lt2, color=Muf_colour, label='MuFLE 2', linewidth=4, linestyle=':')
    ax2.fill_between(DE_Single_channel['Wavelength'], (Spline_lt1 - Lt_CI1), (Spline_lt1 + Lt_CI1),
                     color=Muf_colour, alpha=.1)
    ax2.fill_between(DE_Single_channel['Wavelength'], (Spline_lt2 - Lt_CI2), (Spline_lt2 + Lt_CI2),
                     color=Muf_colour, alpha=.1)
    ax2.set_xlabel('Wavelength (nm)', fontsize=18)
    ax2.set_ylabel('Fluorescence Lifetime (ns)', fontsize=18)
    # plt.legend(fontsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    ax2.set_ylim([0, 4.5])

    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    # plt.savefig('RhBWaterFluo_DoubLt.png', format='png', dpi=1200)
    plt.show()
    plt.close()