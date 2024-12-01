# single pixel or individual channel functions
from lmfit import Parameters, Minimizer, report_fit
from scipy import signal
import numpy as np

# Single Exponential
# Initial Guess to be stored in LMfit Parameter()
def SE_initial_guess(Int_initial_guess, Lt_initial_guess, Bias_initial_guess):
    """ Function that takes the initial individual channel starting parameters and stores them
    in the parameters function to be given to LMfit
    :param Int_initial_guess: Intensity initial guess
    :param Lt_initial_guess: Lifetime initial guess
    :param Bias_initial_guess: Bias initial guess
    :return: The Parameters in the lmfit function/as a dictionary to be used in the minimiser
    """
    SE_params = Parameters()
    SE_params.add('Int', value=Int_initial_guess)
    SE_params.add('Lifetime', value=Lt_initial_guess)
    SE_params.add('Bias', value=Bias_initial_guess)

    return SE_params

# residuals
def SE_single_pixel_residuals(params, Time, data, IRF):
    """ Function that calculates the residuals between the observed and fit decays in the individual channels
    :param params: Initial guess parameters from the SE_initial_guess function
    :param Time: Time axis
    :param data: individual channel decay
    :param IRF: IRF from that specific channel
    :return: the residuals from a convolved exponential decay using the parameters to start
    """
    Int = params['Int']
    Lifetime = params['Lifetime']
    Bias = params['Bias']

    Decay = Int * np.exp(-(Time/Lifetime))

    Conv = signal.convolve(Decay, IRF, mode='full', method='direct')

    model = Conv[:len(Decay)] + np.exp(Bias)

    return model - data

# Double Exponential
def DE_initial_guess(Int_1_initial_guess, Lt_1_initial_guess, Int_2_initial_guess, Lt_2_initial_guess, Bias_initial_guess):
    """ Function that takes the initial individual channel starting parameters and stores them
    in the parameters function to be given to LMfit
    :param Int_1_initial_guess: First exponential intensity initial guess
    :param Lt_1_initial_guess: First exponential lifetime initial guess
    :param Int_2_initial_guess: Second exponential intensity initial guess
    :param Lt_2_initial_guess: Second exponential lifetime initial guess
    :param Bias_initial_guess: Bias initial guess
    :return: The Parameters in the lmfit function/as a dictionary to be used in the minimiser
    """
    DE_Params = Parameters()

    DE_Params.add('Int1', value=Int_1_initial_guess)
    DE_Params.add('Lifetime1', value=Lt_1_initial_guess)
    DE_Params.add('Int2', value=Int_2_initial_guess)
    DE_Params.add('Lifetime2', value=Lt_2_initial_guess)
    DE_Params.add('Bias', value=Bias_initial_guess)

    return DE_Params

def DE_single_pixel_residuals(params, Time, data, IRF):
    """ Function that calculates the residuals between the observed and fit decays of the individual
    channels using a double exponential
    :param params: Takes in the DE_initial_guess parameters
    :param Time: Time axis of one channel
    :param data: Decay from on channel
    :param IRF: IRF from that specific channel
    :return: The residuals between a convolved double exponential decay and the observed decay
    """
    Int1 = params['Int1']
    Int2 = params['Int2']
    Lifetime1 = params['Lifetime1']
    Lifetime2 = params['Lifetime2']
    Bias = params['Bias']

    # check if brackets helps this?
    Decay = Int1 * np.exp(-(Time/Lifetime1)) + Int2 * np.exp(-(Time/Lifetime2)) + np.exp(Bias)
    Conv = signal.convolve(Decay, IRF, mode='full', method='direct')

    model = Conv[:len(Decay)]

    return model - data

# Triple Exponential
def TE_initial_guess(Int_1_initial_guess, Lt_1_initial_guess, Int_2_initial_guess, Lt_2_initial_guess, Int_3_initial_guess, Lt_3_initial_guess, Bias_initial_guess):
    """ Function that takes the initial individual channel starting parameters and stores them
    in the parameters function to be given to LMfit
    :param Int_1_initial_guess: First intensity initial guess
    :param Lt_1_initial_guess: First lifetime initial guess
    :param Int_2_initial_guess: Second intensity initial guess
    :param Lt_2_initial_guess: Second lifetime initial guess
    :param Int_3_initial_guess: Third intensity initial guess
    :param Lt_3_initial_guess: Third lifetime initial guess
    :param Bias_initial_guess: Bias initial guess
    :return: The Parameters in the lmfit function/as a dictionary to be used in the minimiser
    """
    TE_Params = Parameters()

    TE_Params.add('Int1', value=Int_1_initial_guess)
    TE_Params.add('Lifetime1', value=Lt_1_initial_guess)
    TE_Params.add('Int2', value=Int_2_initial_guess)
    TE_Params.add('Lifetime2', value=Lt_2_initial_guess)
    TE_Params.add('Int3', value=Int_3_initial_guess)
    TE_Params.add('Lifetime3', value=Lt_3_initial_guess)
    TE_Params.add('Bias', value=Bias_initial_guess)

    return TE_Params

def TE_single_pixel_residuals(params, Time, data, IRF):
    """ Function to calculate the residual between the observed and fit decays of the individual
    channels using a triple exponential decay
    :param params: Takes in the TE_initial_guess parameters
    :param Time: Time axis for one channel
    :param data: Decay for one channel
    :param IRF: IRF from that channel
    :return: The residuals between a convolved triple exponential decay and the observed decay
    """
    Int1 = params['Int1']
    Int2 = params['Int2']
    Int3 = params['Int3']

    Lifetime1 = params['Lifetime1']
    Lifetime2 = params['Lifetime2']
    Lifetime3 = params['Lifetime3']

    Bias = params['Bias']

    # Again, perhaps something as trivial as adding brackets might help this?
    Decay = Int1 * np.exp(-(Time/Lifetime1)) + Int2 * np.exp(-(Time/Lifetime2)) + Int3 * np.exp(-(Time/Lifetime3))
    Conv = signal.convolve(Decay, IRF, mode='full', method='direct')

    model = Conv[:len(Decay)] + np.exp(Bias)

    return model - data

# Fitting function
def single_pixel_fitting_func(params, Residuals, Time, Data, IRF):
    """ Function that takes the starting parameters, residuals function, time data and IRF and returns
    the parameters that minimise the loss function using the LM gradient descent method
    :param params: Starting parameters
    :param Residuals: Function that calculates the residuals (single, double or triple exponential fit)
    :param Time: Time axis for one channel
    :param Data: Decay from the one channel
    :param IRF: IRF from that one channel
    :return: The fitted decay, the value of the residuals and the confidence interval
    """

    # fit with the default lv algorithm
    minner = Minimizer(Residuals, params, fcn_args=(Time, Data, IRF), nan_policy='propagate')
    result = minner.leastsq()

    # calculate final result
    LMFitDecay = Data + result.residual

    # write error report
    # report = report_fit(result)

    # Add results into data frame
    Title = []
    Value = []
    STD = []

    for name, param in result.params.items():
        Title.append(name)
        Value.append(param.value)
        STD.append(param.stderr)

    return LMFitDecay, Title, Value, STD

"""Tail fitting for multi-exponential assessment"""
# Initial Guess to be stored in LMfit Parameter()
def SE_tailfit_guess(Int_initial_guess, Lt_initial_guess, Bias_initial_guess):

    """ Function that takes the initial individual channel starting parameters and stores them
    in the parameters function to be given to LMfit
    :param Int_initial_guess: Intensity initial guess
    :param Lt_initial_guess: Lifetime initial guess
    :param Bias_initial_guess: Bias initial guess
    :return: The Parameters in the lmfit function/as a dictionary to be used in the minimiser
    """
    SE_params = Parameters()
    SE_params.add('Int', value=Int_initial_guess)
    SE_params.add('Lifetime', value=Lt_initial_guess)
    SE_params.add('Bias', value=Bias_initial_guess)
    return SE_params

# residuals
def SE_tailfit_residuals(params, Time, data):
    """ Function that calculates the residuals between the observed and fit decays in the individual channels
    :param params: Initial guess parameters from the SE_initial_guess function
    :param Time: Time axis
    :param data: individual channel decay
    :param IRF: IRF from that specific channel
    :return: the residuals from a convolved exponential decay using the parameters to start
    """
    Int = params['Int']
    Lifetime = params['Lifetime']
    Bias = params['Bias']

    model = Int * np.exp(-(Time/Lifetime)) + np.exp(Bias)

    return model - data

# Fitting function
def SE_tailfit_fitting_func(params, Residuals, Time, Data):
    """ Function that takes the starting parameters, residuals function, time data and IRF and returns
    the parameters that minimise the loss function using the LM gradient descent method
    :param params: Starting parameters
    :param Residuals: Function that calculates the residuals (single, double or triple exponential fit)
    :param Time: Time axis for one channel
    :param Data: Decay from the one channel
    :return: The fitted decay, the value of the residuals and the confidence interval
    """

    # fit with the default lv algorithm
    minner = Minimizer(Residuals, params, fcn_args=(Time, Data), nan_policy='propagate')
    result = minner.leastsq()

    # calculate final result
    LMFitDecay = Data + result.residual

    # write error report
    # report = report_fit(result)

    # Add results into data frame
    Title = []
    Value = []
    STD = []

    for name, param in result.params.items():
        Title.append(name)
        Value.append(param.value)
        STD.append(param.stderr)

    return LMFitDecay, Title, Value, STD