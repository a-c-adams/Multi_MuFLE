import numpy as np
from scipy.interpolate import BSpline, splrep
from scipy import signal
from scipy.optimize import least_squares

# the if a cubic splines/order of the splines is constant then
# the basis functions are the same, just the coefficients deviate, therefore, the replicates are unneeded
def Basisfunc_Knots_IdealCoef(Wave_axis, Decay_feature, knot_num, degree):
    ''' Function that gives the B-splones basis function, coefficients and knot vector for a 2D array
    (i.e. either the interpolation of the spectral intensity or spectral lifetime)
    :param Wave_axis: X axis for the spline calculation
    :param Decay_feature: Either intensity or lifetime for the initial coefficients to be estimated through
    :param knot_num: The number of knots needed for the splines
    :param degree: Splines degree
    :return: The splines basis function, coefficients and knot vector
    '''

    internal_knots = np.arange(1, knot_num, 1) / knot_num

    # t = the internal knots needed for the task
    # k = the degree of the spline fit -- not the order

    # splrep finds the B-Spline representation of a 1-D curve y = f(X)
    tck = splrep(Wave_axis, Decay_feature, t=internal_knots, k=degree)

    Coefficients = tck[1]
    knots = tck[0]

    # the wave_axis - x is from 0.01 to 0.99, to get the correct knots these need to change
    # back to be set to 0 and 1
    knots[knots == 0.01] = 0
    knots[knots == 0.99] = 1

    # calculation of the basis function
    Bfunc = []

    Start = np.min(np.nonzero(knots)) + 1
    Stop = len(knots) + 1

    for i in np.arange(Start, Stop):
        # print(knots[i - Start:i])
        #  basis_element returns a basis element for the knot vector
        b = BSpline.basis_element(knots[i - Start:i])
        mask = np.zeros(Wave_axis.shape)
        mask[(Wave_axis >= knots[i - Start]) & (Wave_axis <= knots[i - 1])] = 1
        # when WaveCol has a 1 at the end, the last basis function is 0 where it should be 1
        basis = b(Wave_axis) * mask
        Bfunc.append(basis)

    return Coefficients, knots, Bfunc

# goal is to write a function that takes l as a parameter and calculates MuFLE accrodingly
# first lets do it for fixed lifetimes as will be easier:
def Spline_residuals(Params, Data, Time, IRF, Cubic_func, l):
    """ Function that calculates the residuals of a single exponential MuFLE model, using splines to fit both
    the spectral intensity and spectral lifetime, and a single exponential decay to fit the decays at each channel
    :param Params: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :param l: the number of lifetime's being estimated
    :return: The residuals between the observed data and fitted data
    """

    p = len(Data[:, 0])
    q = len(Cubic_func)
    m = len(Time.T)

    # this is written as if the bias term is always the last coefficient:
    # this is also written like tau is fixed:
    gamma_all = Params[: (q * l)]
    tau_all = Params[(q * l): (q * l) + l]
    b_p = Params[-p:]

    # ok now we reshape gammas into p, m, l
    gamma = np.reshape(gamma_all, (l, len(Cubic_func)))
    gamma_Coef = [g @ Cubic_func for g in gamma]
    gamma_Int = np.reshape(np.tile(gamma_Coef, m), (l, p, m), order='F')

    Model = [gamma_Int[l_fin, :, :] * np.exp(-Time / tau_all[l_fin]) for l_fin in range(l)]
    Model = np.sum(Model, axis=0)

    # Conv = []
    #
    # for index, pixel in enumerate(Model):
    #     ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
    #     Conv.append(ModelConv[:len(Time.T)])
    #
    # Conv = np.array(Conv)

    Conv = signal.fftconvolve(Model, IRF, mode='full', axes=1)[:, :m]

    Bias = np.exp(b_p)[:, np.newaxis]

    Conv = Conv + Bias

    return np.ravel(Data - Conv)

def LS_fittingfunc(InitialGuess, Data, Time, IRF, Cubic_func, l):
    """ MuFlE single exponential fitting function
    :param InitialGuess: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: The estimated intensity, lifetime and bias values + the results message
    """

    # least_squares function uses lm gradient descent to find the local minima
    # # Bounds for the subset of coefficients (between 0 and 1)
    # lower_bounds = np.ones(len(InitialGuess)) * -np.inf
    # upper_bounds = np.ones(len(InitialGuess)) * np.inf

    # Specify bounds only for the subset of coefficients (e.g., coefficients 10 to 15)
    # lower_bounds[:18] = 0  # Lower bound for coefficients 10 to 15
    # upper_bounds[:18] = 1  # Upper bound for coefficients 10 to 15

    # Combine lower and upper bounds into a tuple
    # bounds_test = (lower_bounds, upper_bounds)

    # results = least_squares(fun=Spline_residuals, x0=InitialGuess, bounds=bounds_test, args=(Data, Time, IRF, Cubic_func, l),
    #                          ftol=1e-6, method='trf', loss='linear', max_nfev=100000)

    results = least_squares(fun=Spline_residuals, x0=InitialGuess, args=(Data, Time, IRF, Cubic_func, l),
                            ftol=1e-6, method='trf', loss='linear', max_nfev=100000)


    p = len(Data[:, 0])
    q = len(Cubic_func)

    # this is written as if the bias term is always the last coefficient:
    # this is also written like tau is fixed:
    gamma_Coef = results.x[: (q * l)]
    tau_Coef = results.x[(q * l): (q * l) + l]
    bias = results.x[-p:]

    return gamma_Coef, tau_Coef, bias, results

# def JacobFunc(Coef_matrix, Data, Time, IRF, BfuncInt, l):
#
#     # first we need to start this like the other functions:
#
#     p = len(Data[:, 0])
#     q = len(BfuncInt)
#     m = len(Time.T)
#
#     gamma_all = Coef_matrix[: (q * l)]
#     tau_all = Coef_matrix[(q * l): (q * l) + l]
#     b_p = Coef_matrix[-p:]
#
#     # ok now we reshape gammas into p, m, l
#     gamma = np.reshape(gamma_all, (l, len(BfuncInt)))
#     gamma_Coef = [g @ BfuncInt for g in gamma]
#     gamma_Int = np.reshape(np.tile(gamma_Coef, m), (l, p, m), order='F')
#
#
#     Model = [gamma_Int[l_fin, :, :] * np.exp(-Time / tau_all[l_fin]) for l_fin in range(l)]
#     Model = np.sum(Model, axis=0)
#
#     Conv = []
#
#     for index, pixel in enumerate(Model):
#         ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
#         Conv.append(ModelConv[:len(Time.T)])
#
#     Conv = np.array(Conv)
#
#     Bias = np.exp(b_p)[:, np.newaxis]
#
#     FitHist = Conv + Bias
#
#     O_p_m = Data - FitHist
#
#     # the setup equations:"
#     # keeping it as loops to start and make it make sense -- then will
#     # make more computationally efficient
#     V_mod = [np.exp(-Time / tau_all[l_fin]) for l_fin in range(l)]
#
#     V_pml = []
#     for mod in V_mod:
#
#         inner_conv = []
#
#         for index, pixel in enumerate(mod):
#
#             V_mod_conv = signal.convolve(mod[index], IRF[index], mode='full', method='direct')
#             inner_conv.append(V_mod_conv[:len(Time.T)])
#
#         V_pml.append(inner_conv)
#
#     V_pml = np.array(V_pml)
#
#
#     C_mod = [Time * np.exp(-Time / tau_all[l_fin]) for l_fin in range(l)]
#
#     C_pml = []
#
#     for mod in C_mod:
#
#         inner_conv = []
#
#         for index, pixel in enumerate(mod):
#
#             C_mod_conv = signal.convolve(mod[index], IRF[index], mode='full', method='direct')
#             inner_conv.append(C_mod_conv[:len(Time.T)])
#
#         C_pml.append(inner_conv)
#
#     C_pml = np.array(C_pml)
#
#     #### The equations
#     # First derivative of gamma:
#
#     Jac_gamma = []
#
#     for func in BfuncInt:
#
#         gamma_B_fuc = np.reshape(np.tile(func, len(Time.T)), np.shape(Time), order='F')
#
#         Jac_g_mod = []
#
#         for mod in C_pml:
#
#             Jac_g_mod.append(-2 * O_p_m * mod * gamma_B_fuc)
#
#         Jac_gamma.append(Jac_g_mod)
#
#     Jac_gamma = np.reshape(Jac_gamma, ((l * q), p, m), order='F')
#
#     # Jac_tau should be all 0's?
#
#     Jac_tau = []
#
#     for func in BfuncLt:
#
#         func_matrix = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')
#
#         Numer = FitModel * (m * func_matrix)
#         Denom = Lifetime ** 2
#
#         Frac = Numer/Denom
#
#         conv = []
#
#         for index, pixel in enumerate(m):
#             sConv = signal.convolve(Frac[index], IRF[index], mode='full', method='direct')
#             conv.append(sConv[:len(m.T)])
#
#         conv = np.array(conv)
#
#         Jac_tau.append(- np.sum(O_p_m * conv))
#
#     # bias gradient:
#    Jac_bias = -2 * b_p[:, np.newaxis] * O_p_m
#
#     Jacobian = Jac_gamma + Jac_tau + Jac_bias
#
#     return Jacobian

# older functions:

def DE_Spline_residuals(Params, Data, Time, IRF, BfuncInt1, BfuncLt1, BfuncInt2, BfuncLt2):
    """ Function that calculates the residuals of a double exponential MuFLE model, using splines to fit both
    the double spectral intensity and double spectral lifetime, and a double exponential decay to fit the decays at each channel
    :param Params: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt1: First intensity basis function
    :param BfuncLt1: First lifetime basis function
    :param BfuncInt2: Second intensity basis function
    :param BfuncLt2: Second lifetime basis function
    :return: The residuals between the observed data and fitted data
    """

    IntCoef1 = Params[:len(BfuncInt1)]
    LtCoef1 = Params[len(BfuncInt1): (len(BfuncInt1) + len(BfuncLt1))]
    IntCoef2 = Params[(len(BfuncInt1) + len(BfuncLt1)):(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2))]
    LtCoef2 = Params[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2)): (len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2))]
    Bias = Params[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2)):]

    Int1 = IntCoef1 @ BfuncInt1
    Int1 = np.reshape(np.tile(Int1, len(Data.T)), np.shape(Data), order='F')

    Lifetime1 = LtCoef1 @ BfuncLt1
    Lifetime1 = np.reshape(np.tile(Lifetime1, len(Data.T)), np.shape(Data), order='F')

    Int2 = IntCoef2 @ BfuncInt2
    Int2 = np.reshape(np.tile(Int2, len(Data.T)), np.shape(Data), order='F')

    Lifetime2 = LtCoef2 @ BfuncLt2
    Lifetime2 = np.reshape(np.tile(Lifetime2, len(Data.T)), np.shape(Data), order='F')

    Model = Int1 * np.exp(-Time/Lifetime1) + Int2 * np.exp(-Time/Lifetime2)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Conv = np.array(Conv)

    Bias = Bias[:, np.newaxis]

    Conv = Conv + np.exp(Bias)

    return np.ravel(Data - Conv)

def DE_LS_fittingfunc(InitialGuess, Data, Time, IRF, BfuncInt1, BfuncLt1, BfuncInt2, BfuncLt2):
    """ MuFLE double exponential fitting function
    :param InitialGuess: Initial guess
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt1: First intensity basis function
    :param BfuncLt1: First lifetime basis function
    :param BfuncInt2: Second intensity basis function
    :param BfuncLt2: Second lifetime basis function
    :return: The estimated intensity, lifetime and bias values + the results message
    """
    # least_squares function uses lm gradient descent to find the local minima
    results = least_squares(fun=DE_Spline_residuals, x0=InitialGuess, args=(Data, Time, IRF, BfuncInt1, BfuncLt1, BfuncInt2, BfuncLt2), ftol=1e-12, method='lm')

    IntCoef1 = results.x[:len(BfuncInt1)]
    LtCoef1 = results.x[len(BfuncInt1): (len(BfuncInt1) + len(BfuncLt1))]
    IntCoef2 = results.x[(len(BfuncInt1) + len(BfuncLt1)):(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2))]
    LtCoef2 = results.x[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2)): (len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2))]
    Bias = results.x[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2)):]

    return IntCoef1, LtCoef1, IntCoef2, LtCoef2, Bias, results

def JacobFunc(Coef_matrix, Data, Time, IRF, BfuncInt, BfuncLt):

    gammaCoef = Coef_matrix[:len(BfuncInt)]
    tauCoef = Coef_matrix[len(BfuncInt):len(BfuncInt) + len(BfuncLt)]
    Bias = Coef_matrix[len(BfuncInt) + len(BfuncLt):]

    Int = np.array(gammaCoef) @ BfuncInt
    Lt = np.array(tauCoef) @ BfuncLt

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')

    Int1 = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    FitModel = Int1 * np.exp(-Time / Lifetime)

    FitConv = []

    for index, pixel in enumerate(FitModel):
        ModelConv = signal.convolve(FitModel[index], IRF[index], mode='full', method='direct')
        FitConv.append(ModelConv[:len(Time.T)])

    Bias1 = np.exp(Bias)[:, np.newaxis]

    FitHist = FitConv + Bias1

    m = Time
    O = Data - FitHist

    Jac_gamma = []

    for func in BfuncInt:

        gamma_B_fuc = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        gamma_exp = gamma_B_fuc * np.exp(-m / Lifetime)

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(gamma_exp[index], (IRF)[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_gamma.append(- np.sum(O * conv))

    Jac_tau = []

    for func in BfuncLt:

        func_matrix = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        Numer = FitModel * (m * func_matrix)
        Denom = Lifetime ** 2

        Frac = Numer/Denom

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(Frac[index], IRF[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_tau.append(- np.sum(O * conv))

    Jac_bias = []

    for b, resid in zip(Bias, O):
        Jac_bias.append((- np.exp(b)) * (np.sum(resid)))

    Jacobian = Jac_gamma + Jac_tau + Jac_bias

    return Jacobian

def Gradient(Coef_matrix, Data, Time, IRF, BfuncInt, BfuncLt):
    """ Function that calculates the derivatives between the coefficients and the observed data
    :param Coef_matrix: Coefficient results from minimising the loss functions above (single MuFLE exponential)
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: Gradient
    """

    gammaCoef = Coef_matrix[:len(BfuncInt)]
    tauCoef = Coef_matrix[len(BfuncInt):len(BfuncInt) + len(BfuncLt)]
    Bias = Coef_matrix[len(BfuncInt) + len(BfuncLt):]

    Int = np.array(gammaCoef) @ BfuncInt
    Lt = np.array(tauCoef) @ BfuncLt

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')

    Int1 = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    FitModel = Int1 * np.exp(-Time / Lifetime)

    FitConv = []

    for index, pixel in enumerate(FitModel):
        ModelConv = signal.convolve(FitModel[index], IRF[index], mode='full', method='direct')
        FitConv.append(ModelConv[:len(Time.T)])

    Bias1 = np.exp(Bias)[:, np.newaxis]

    FitHist = FitConv + Bias1

    m = Time
    O = Data - FitHist

    Jac_gamma = []

    for func in BfuncInt:

        gamma_B_fuc = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        gamma_exp = gamma_B_fuc * np.exp(-m / Lifetime)

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(gamma_exp[index], (IRF)[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_gamma.append(-(O * conv))

    Jac_tau = []

    for func in BfuncLt:

        func_matrix = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        Numer = FitModel * (m * func_matrix)
        Denom = Lifetime ** 2

        Frac = Numer / Denom

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(Frac[index], IRF[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_tau.append(-(O * conv))

    Jac_bias = []

    for b, resid in zip(Bias, O):
        Jac_bias.append((- np.exp(b)) * (resid))

    Grad = Jac_gamma + Jac_tau + Jac_bias

    return Grad

def FitHistFunc(tauCoef, Bfunctau, gammaCoef, Bfuncgamma, Time, IRF, Bias):
    """ Function that calculates the fitted histogram from the optimal coefficients
    :param tauCoef: Lifetime/tau coefficient
    :param Bfunctau: lifetime basis function
    :param gammaCoef: Intensity/gamma coefficient
    :param Bfuncgamma: intensity basis function
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param Bias: Bias value
    :return: Fitted histogram
    """

    Int = np.array(gammaCoef) @ Bfuncgamma
    Lt = np.array(tauCoef) @ Bfunctau

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')
    Int = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    Model = Int * np.exp(-Time / Lifetime)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Bias = np.exp(Bias)[:, np.newaxis]

    return np.array(Conv + Bias)

def DoubFitHistFunc(tauCoef1, tauCoef2, Bfunctau1, Bfunctau2, gammaCoef1, Bfuncgamma1, gammaCoef2, Bfuncgamma2, Time, IRF, Bias):
    """ Function that calculates the fitted histogram from the optimal double MuFLE coefficients
    :param tauCoef1: Lifetime 1 coefficients
    :param tauCoef2: Lifetime 2 coefficients
    :param Bfunctau1: Lifetime 1 basis function
    :param Bfunctau2: Lifetime 2 basis function
    :param gammaCoef1: Intensity 1
    :param Bfuncgamma1: Intensity 1 basis function
    :param gammaCoef2: Intensity 2
    :param Bfuncgamma2: Intensity 2 basis function
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param Bias: Bias value
    :return: Fitted Histogram
    """

    Int1 = np.array(gammaCoef1) @ Bfuncgamma1
    Lt1 = np.array(tauCoef1) @ Bfunctau1

    Int2 = np.array(gammaCoef2) @ Bfuncgamma2
    Lt2 = np.array(tauCoef2) @ Bfunctau2

    Lifetime1 = np.reshape(np.tile(Lt1, len(Time.T)), np.shape(Time), order='F')
    Int1 = np.reshape(np.tile(Int1, len(Time.T)), np.shape(Time), order='F')

    Lifetime2 = np.reshape(np.tile(Lt2, len(Time.T)), np.shape(Time), order='F')
    Int2 = np.reshape(np.tile(Int2, len(Time.T)), np.shape(Time), order='F')

    Model = Int1 * np.exp(-Time / Lifetime1) + Int2 * np.exp(-Time / Lifetime2)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Bias = np.exp(Bias)[:, np.newaxis]

    return np.array(Conv + Bias)

# single exponential with a fixed lifetime
def SE_FL_Spline_residuals(Params, Data, Time, IRF, BfuncInt1, Lt1):
    """ Function that calculates the residuals of between the observed and fitted data in a double exponential with a fixed lifetime
    :param Params: Starting parameters
    :param Data: Observed data
    :param Time: Time histogram
    :param IRF: IRF histogram
    :param BfuncInt1: First exponential B splines intensity function
    :param Lt1: First exponential lifetime
    :return: The residuals
    """

    IntCoef1 = Params[:len(BfuncInt1)]
    Lt_start_1 = Params[len(BfuncInt1): (len(BfuncInt1) + len(Lt1))]

    Bias = Params[(len(BfuncInt1) + len(Lt1)):]

    Int1 = IntCoef1 @ BfuncInt1
    Int1 = np.reshape(np.tile(Int1, len(Data.T)), np.shape(Data), order='F')

    Lifetime1 = Lt_start_1

    Model = Int1 * np.exp(-Time/Lifetime1)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Conv = np.array(Conv)

    Bias = Bias[:, np.newaxis]

    Conv = Conv + np.exp(Bias)

    return np.ravel(Data - Conv)

def SE_FL_fittingfunc(InitialGuess, Data, Time, IRF, BfuncInt1, Lt1):
    '''
    :param InitialGuess: Intensity guess
    :param Data: Obserevd histogram
    :param Time: Time histogram
    :param IRF: IRF histogram
    :param BfuncInt1: First exponential B splines intensity function
    :param Lt1: First exponential lifetime
    :returns The parameters that minimise the cost function
    '''

    results = least_squares(fun=SE_FL_Spline_residuals, x0=InitialGuess, args=(Data, Time, IRF, BfuncInt1, Lt1), ftol=1e-12, method='lm')

    IntCoef1 = results.x[:len(BfuncInt1)]
    LtCoef1 = results.x[len(BfuncInt1): (len(BfuncInt1) + len(Lt1))]

    Bias = results.x[(len(BfuncInt1) + len(Lt1)):]

    return IntCoef1, LtCoef1, Bias, results

# double exponential with a fixed lifetime
def DE_FL_Spline_residuals(Params, Data, Time, IRF, BfuncInt1, Lt1, BfuncInt2, Lt2):
    """ Function that calculates the residuals of between the observed and fitted data in a double exponential with a fixed lifetime
    :param Params: Starting parameters
    :param Data: Observed data
    :param Time: Time histogram
    :param IRF: IRF histogram
    :param BfuncInt1: First exponential B splines intensity function
    :param Lt1: First exponential lifetime
    :param BfuncInt2: Second exponential B splines intensity function
    :param Lt2: Second exponential lifetime
    :return: The residuals
    """

    IntCoef1 = Params[:len(BfuncInt1)]
    Lt_start_1 = Params[len(BfuncInt1): (len(BfuncInt1) + len(Lt1))]

    IntCoef2 = Params[(len(BfuncInt1) + len(Lt1)):(len(BfuncInt1) + len(Lt1) + len(BfuncInt2))]
    Lt_start_2 = Params[(len(BfuncInt1) + len(Lt1) + len(BfuncInt2)): (len(BfuncInt1) + len(Lt1) + len(BfuncInt2) + len(Lt2))]

    Bias = Params[(len(BfuncInt1) + len(Lt1) + len(BfuncInt2) + len(Lt2)):]

    Int1 = IntCoef1 @ BfuncInt1
    Int1 = np.reshape(np.tile(Int1, len(Data.T)), np.shape(Data), order='F')

    Lifetime1 = Lt_start_1

    Int2 = IntCoef2 @ BfuncInt2
    Int2 = np.reshape(np.tile(Int2, len(Data.T)), np.shape(Data), order='F')

    Lifetime2 = Lt_start_2

    Model = Int1 * np.exp(-Time/Lifetime1) + Int2 * np.exp(-Time/Lifetime2)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Conv = np.array(Conv)

    Bias = Bias[:, np.newaxis]

    Conv = Conv + np.exp(Bias)

    return np.ravel(Data - Conv)

def DE_FL_fittingfunc(InitialGuess, Data, Time, IRF, BfuncInt1, Lt1, BfuncInt2, Lt2):
    '''
    :param InitialGuess: Intensity guess
    :param Data: Obserevd histogram
    :param Time: Time histogram
    :param IRF: IRF histogram
    :param BfuncInt1: First exponential B splines intensity function
    :param Lt1: First exponential lifetime
    :param BfuncInt2: Second exponential B splines intensity function
    :param Lt2: Second exponential lifetime
    :returns The parameters that minimise the cost function
    '''

    results = least_squares(fun=DE_FL_Spline_residuals, x0=InitialGuess, args=(Data, Time, IRF, BfuncInt1, Lt1, BfuncInt2, Lt2), ftol=1e-12, method='lm')

    IntCoef1 = results.x[:len(BfuncInt1)]
    LtCoef1 = results.x[len(BfuncInt1): (len(BfuncInt1) + len(Lt1))]

    IntCoef2 = results.x[(len(BfuncInt1) + len(Lt1)):(len(BfuncInt1) + len(Lt1) + len(BfuncInt2))]
    LtCoef2 = results.x[(len(BfuncInt1) + len(Lt1) + len(BfuncInt2)): (len(BfuncInt1) + len(Lt1) + len(BfuncInt2) + len(Lt2))]

    Bias = results.x[(len(BfuncInt1) + len(Lt1) + len(BfuncInt2) + len(Lt2)):]

    return IntCoef1, LtCoef1, IntCoef2, LtCoef2, Bias, results

# Tail fitting MuFLE
def Spline_tailfit_residuals(Params, Data, Time, BfuncInt, l):
    """ Function that calculates the residuals of a single exponential MuFLE model, using splines to fit both
    the spectral intensity and spectral lifetime, and a single exponential decay to fit the decays at each channel
    :param Params: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :param l: the number of lifetime's being estimated
    :return: The residuals between the observed data and fitted data
    """

    p = len(Data[:, 0])
    q = len(BfuncInt)
    m = len(Time.T)

    # this is written as if the bias term is always the last coefficient:
    # this is also written like tau is fixed:
    gamma_all = Params[: (q * l)]
    tau_all = Params[(q * l) : (q * l) + l]
    b_p = Params[-p:]

    # ok now we reshape gammas into p, m, l
    gamma = np.reshape(gamma_all, (l, len(BfuncInt)))
    gamma_Coef = [g @ BfuncInt for g in gamma]
    gamma_Int = np.reshape(np.tile(gamma_Coef, m), (l, p, m), order='F')

    Model = [gamma_Int[l_fin, :, :] * np.exp(-Time / tau_all[l_fin]) for l_fin in range(l)]
    Model = np.sum(Model, axis=0)

    Bias = np.exp(b_p)[:, np.newaxis]

    Conv = Model + Bias

    return np.ravel(Data - Conv)

def LS_tailfit_fittingfunc(InitialGuess, Data, Time, BfuncInt, l):
    """ MuFlE single exponential fitting function
    :param InitialGuess: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: The estimated intensity, lifetime and bias values + the results message
    """

    # least_squares function uses lm gradient descent to find the local minima
    results = least_squares(fun=Spline_tailfit_residuals, x0=InitialGuess, args=(Data, Time, BfuncInt, l), ftol=1e-15, method='lm', loss='linear', max_nfev=100000)

    p = len(Data[:, 0])
    q = len(BfuncInt)

    # this is written as if the bias term is always the last coefficient:
    # this is also written like tau is fixed:
    gamma_Coef = results.x[: (q * l)]
    tau_Coef = results.x[(q * l): (q * l) + l]
    bias = results.x[-p:]

    return gamma_Coef, tau_Coef, bias, results

# def DE_tailfit_Spline_residuals(Params, Data, Time, BfuncInt1, BfuncLt1, BfuncInt2, BfuncLt2):
#     """ Function that calculates the residuals of a double exponential MuFLE model, using splines to fit both
#     the double spectral intensity and double spectral lifetime, and a double exponential decay to fit the decays at each channel
#     :param Params: Starting parameters
#     :param Data: Observed histogram
#     :param Time: Time histogram
#     :param BfuncInt1: First intensity basis function
#     :param BfuncLt1: First lifetime basis function
#     :param BfuncInt2: Second intensity basis function
#     :param BfuncLt2: Second lifetime basis function
#     :return: The residuals between the observed data and fitted data
#     """
#
#     IntCoef1 = Params[:len(BfuncInt1)]
#     LtCoef1 = Params[len(BfuncInt1): (len(BfuncInt1) + len(BfuncLt1))]
#     IntCoef2 = Params[(len(BfuncInt1) + len(BfuncLt1)):(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2))]
#     LtCoef2 = Params[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2)): (len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2))]
#     Bias = Params[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2)):]
#
#     Int1 = IntCoef1 @ BfuncInt1
#     Int1 = np.reshape(np.tile(Int1, len(Data.T)), np.shape(Data), order='F')
#
#     Lifetime1 = LtCoef1 @ BfuncLt1
#     Lifetime1 = np.reshape(np.tile(Lifetime1, len(Data.T)), np.shape(Data), order='F')
#
#     Int2 = IntCoef2 @ BfuncInt2
#     Int2 = np.reshape(np.tile(Int2, len(Data.T)), np.shape(Data), order='F')
#
#     Lifetime2 = LtCoef2 @ BfuncLt2
#     Lifetime2 = np.reshape(np.tile(Lifetime2, len(Data.T)), np.shape(Data), order='F')
#
#     Model = Int1 * np.exp(-Time/Lifetime1) + Int2 * np.exp(-Time/Lifetime2)
#
#     Bias = Bias[:, np.newaxis]
#
#     Conv = Model + np.exp(Bias)
#
#     return np.ravel(Data - Conv)
#
# def DE__tailfit_fittingfunc(InitialGuess, Data, Time, BfuncInt1, BfuncLt1, BfuncInt2, BfuncLt2):
#     """ MuFLE double exponential fitting function
#     :param InitialGuess: Initial guess
#     :param Data: Observed histogram
#     :param Time: Time histogram
#     :param BfuncInt1: First intensity basis function
#     :param BfuncLt1: First lifetime basis function
#     :param BfuncInt2: Second intensity basis function
#     :param BfuncLt2: Second lifetime basis function
#     :return: The estimated intensity, lifetime and bias values + the results message
#     """
#     # least_squares function uses lm gradient descent to find the local minima
#     results = least_squares(fun=DE_Spline_residuals, x0=InitialGuess, args=(Data, Time, BfuncInt1, BfuncLt1, BfuncInt2, BfuncLt2), ftol=1e-12, method='lm')
#
#     IntCoef1 = results.x[:len(BfuncInt1)]
#     LtCoef1 = results.x[len(BfuncInt1): (len(BfuncInt1) + len(BfuncLt1))]
#     IntCoef2 = results.x[(len(BfuncInt1) + len(BfuncLt1)):(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2))]
#     LtCoef2 = results.x[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2)): (len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2))]
#     Bias = results.x[(len(BfuncInt1) + len(BfuncLt1) + len(BfuncInt2) + len(BfuncLt2)):]
#
#     return IntCoef1, LtCoef1, IntCoef2, LtCoef2, Bias, results