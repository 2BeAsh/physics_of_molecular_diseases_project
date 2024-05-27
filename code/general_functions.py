import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import csv
import sympy as sp
from IPython.display import display
from iminuit import Minuit
from scipy.optimize import curve_fit, minimize
import scipy.stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import nestle
from tqdm import tqdm

# Customizing matplotlib settings
default_cycler = (cycler(color=["rebeccapurple", "red", "darkorange", "seagreen", "deepskyblue", "black"])
                  + cycler(linestyle = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))])
                  + cycler(marker=[".", "x", "v", "1", "p", ">"]))
plt.rc('axes', prop_cycle=default_cycler)
plt.style.use("presentation.mplstyle")

fig_background_color = (224/255, 224/255, 224/255, 1)


# -- DATA IMPORTING -- 
def load_data_with_subdatasets(fname, delimiter, skip_header, dtype):
    """Load a datafile which contains multiple datasets. Does not work if last element in datafile is whitespace."""

    # Read datafile and get length of datafile
    datafile = open(fname, "r")
    datareader = csv.reader(datafile)
    datafile_for_length = open(fname, "r") 
    row_count = sum(1 for _ in csv.reader(datafile_for_length))
    
    # Skip header rows
    for _ in range(skip_header):
        next(datareader, None)        
    
    # Empty lists to append subdatasets to 
    data_sets_combined = []
    
    # Loop
    line_number = 0
    while row_count > line_number:
        sub_data_set = []
            
        for row in datareader:
            line_number = datareader.line_num     
            # Skip empty rows
            if len(row) == 0:
                continue
            
            # Need to seperate row values and convert them to numbers
            # See if can string convert to float. If cannot, then we have encountered header and thus end of subdataset.
            # At last line need to append data and break. OBS could alternatively add a string to last line.
            row = row[0]  # "Convert" list to string
            try:
                row_value = row.split(delimiter)
                row_float = np.array(row_value, dtype=dtype)
                sub_data_set.append(row_float)
                
                if line_number == row_count: # Last line 
                    print("Not more data")
                    data_sets_combined.append(sub_data_set)
                    break
                                    
            except:  # header cannot convert to float
                data_sets_combined.append(sub_data_set)
                break
            
    return data_sets_combined


# -- BAYSIAN STATISTICS --
def find_normalization_const(f_expr, var, lower_lim, upper_lim, display_result=False):
    """Calculate normalization constant of a function by finding the definite integral using Sympy. 

    Args:
        f_expr (sympy expression): Function whose normalization constant is needed, as a Sympy expression
        var (sympy symbol): Integration variable.
        lower_lim (float): Lower integration limit
        upper_lim (float): Upper integration limit
        display_result (bool, optional): Display function and definite integral. Defaults to False.

    Returns:
        float: Normalization constant of f_expr
    """
    def_integral = sp.integrate(f_expr, (var, lower_lim, upper_lim)).doit()
    norm_const = 1 / def_integral
    if display_result:
        print("Function")
        display(f_expr)
        print("Definite Integral")
        display(sp.simplify(def_integral))
        print("Norm Constant")
        display(sp.simplify(norm_const))
    return norm_const


def numerical_normalization_const(f_pdf, x, par=()):
    """Numerically normalize a function f_pdf evaluated at x with parameters par.

    Args:
        f_pdf (func): Function to be normalized
        x (1darray): Independent variable where f_pdf is evaluated 
        par (tupple of floats): Parameter values for f_pdf

    Returns:
        1darray of floats: Normalized values for f_pdf evaluated in x
    """
    f_vals = f_pdf(x, *par)
    integrated_f_pdf = np.trapz(f_vals, x)
    return 1 / integrated_f_pdf


def bayesian_posterior(f_likelihood, f_prior, f_marginal, x, par_prior, par_lh, par_marginal):
    return f_likelihood(x, *par_lh) * f_prior(x, *par_prior) / f_marginal(x, *par_marginal)


def visualiuze_bayesian_distributions(f_likelihood, f_prior, f_marginal, x, par_prior, par_lh, par_marginal, xlabel="", ylabel="", title=""):
    prior = f_prior(x, par_prior)
    lh = f_likelihood(x, par_lh)
    posterior = bayesian_posterior(f_likelihood, f_prior, f_marginal, x, par_prior, par_lh, par_marginal)
    
    fig, ax = plt.subplots()
    ax.plot(x, posterior, label="Posterior")
    ax.plot(x, prior, label="Prior")
    ax.plot(x, lh, label="Likelihood")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    plt.show()


def loglikelihood(par, x, pdf, sum_axis=0):
    """Calculate NEGATIVE log likelihood of the pdf evaluated in x given parameters par. Assumes par is a tupple.

    Args:
        par (tupple): Parameter values for the pdf
        x (1darray): Points where the pdf is evaluated
        pdf (func): Probability density function
        sum_axis (int, optional): Which axis to sum over. Defaults to 0.

    Returns:
        1darray: llh values for each pararameter. Size par. 
    """
    return -np.sum(np.log(pdf(x, *par)), axis=sum_axis)


def minimize_llh(f_pdf, x, p0: tuple, f_jac=None, optimizer_method="Nelder-Mead") -> tuple:
    """'Fit' a function by minimizing its natural log likelihood. 

    Args:
        f_pdf (function): _description_
        x (1darray): _description_
        p0 (tuple): _description_
        f_jac (function, optional): _description_. Defaults to None.
        optimizer_method (str, optional): _description_. Defaults to "Nelder-Mead".

    Returns:
        tuple: par, err. Error is None unless f_jac is given
    """
    res = minimize(loglikelihood, x0=p0, args=(x, f_pdf), jac=f_jac, method=optimizer_method)
    par = res.x
    err = None
    if f_jac is not None:
        try:
            err = np.sqrt(np.diag(res.hess_inv.todense()))
        except AttributeError:
            print(f"Warning: Parameter uncertainty cannot be found using optimizer method {optimizer_method}. Instead, use any of: \nNewton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr")
            err = [np.NaN]
    return par, err


def parameter_bootstrapping(f_pdf, fit_par, x_bound, bootstrap_steps, MC_steps):
    par_vals = np.empty((fit_par.size, bootstrap_steps))
    
    for i in range(bootstrap_steps):
        # Get samples from pdf
        x_pdf = monte_carlo_sample_from_pdf(f_pdf, fit_par, x_bound, MC_steps)
        # Fit the samples to get the fit par
        par, err = minimize_llh(f_pdf, x_pdf, p0=fit_par)
        par_vals[:, i] = par
        
    return par_vals


def llh_raster1d(f_pdf, x, par_vals, plot=False):
    """1d Raster LLH scan. 

    Args:
        f_pdf (func): The pdf for which the llh is calculated.
        x (1darray): x-data for the pdf
        par (1darray): Array of parameter values
        plot (bool, optional): Visualize the llh values and MLE. Defaults to False.

    Returns:
        (MLE_idx, MLE, par_MLE): The index of the MLE value, the MLE value, and the paramter evaluated at the MLE idx
    """
    # Calculate loglikelihood
    llh_vals = np.empty(np.size(par_vals))
    llh_vals = loglikelihood(par_vals[None, :], x[:, None], f_pdf)
    # Find minimum value and parameter at that value
    MLE_idx = np.argmin(llh_vals)
    MLE = llh_vals[MLE_idx]
    par_MLE = par_vals[MLE_idx]
    # Calculate parameter uncertainty
    # Find where LLH - MLE = 0.5
    delta_LLH_equal_half = find_nearest(np.abs(MLE - llh_vals), 0.5)
    # Find difference in par values from LLH - MLE = 0.5 and MLE par
    LLH_1sigma = llh_vals[delta_LLH_equal_half]
    par_1sigma = par_vals[delta_LLH_equal_half]
    sigma_par_MLE = np.abs(par_MLE - par_1sigma)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(par_vals, llh_vals, ".", label="LLH")
        ax.plot(par_MLE, MLE, "x", label="Max LLH")
        ax.axhline(LLH_1sigma, ls="dashed", color="grey", alpha=0.9, label=r"$\Delta LLH=0.5$")
        str_title = fr"$\sigma =$ {sigma_par_MLE:.2f}"
        ax.set(xlabel="Parameter value", ylabel="LLH", title=str_title)
        ax.legend()
        plt.show()
    
    return MLE_idx, MLE, par_MLE, sigma_par_MLE, llh_vals


def llh_2d_confidence_region(f_pdf, x, par1, par2, plot=False):
    # 2d raster    
    par1_mesh, par2_mesh = np.meshgrid(par1, par2)
    pp1 = par1_mesh[None, :, :]
    pp2 = par2_mesh[None, :, :]
    xx = x[:, None, None]

    llh_vals = loglikelihood((pp1, pp2), xx, f_pdf)
    
    # MLE
    MLE_idx = np.unravel_index(np.argmin(llh_vals, axis=None), llh_vals.shape)
    MLE = llh_vals[MLE_idx]
        
    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(llh_vals, extent=(par1.min(), par1.max(), par2.min(), par2.max()), origin="upper", vmin=MLE, vmax=1060)
        # Confidence region
        ax.contour(llh_vals, levels=[MLE+1.15, MLE+3.09, MLE+5.92], extent=(par1.min(), par1.max(), par2.min(), par2.max()), origin="upper")
        ax.set(xlabel="alpha", ylabel="beta")
        plt.colorbar(im)
        plt.show()


def LLH_parameter_uncertainty(f_pdf, x, par_vals, plot=False):
    """Use MLE - LLH = 0.5 to find uncertainty on parameter. f_pdf must only have one parameter.
    Returns:
        (float, float): Parameter at MLE, uncertainty of parameter at MLE
    """
    # 1d Raster scan to get MLE
    MLE_idx, MLE, par_MLE, llh_vals = llh_raster1d(f_pdf, x, par_vals)
    
    # Find where LLH - MLE = 0.5
    delta_LLH_equal_half = find_nearest(np.abs(MLE - llh_vals), 0.5)
    # Find difference in par values from LLH - MLE = 0.5 and MLE par
    LLH_1sigma = llh_vals[delta_LLH_equal_half]
    par_1sigma = par_vals[delta_LLH_equal_half]
    sigma_par = np.abs(par_MLE - par_1sigma)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(par_vals, llh_vals, ".", label="LLH")
        ax.axhline(LLH_1sigma, ls="dashed", color="grey", alpha=0.9, label=r"$\Delta LLH=0.5$")
        str_title = fr"$\sigma =$ {sigma_par:.2f}"
        ax.set(xlabel=r"$\beta$", ylabel="LLH", title=str_title)
        plt.show()
    
    return par_MLE, sigma_par


def confidence_interval(data, sigma_fraction=0.6827):
    """Confidence interval on mean of given unbinned data.
    Args:
        data (1darray): Unbinned data. 
        sigma_fraction (float): 1sigma CI is 0.6827, 2sigma CI is XXX, and 3sigma CI is YYY.
        
    Returns: (float, float, float, float, float, float)
        First 3 values are "x-values": lower_bound, middle_val (same as mean), upper_bound
        Last 3 values are "y-values": The corresponding cummulative probability at these "x-values". 
        They should theoretically be uqual to 0.158862, 0.5, 0.84135
    """
    sorted_data = np.sort(data)
    cumsum = np.arange(len(data)) / (float(len(data))-1)  # Percent of data. 50% of data is in the middle

    idx_lower = np.where(cumsum < (1 - sigma_fraction) / 2)[0][-1]
    idx_upper = np.where(cumsum < (1 + sigma_fraction) / 2)[0][-1]
    idx_middle = np.where(cumsum < 1 / 2)[0][-1]

    lower_bound = sorted_data[idx_lower]  # [0] to get rid of dtype, [-1] because want leftmost value
    upper_bound = sorted_data[idx_upper] 

    cumsum_lower_bound = cumsum[idx_lower] 
    cumsum_upper_bound = cumsum[idx_upper] 

    middle_val = sorted_data[idx_middle] 
    cumsum_middle_val = cumsum[idx_middle] 
    
    return lower_bound, middle_val, upper_bound, cumsum_lower_bound, cumsum_middle_val, cumsum_upper_bound 


# -- FITTING --
def fit_unbinned_data(fit_pdf, x, y, p0, plot=False):
    par, cov = curve_fit(fit_pdf, x, y, p0)
    err = np.sqrt(np.diag(cov))
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, fmt=".", label="data")        
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = fit_pdf(x_fit, *par)
        ax.plot(x_fit, y_fit, label="Fit")
        ax.legend()
        plt.show()
    
    return par, err
    
    
def llh_test_null_alt(x, f_pdf_null, f_pdf_alt, p0_null, p0_alt, display_result=False):
    """Find the probability that the null hypothesis agrees with the alternative hypothesis at 2(LLH h0 - llh hA). 
    A low p-value means the null hypothesis performs poorly compared to the alternative hypothesis. A high probability means they compare similarly.

    Args:
        x (1darray): Independent variable
        f_pdf_null (func): Null hypothesis probability density
        f_pdf_alt (func): Alternative hypothesis probability density
        p0_null (tupple): Initial fit guess for h0
        p0_alt (tupple): Initial fit guess for hA
        display_result (bool, optional): Print p-value and more, and plot. Defaults to False.

    Returns:
        float: p-value
    """
    # Fit
    par_null = minimize_llh(f_pdf_null, x, p0_null)
    par_alt = minimize_llh(f_pdf_alt, x, p0_alt)
    
    llh_null = loglikelihood(par_null, x, f_pdf_null)
    llh_alt = loglikelihood(par_alt, x, f_pdf_alt)
    
    # Test statistic
    llh_diff = 2 * (llh_null - llh_alt)  # llh already has minus from the loglikelihood function
    Ndof = len(par_alt) - len(par_null)
    p_value = scipy.stats.chi2.sf(llh_diff, Ndof)

    if display_result:
        fig, ax = plt.subplots()
        x_fit = np.linspace(x.min(), x.max(), 300)
        ax.plot(x_fit, f_pdf_null(x_fit, *par_null), label="h0")
        ax.plot(x_fit, f_pdf_alt(x_fit, *par_alt),  "-.", label="hA")
        ax.hist(x, bins=int(np.sqrt(len(x))), histtype="step", label="Data", density=True)
        ax.legend()
        plt.show()
        
        print(f"LLH Null: {llh_null:.4f}")
        for i, par in enumerate(par_null):
            print(f"\tPar {i} null: {par:.4f}")
        print(f"LLH Alt: {llh_alt:.4f}")
        for i, par in enumerate(par_alt):
            print(f"\tPar {i} alt: {par:.4f}")
        print(f"-2(LLH Null - LLH Alt): {llh_diff:.4f}")
        print(f"P(Ndof={Ndof}): {p_value:.4f}")
        print("")

    return llh_diff, p_value


# --MONTE CARLO --
def monte_carlo_sample_from_pdf(f_pdf, par, x_bound, MC_steps):
    x_pdf = []
    rng = np.random.default_rng()
    while len(x_pdf) < MC_steps:
        x = rng.uniform(low=x_bound[0], high=x_bound[1])
        u = rng.uniform(low=0, high=1)
        if f_pdf(x, *par) > u:
            x_pdf.append(x)
    return np.array(x_pdf)


def points_on_sphere_gauss(N: int) -> tuple:
    """Randomly distributed points on a unit sphere.

    Args:
        N (int): Number of points on the sphere

    Returns:
        tuple: x, y, z
    """
    xyz = np.random.normal(loc=0, scale=1, size=(3, N))
    x = xyz[0, :]
    y = xyz[1, :]
    z = xyz[2, :]
    scale = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= scale
    y /= scale 
    z /= scale
    return x, y, z


def mcmc_accept_ratio(f_likelihood, f_prior, f_marginal, x_current, x_proposed, par_prior, par_lh, par_marginal):
    """Default Metropolis Hasting acceptance ratio function.    

    Args:
        f_likelihood (_type_): _description_
        f_prior (_type_): _description_
        f_marginal (_type_): _description_
        x_current (_type_): _description_
        x_proposed (_type_): _description_
        par_prior (_type_): _description_
        par_lh (_type_): _description_
        par_marginal (_type_): _description_

    Returns:
        _type_: _description_
    """
    P_current = bayesian_posterior(f_likelihood, f_prior, f_marginal, x_current, par_prior, par_lh, par_marginal)
    P_proposed = bayesian_posterior(f_likelihood, f_prior, f_marginal, x_proposed, par_prior, par_lh, par_marginal)
    return P_proposed / P_current
        

def metropolis_hasting_sample_posterior(f_accept, f_propose, x0: np.array, steps: int, par_lh: tuple, par_prior: tuple, par_marginal: tuple, x_bound: tuple) -> np.array:
    """Sample from a posterior distribution given the the likelihood, prior and marginal functions together with a proposal function.

    Args:
        f_likelihood (function): The likelihood 
        f_prior (function): Prior pdf
        f_marginal (function): Marginal pdf 
        f_propose (function): How new steps are chosen 
        x0 (np.array): Initial guess/values
        steps (int): Number of MCMC steps
        par_lh (tuple): Likelihood parameter values
        par_prior (tuple): Prior parameter values
        par_marginal (tuple): Marginal parameter values
        x_bound (tuple): The values in which x must be within

    Returns:
        np.array: Values sampled from posterior
    """
    x_posterior = [x0]
    rng = np.random.default_rng()
    while len(x_posterior) < steps:
        # Proposed step
        x_current = x_posterior[-1]
        # Keep proposing until find a legal value
        proposed_is_legal = False
        while not proposed_is_legal:
            x_proposed = f_propose(x_current)
            proposed_is_legal = np.logical_and(x_proposed > x_bound[0], x_proposed < x_bound[1])
        
        # Acceptance ratio
        r = f_accept(x_current, x_proposed, par_lh, par_prior, par_marginal)
        u = rng.uniform(low=0, high=1)
        if r > u:  # Accept proposed update
            x_posterior.append(x_proposed)
        else:  # Reject update
            x_posterior.append(x_current)
            
    return np.array(x_posterior)


def plot_metropolis_hasting(f_accept, f_propose, x0, steps, par_prior, par_lh, par_marginal, x_bound, fig_save_path=""):
    x_posterior = metropolis_hasting_sample_posterior(f_accept, f_propose, x0, steps, par_prior, par_lh, par_marginal, x_bound)
    
    fig, (ax, ax1) = plt.subplots(ncols=2)
    Nbins = int(np.sqrt(steps))
    counts, bins, _ = ax.hist(x_posterior, bins=Nbins, density=True)
    ax.set(xlabel="Parameter value", ylabel="Probability dist.")
    ax1.plot(x_posterior, ".")
    ax1.set(xlabel="Steps", ylabel="MCMC val")
    
    # Find Maxiumum a posteriori
    MAP = bins[np.argmax(counts)]
    print("Maximum: ", MAP)
    plt.savefig(fig_save_path + "MC_posterior_sample.png")
    plt.show()


# -- KERNEL DENSITY ESTIMATION
def twodim_KDE(x, y, bandwidth, x_eval=[], y_eval=[], N_eval_points=100, kernel="gaussian"):
    """_summary_

    Args:
        x (array): x data 
        y (array): y data
        bandwidth (float): bandwidth for KDE
        x_eval (array, optional): Points at which to evaluate KDE. Defaults to whole space.
        y_eval (array, optional): Points at which to evaluate. Defaults to whole space.
        N_eval_points (int, optional): _description_. Defaults to 100.
        kernel (str, optional): _description_. Defaults to "gaussian".

    Returns:
        _type_: _description_
    """
    
    assert kernel in ["gaussian", "epanechnikov"]
    # Create grid for evaluation
    # If not given specific evaluation points, evaluate over the whole grid
    if len(x_eval) == 0 and len(y_eval) == 0:
        x_eval = np.linspace(x.min(), x.max(), N_eval_points)
        y_eval = np.linspace(y.min(), y.max(), N_eval_points)
    xx_eval, yy_eval = np.meshgrid(x_eval, y_eval)
    
    # Shape data and eval points correctly for sklearn
    xy_eval = np.vstack([yy_eval.ravel(), xx_eval.ravel()]).T
    xy_data = np.vstack([y, x]).T
    
    # KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(xy_data)
    
    # Prob
    P = np.exp(kde.score_samples(xy_eval))
    
    return xx_eval, yy_eval, P


def plot_twodim_KDE(x, y, bandwidth, N_eval_points, kernel, xlabel="x", ylabel="y", title="", figsave_path=""):
    xx, yy, P = twodim_KDE(x, y, bandwidth, N_eval_points=N_eval_points, kernel=kernel)
    P = np.reshape(P, xx.shape)
    fig, ax = plt.subplots()
    #im = ax.pcolormesh(xx, yy, P)
    im = ax.contourf(xx, yy, P, cmap="magma")
    ax.scatter(x, y, s=1, facecolor="white", alpha=0.01)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(im)
    figname = figsave_path + "two_dim_kde.png"
    plt.savefig(figname)
    plt.show()


# -- MACHINE LEARNING
def hist_ML_training_data(X_signal, X_background, ncols, nrows, labels=["Signal", "Background"], title_list=[], Nbin_list=-1, figsave_path=""):    
    """Plot background and signal data in histograms for each variable

    Args:
        X_signal ((Data, var)-array): Data along rows, variables along columns. Signal data
        X_background ((Data, var)-array): Data along rows, variables along columns. Background data
        ncols (int): Number of Figure columns. ncols times nrows must be equal to number of variables.
        nrows (int): Number of Figure rows. ncols times nrows must be equal to number of variables.
        Nbin_list (list of int, optional): If set will specify the number of bins for each figure. Default is square root of number of data points. Defaults to -1.
    """
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows)

    if np.any(Nbin_list == -1):
        Nbin_list = -1 * np.ones_like(ax.flatten())
    if len(title_list) == 0:
        title_list = [""] * ncols * nrows  # Empty title for each figure
    
    def plot_hist(axis, signal, background, Nbins, title):
        # Bins
        if Nbins == -1:
            Nbins = int(np.sqrt(len(signal)))
        min_val = np.min((signal.min(), background.min()))
        max_val = np.max((signal.max(), background.max()))
        bins = np.linspace(min_val, max_val, Nbins)
        # Plot
        counts_sig, _, _ = axis.hist(signal, bins=bins, color="black", label=labels[0], histtype="step")
        counts_back, _, _ = axis.hist(background, bins=bins, color="red", label=labels[1], histtype="step") 
        # Ticks and labels
        bin_width = bins[1] - bins[0]
        counts_min = np.min((counts_sig.min(), counts_back.min()))
        counts_max = np.max((counts_sig.max(), counts_back.max()))
        xticks_vals = np.linspace(min_val, max_val, 5)
        yticks_vals = np.linspace(counts_min, counts_max, 5)
        axis.set_xticks(xticks_vals, labels=xticks_vals, fontsize=4)
        axis.set_yticks(yticks_vals, labels=yticks_vals, fontsize=4)
        axis.set_ylabel(ylabel=f"Bin width = {bin_width:.3f}", fontsize=6)
        axis.set_title(label=title, fontsize=8)
        
    for axis, signal, background, Nbins, title in zip(ax.flatten(), X_signal.T, X_background.T, Nbin_list, title_list):
        plot_hist(axis, signal, background, Nbins, title)
    
    fig.legend(labels, ncols=2, loc="upper center")
    figname = figsave_path + "signal_background_variable_hist.png"
    plt.savefig(figname)
    plt.show()


def plot_BDT_test_stat(clf, X_signal, X_background, labels=["Signal", "Background"], figsave_path="", return_data=False):
    """Calculate and plot the BDT score using the AdaBoostClassifier.

    Args:
        train_data_signal ((Npoints, var)-array): Signal data
        train_data_background ((Npoints, var)-array): Background data.
    """
    df_signal = clf.decision_function(X_signal)
    df_background = clf.decision_function(X_background)

    bins = np.linspace(-1, 1, 50)
    counts_signal, edges_signal = np.histogram(df_signal, bins=bins)
    counts_background, edges_background = np.histogram(df_background, bins=bins)

    if return_data:
        return counts_signal, edges_signal, counts_background, edges_background    
        
    fig, ax = plt.subplots()
    
    ax.hist(df_background, bins=bins, label=labels[1], color="red", histtype="bar", alpha=0.4, edgecolor=fig_background_color)
    ax.hist(df_signal, bins=bins, label=labels[0], color="black", histtype="bar", alpha=0.4, edgecolor=fig_background_color)
    
    ax.set(xlabel="BDT score", label="Counts")
    ax.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncols=2)
    figname = figsave_path + "BDT_score.png"
    plt.savefig(figname)
    plt.show()
    

def BDT_fit(X_train, y_train, n_estimators=100, learning_rate=1, max_depth=1, estimator=None, random_state=None):
    """Calculate AdaBoostClassifier fit classifier

    Args:
        train_data_signal ((Npoints, var)-array): Signal data
        train_data_background ((Npoints, var)-array): Background data.
    """    
    # Classifier
    assert estimator in [None, "logisticRegression", "SVC"]
    if estimator == "logisticRegression":
        estimator = LogisticRegression()
    elif estimator == "SVC":
        estimator = SVC()
    elif max_depth > 1:
        estimator = DecisionTreeClassifier(max_depth=max_depth)
    clf = AdaBoostClassifier(n_estimators=n_estimators, estimator=estimator, algorithm="SAMME", learning_rate=learning_rate, random_state=random_state).fit(X_train, y_train)
    return clf


# evaluate a given model using cross-validation
def cross_validation_clf_evaluation(clf, X_train, y_train):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores


def fit_evaluation_and_plot_CM(clf, X_test, y_test):
    # Conc. data and make predictions
    y_predictions = clf.predict(X_test)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predictions)
    
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation="nearest", extent=(0, 1, 0, 1), cmap="cividis")
    ax.set_xlabel("Predicted", fontstyle="oblique")
    ax.set_ylabel("Actual", fontstyle="oblique")
    ax.set_xticks([0.25, 0.75], labels=["Positive", "Negative"], fontsize=6)
    ax.set_yticks([0.25, 0.75], labels=["Negative", "Positive"], fontsize=6)
    ax.text(0.25, 0.25, conf_matrix[1, 0])
    ax.text(0.75, 0.25, conf_matrix[1, 1])
    ax.text(0.25, 0.75, conf_matrix[0, 0])
    ax.text(0.75, 0.75, conf_matrix[0, 1])

    plt.colorbar(im)
    plt.show()
    
    # Accuracy, Precision, Recall, and F1 score
    accuracy = accuracy_score(y_test, y_predictions)  # Correct predictions / Total predictions
    precision = precision_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions)
    f1score = f1_score(y_test, y_predictions)
    
    print(f"Accuracy & {accuracy:.3f}")
    print(f"Precision & {precision:.3f}")
    print(f"Recall & {recall:.3f}")
    print(f"F1 Score & {f1score:.3f}")


def calc_ROC(counts_signal, edges_signal, counts_background, edges_background) :
    """Calculate ROC curve given the counts and edges from signal and background.

    Args:
        counts_signal (1darray): Counts from binned signal data
        edges_signal (1darray): Counts from binned background data
        counts_background (1darray): Bin edges from signal data
        edges_background (1darray): Bin edges from background data

    Returns:
        (FPR, TPR): True positive rate (Sensitivity) and False positive rate (Fallout)
    """
    y_sig, x_sig_edges = counts_signal, edges_signal
    y_bkg, x_bkg_edges = counts_background, edges_background
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges):
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5 * (x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background for normalization
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = [] # True positive rate (sensitivity)
        FPR = [] # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            # The cut mask
            cut = (x_centers < x)
            selected_below = np.sum(cut)
            selected_above = np.sum(~cut)
            if selected_above == 0 or selected_below == 0:
                continue
            # True positive
            TP = np.sum(y_sig[~cut]) / selected_above    # True positives
            FN = np.sum(y_sig[cut]) / selected_below     # False negatives
            TPR.append(TP / (TP + FN))                   # True positive rate
            
            # True negative
            TN = np.sum(y_bkg[cut]) / selected_below      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / selected_above     # False positives
            FPR.append(FP / (FP + TN))                    # False positive rate            
            
        return np.array(FPR), np.array(TPR), x_centers
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")


def plot_BDT_ROC(clf, X_train, y_train, figsave_path="",):
    counts_signal, edges_signal, counts_background, edges_background = plot_BDT_test_stat(clf, X_train, y_train, return_data=True)
    FPR, TPR, _ = calc_ROC(counts_signal, edges_signal, counts_background, edges_background)    
    fig, ax = plt.subplots()
    ax.plot(FPR, TPR, ".-")
    ax.set(xlabel="False Positive Ratio", ylabel="True Positive Ratio")
    figname = figsave_path + "BDT_ROC_curve.png"
    plt.savefig(figname)
    plt.show()


def classify_real_data(clf, X, ID, cut_val, local_dir_path=""):
    # Predict and extract label indices
    score = clf.decision_function(X)
    idx_above_cut = np.where(score >= cut_val)
    idx_below_cut = np.where(score < cut_val)
    
    ID_low = ID[idx_below_cut]
    ID_high = ID[idx_above_cut]
    
    
    # Write indices to files
    with open(local_dir_path + "cwr879.below_cut.txt", "w", newline='') as f:
        for id in ID_low:
            writer = csv.writer(f)
            writer.writerow([id])
    with open(local_dir_path + "cwr879.above_cut.txt", "w", newline='') as f:
        for id in ID_high:
            writer = csv.writer(f)
            writer.writerow([id])


def plot_multinest_2d(optimize_func, prior_func, par1_eval, par2_eval, N_nestle: int, method="single", figsave_path=""):
    # Multinest
    result = nestle.sample(optimize_func, prior_func, npoints=N_nestle, ndim=2, method=method)
    par1, par2 = result.samples.T

    # LLH plot setup using KDE
    # Points
    xx, yy = np.meshgrid(par1_eval, par2_eval)
    kernel = scipy.stats.gaussian_kde([par1, par2])
    Z_eval = kernel(np.append(xx.reshape(-1, 1), yy.reshape(-1, 1), axis=1).T)
    Z_eval = np.reshape(Z_eval, xx.shape)
    P_KDE = kernel([par1, par2])

    # Levels
    one_sigma = np.sort(P_KDE)[-int(len(P_KDE) * 0.684)]  # Get 0.684% of data
    two_sigma = np.sort(P_KDE)[-int(len(P_KDE) * 0.90)]
    three_sigma = np.sort(P_KDE)[-int(len(P_KDE) * 0.95)]
    levels = [three_sigma, two_sigma, one_sigma]
    
    # # Plot
    fig, ax = plt.subplots()
    bar = ax.contour(xx, yy, Z_eval, levels=levels)  # KDE
    ax.scatter(par1, par2, s=2, alpha=0.8)  # Multinest samples
    ax.set(xlabel="Parameter 1", ylabel="Parameter 2")
    # Get percentages on 
    level_names = [r"95%", r"90%", r"68.4%"]
    fmt = {}
    for l, s in zip(bar.levels, level_names):
        fmt[l] = s
    ax.clabel(bar, bar.levels, inline=True, fmt=fmt)
    figname = figsave_path + "multi_nest.png"
    plt.savefig(figname)
    plt.show()


# -- NUMERICAL INTEGRATION --
def runge_kutta(dfdt, x0, time_steps, dt, **par):
    """Perform the 4th order runge kutta method on a differential dfdt.

    Args:
        dfdt (function): The RHS of dy/dt. First variable is time, next is x.
        x0 (array like): Initial values. Size equal to the number of variables.
        time_steps (int): Number of integration steps
        dt (float): Time step size.
        par (dictionary): Function parameters.

    Returns:
        tupple (array-like, array-like): Time values, function values
    """
    # Initial values
    try: 
        x_vals = np.empty((len(x0), time_steps))
    except AttributeError:
        x_vals = np.empty((1, time_steps))
    x_vals[:, 0] = x0
    t = dt
    for i in tqdm(range(1, time_steps)):
        x = x_vals[:, i-1]
        
        # k factors
        k1 = dfdt(t, 
                  x, 
                  **par)
        k2 = dfdt(t + dt / 2, 
                  x + dt * k1 / 2, 
                  **par)
        k3 = dfdt(t + dt / 2, 
                  x + dt * k2 / 2, 
                  **par)
        k4 = dfdt(t + dt, 
                  x + dt * k3,
                  **par)
        
        # Combine and find new x values
        x_vals[:, i] = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        
    t_vals = np.linspace(0, t, time_steps)
    return t_vals, x_vals


# -- SPECIAL CASES --

# - Astronomy / Two point correlation -
def random_point_on_sphere(N_tot):
    """
    Generates Random points on a unit sphere 
    with N_tot points using the method from: 
    N. (2024, March 16). In Wikipedia. https://en.wikipedia.org/wiki/N-sphere
    Returns the x,y,z coordinates of these points.
    """
 
    x_gauss,y_gauss,z_gauss = np.random.normal(size=N_tot), np.random.normal(size=N_tot), np.random.normal(size=N_tot)
    r = np.sqrt(x_gauss**2+ y_gauss**2 +z_gauss**2)
    x_marsa,y_marsa,z_marsa = 1/r* ( x_gauss,y_gauss,z_gauss)
    return x_marsa,y_marsa,z_marsa
 
 
def xyz_to_lonlat(x, y, z):
    """
    Transforms from xyz coordinate system to 
    longitude and lattitude so that it can be plotted 
    using mollweide
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return lon, lat


def plot_longitude_latitude(N_points):
    # Generate random points on surface and convert to longitude/latitude
    x, y, z = points_on_sphere_gauss(N_points)
    lon, lat = xyz_to_lonlat(x, y, z) 
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollweide')
    ax.grid()
    im = ax.scatter(lon, lat)
    plt.show()


def autocorrelation_two_point(cos_phi_compare, x, y, z):
    
    x_vals = x[:, None] * x[None, :]
    y_vals = y[:, None] * y[None, :]
    z_vals = z[:, None] * z[None, :]    
    
    cos_phi_ij = x_vals + y_vals + z_vals
    
    cos_phi_ij = np.tril(cos_phi_ij, k=-1).flatten()  # Avoid double counting
    cos_phi_ij = cos_phi_ij[np.nonzero(cos_phi_ij)]  # Remove 0's, otherwise negative cos(angles) will be counted below
    cos_phi_diff = cos_phi_ij - cos_phi_compare
    # Heaviside    
    heaviside = np.sum(np.heaviside(cos_phi_diff, 0))
    N_tot = x.size
    norm_fac = 2 / (N_tot * (N_tot - 1)) 
    
    return norm_fac * heaviside


def plot_autocorrelation_two_point(N_points: int) -> None:
    """Plot two point autocorrelation for isotropic points on a sphere

    Args:
        N_points (int): Number of points on the sphere
    """
    # Get values distributed on sphere
    x, y, z = points_on_sphere_gauss(N_points)
    
    # Loop over all the cos(phi) distances and count how many pairs are within that distance
    cos_phi_vals = np.linspace(-1, 1, 200)
    corr_vals = np.empty_like(cos_phi_vals)
    for i, cos_phi in enumerate(cos_phi_vals):
        val = autocorrelation_two_point(cos_phi, x, y, z)
        corr_vals[i] = val

    # Plot
    fig, ax = plt.subplots()
    ax.plot(cos_phi_vals, corr_vals)
    ax.set(xlabel=r"$\cos\phi$", ylabel=r"Cum. auto-corr. $C(\phi)$")
    plt.show()


# -- GENERAL --  
def to_spherical(x, y, z) -> tuple:
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
    radius = x ** 2 + y ** 2 + z ** 2
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi = np.arctan2(y, x)
    return (radius, theta, phi)


def vec_mag(x):
    """Find the magnitude of a vector x

    Args:
        x (array): vector

    Returns:
        float: Vector magnitude
    """
    return np.sqrt(np.sum(x**2))


def find_nearest(x, val) -> int:
    """Find the value in the array x that is closest to the given value val.

    Args:
        x (1darray): Data values
        val (float): Value we want an index in x close to.

    Returns:
        int: Index of the value in x closest to val
    """
    idx = np.abs(x - val).argmin()
    return idx


