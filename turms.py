import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


def drawdown(return_series= pd.Series):
    '''
    Returns wealth index, previous peaks and drawdown into a Pandas Dataframe.
    '''
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdowns
        }
    )


def skewness(r):
    '''
    Returns skewness of Series or Dataframe.
    '''
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp / sigma_r**3


def kurtosis(r):
    '''
    Returns kurtosis of Series or Dataframe.
    '''
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp / sigma_r**4


def is_normal(r, level=0.01):
    '''
    Tests Jarque-Bera normality on set of units.
    '''
    stat, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def myround(x, base=0.05):
    return round(base * round(x/base), 3)


def binnify(indx, base=0.05, bin_width=0.025):
    min = indx.min()
    max = indx.max()
    lower = myround(min, base)
    upper = myround(max, base)
    num = round((upper - lower) / bin_width) + 1 
    bins = np.linspace(lower, upper, num)
    result = np.zeros(len(bins) - 1)
    for i in indx:
        for b in range(0, len(bins) - 1):
            if i < bins[b + 1]:
                result[b] += 1
                break
    return pd.DataFrame({'x': bins[:-1], 'y':result})


def binnify_table(table, base=0.05, bin_width=0.025):
    return_table = pd.DataFrame()
    for index in table.columns:
        df = binnify(table[index], base, bin_width)
        df.columns = pd.MultiIndex.from_product([['USD-BTC'], df.columns])
        pd.concat([return_table, df], axis=1)
    return return_table


def var_historic(r, level=5):
    '''
    VaR historic.
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be series or dataframe")


def var_gaussian(r, level=5, modified=False):
    '''
    Parametric Gaussian
    '''
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    '''
    CVaR historic.
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= - var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be series or dataframe")


def annualize_rets(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def portfolio_return(weights, returns):
    '''
    Weights --> Returns.
    '''
    return np.dot(weights.T, returns)


def portfolio_vol(weights, covmat):
    '''
    Weights --> Vol
    '''
    return np.dot(weights.T, np.dot(covmat, weights))**0.5


def plot_ef2(n_points, er, cov):
    '''
    Plots 2 Asset ef.
    '''
    if er.shape[0] != 2 or er.shape[1] != 2:
        raise ValueError("plot_ef2 can only plot 2 asset-frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
    return ef.plot.line(x='Volatility', y='Returns', style=".-")


def minimize_vol(target_return, er, cov):
    '''
    Target_ret --> W
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                        args=(cov,), method='SLSQP',
                        options={'disp': False},
                        constraints=(return_is_target, weights_sum_to_1),
                        bounds=bounds
                        )
    return results.x


def optimal_weights(n_points, er, cov):
    '''
    Weights for optimizer
    '''
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate, er, cov):
    '''
    Riskfree rate + ER + COV --> W
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        Negative of Sharpe ratio, given weights.
        '''
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio, init_guess,
                        args=(riskfree_rate, er, cov,), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1),
                        bounds=bounds
                        )
    return results.x


def gmv(cov):
    '''
    Returns the weights of the Global Minimum Vol Portfolio given the covariance matrix.
    '''
    n = cov.shape[0]
    # Give fake vector.
    return msr(0, np.repeat(1, n), cov)


def plot_ef(n_points, er, cov, show_cml=False, style=".-", riskfree_rate=0, show_ew=False, show_gmv=False):
    '''
    Plots N Asset ef.
    '''
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets, 
        'Volatility': vols
        })
    ax = ef.plot.line(x='Volatility', y='Returns', style=style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display EW
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display GMV
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker="o", linestyle="dashed", markersize=12, linewidth=2)
    return ax