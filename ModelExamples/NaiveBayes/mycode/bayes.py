import pandas as pd
import math
import numpy as np

# Calculate value of Gaussian probability density function at x
def calc_probability_density(x, mu, sigma):
    p = []
    for v in x:
        exp_term = math.exp( -0.5 * ( (v-mu)/sigma )**2 )
        coeff_term = 1/(sigma*math.sqrt(2*math.pi) )
        p.append(coeff_term*exp_term)
    return p


# Calculate the conditional probability of a particular observation, given data
# Assumes a Gaussian probability distribution parameterized by mean and std.
def calc_probability(observations, stats):

    assert( (observations.keys() == stats.keys()).all() )

    probs = []
    for c in observations:
        p = calc_probability_density(observations[c].tolist(), mu=stats[c]['mean'], sigma=stats[c]['std'])
        probs.append(p)

    probs = pd.DataFrame(probs).transpose()
    probs.columns = observations.keys()
    return probs

# DUH
def calc_mean_std(x, ddof):
    tmp_mean = x.apply('mean')
    tmp_std = x.apply('std', ddof=ddof)
    tmp = pd.DataFrame([tmp_mean, tmp_std], index=['mean','std'])
    return tmp

def bayes(df_x, df_y):
    p_y = np.log(df_y.value_counts()/len(df_y))

    # DEBUG
    import pdb; pdb.set_trace();


    p_x = np.log(calc_probability(df_x, calc_mean_std(df_x, 1))).sum(axis=1)
    p_x_given_no_diabetes = np.log(calc_probability(df_x, calc_mean_std(df_x[df_y == 0],0))).sum(axis=1)
    p_x_given_diabetes = np.log(calc_probability(df_x, calc_mean_std(df_x[df_y == 1],0))).sum(axis=1)

    posterior_n = p_y[0] + p_x_given_no_diabetes#.add(-p_x)
    posterior_y = p_y[1] + p_x_given_diabetes#.add(-p_x)

    posterior = np.exp(pd.DataFrame([posterior_y, posterior_n])).transpose()
    posterior.columns = ['diabetes','no_diabetes']
    posterior = posterior.div(posterior.sum(axis=1), axis=0)

    mdl_classification = posterior['diabetes'] > posterior['no_diabetes']

    return mdl_classification

def hello_world():
    return 'hello world!'
