import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests

from tqdm import tqdm

import patsy

def lr_tests(sample_info, expression_matrix, full_model, reduced_model='expression ~ 1'):
    ''' Compare full_model and reduced_model by a Likelihood Ratio Test
    for every gene in the expression_matrix.
    '''
    tmp = sample_info.copy()

    fit_results = pd.DataFrame(index=expression_matrix.index)

    gene = expression_matrix.index[0]
    tmp['expression'] = expression_matrix.ix[gene]
    m1 = smf.ols(full_model, tmp).fit()
    m2 = smf.ols(reduced_model, tmp).fit()

    for param in m1.params.index:
        fit_results['full ' + param] = np.nan

    params = m1.params.add_prefix('full ')
    fit_results.ix[gene, params.index] = params

    for param in m2.params.index:
        fit_results['reduced ' + param] = np.nan

    params = m2.params.add_prefix('reduced ')
    fit_results.ix[gene, params.index] = params

    fit_results['pval'] = np.nan

    fit_results.ix[gene, 'pval'] = m1.compare_lr_test(m2)[1]

    for gene in tqdm(expression_matrix.index[1:]):
        tmp['expression'] = expression_matrix.ix[gene]

        m1 = smf.ols(full_model, tmp).fit()
        params = m1.params.add_prefix('full ')
        fit_results.ix[gene, params.index] = params

        m2 = smf.ols(reduced_model, tmp).fit()
        params = m2.params.add_prefix('reduced ')
        fit_results.ix[gene, params.index] = params

        fit_results.ix[gene, 'pval'] = m1.compare_lr_test(m2)[1]

    fit_results['qval'] = multipletests(fit_results['pval'], method='b')[1]

    return fit_results


def regress_out(sample_info, expression_matrix, covariate_formula, design_formula='1'):
    ''' Implementation of limma's removeBatchEffect function
    '''
    # Ensure intercept is not part of covariates
    covariate_formula += ' - 1'
    covariate_matrix = patsy.dmatrix(covariate_formula, sample_info)
    design_matrix = patsy.dmatrix(design_formula, sample_info)

    design_batch = np.hstack((design_matrix, covariate_matrix))

    coefficients, res, rank, s = np.linalg.lstsq(design_batch, expression_matrix.T)
    beta = coefficients[-design_matrix.shape[1]][:, None]
    regressed = expression_matrix - beta.dot(covariate_matrix.T)

    return regressed
