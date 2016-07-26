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


def in_silico_fold_change(concentration, fold_change_limit=9):
    ''' Take a series of known concentration, and a fold change limit, and relabel
    the concentrations such that new labels are within the fold change limit.

    Returns the given concentrations, the replaced concentration, the fold change between them,
    as well as the dictionary used to make the relabeling.
    '''
    conc_df = pd.DataFrame(concentration)
    conc_df.columns = ['concentration']
    global_candidates = conc_df.index

    # Create swappings which are consistent with fold change limit
    candidates = {}
    for e in conc_df.index:
        cc = conc_df.loc[e, 'concentration']
        ll = cc / fold_change_limit
        ul = cc * fold_change_limit
        candidates[e] = conc_df.query('{} < concentration < {}'.format(ll, ul)).index
        candidates[e] = candidates[e].drop(e)

    replacement = {}
    for e in conc_df.index:
        if e not in global_candidates:
            continue

        possible_swaps = global_candidates.intersection(candidates[e])
        # break
        if len(possible_swaps) < 1:
            replacement[e] = e
            global_candidates = global_candidates.drop(replacement[e])
        else:
            replacement[e] = np.random.choice(possible_swaps, replace=False)
            replacement[replacement[e]] = e
            global_candidates = global_candidates.drop(replacement[e])
            global_candidates = global_candidates.drop(e)

    # Make randomly swapped annotation
    shuff_concentration = concentration.copy().rename(replacement)
    concentration = concentration.sort_index()
    shuff_concentration = shuff_concentration.sort_index()

    log2_fc = np.log2(shuff_concentration / concentration)

    return concentration, shuff_concentration, log2_fc, replacement


def in_silico_conditions(expression_table, replacement):
    ''' Takes and expression table, and a dictionary for remapping gene names
    in the expression table.

    This randomly partitions the table in to two conditions, A and B.
    The B condition will have genes renamed based on the replacement dict.

    Returns the new table, and annotation about which is which.
    '''
    # Split samples in two
    n_samples = expression_table.columns.shape[0]
    shuffled_samples = np.random.choice(expression_table.columns, n_samples, replace=False)
    A_samples = shuffled_samples[:n_samples // 2]
    B_samples = shuffled_samples[n_samples // 2:]

    # Swap input abundance annotation for half of the samples
    A_table = expression_table[A_samples]
    B_table = expression_table[B_samples]
    B_table = B_table.rename(index=replacement)

    # Put everything back to a single table
    shuff_table = A_table.join(B_table)

    # Create annotation for the shuffled samples
    sample_info = pd.DataFrame({'n_genes': (shuff_table > 1.).sum(0)})
    sample_info['condition'] = \
    ['A' if s in A_samples else 'B' for s in sample_info.index]

    return shuff_table, sample_info
