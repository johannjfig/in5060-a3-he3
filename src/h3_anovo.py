#!/usr/bin/env python3
"""
H3 Learning Effect Analysis Script
===================================
Analyzes whether participants improve performance over repeated trials.

Usage:
    python H3_learning_analysis.py <xlsx_file>
    
Example:
    python H3_learning_analysis.py questionnaire_data-561422-2025-11-11-1622.xlsx

Requirements:
    pip install pandas openpyxl scipy pingouin matplotlib numpy
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

# Import assumption tests module
try:
    from assumption_tests import (
        check_missing_values, print_missing_values_report,
        print_assumption_tests, save_assumption_tests
    )
    HAS_ASSUMPTION_TESTS = True
except ImportError:
    HAS_ASSUMPTION_TESTS = False
    print("Note: assumption_tests.py not found - skipping assumption checks")

# Try to import pingouin for ANOVA, fall back to scipy if not available
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not installed. Install with: pip install pingouin")
    print("Using scipy for basic analysis instead.\n")

def load_and_reshape_data(xlsx_path):
    """Load xlsx file and reshape for trial-by-trial analysis."""
    
    # Load data
    df = pd.read_excel(xlsx_path)
    
    # Clean column names (fix HTML encoding)
    df.columns = df.columns.str.replace('&#39;', "'")
    
    # Extract participant info
    participant_ids = df['Participant number'].values
    n_participants = len(participant_ids)
    
    print(f"Loaded {n_participants} participants")
    print(f"Participant IDs: {list(participant_ids)}")
    
    # Extract difficulty ratings (Q2) for each trial (0-9)
    difficulty_cols = ['How difficult was it to perform the task?']
    for i in range(1, 10):
        difficulty_cols.append(f'How difficult was it to perform the task?.{i}')
    
    # Extract control ratings (Q3) for each trial
    control_cols = ['I felt like I was controlling the movement of the robot']
    for i in range(1, 10):
        control_cols.append(f'I felt like I was controlling the movement of the robot.{i}')
    
    # Extract delay perception (Q1) for each trial
    delay_cols = ["Did you experience delays between your actions and the robot's movements?"]
    for i in range(1, 10):
        delay_cols.append(f"Did you experience delays between your actions and the robot's movements?.{i}")
    
    # Extract embodiment (Q4) for each trial
    embodiment_cols = ['It felt like the robot was part of my body']
    for i in range(1, 10):
        embodiment_cols.append(f'It felt like the robot was part of my body.{i}')
    
    # Get data matrices
    difficulty_data = df[difficulty_cols].values
    control_data = df[control_cols].values
    delay_data = df[delay_cols].values
    embodiment_data = df[embodiment_cols].values
    
    # Create long-format dataframe for ANOVA
    long_df = pd.DataFrame({
        'Participant': np.repeat(participant_ids, 10),
        'Trial': np.tile(np.arange(1, 11), n_participants),
        'Difficulty': difficulty_data.flatten(),
        'Control': control_data.flatten(),
        'Delay': delay_data.flatten(),
        'Embodiment': embodiment_data.flatten()
    })
    
    return long_df, difficulty_data, control_data, participant_ids

def calculate_descriptive_stats(data, participant_ids):
    """Calculate means and SDs per trial."""
    n_trials = data.shape[1]
    
    stats_df = pd.DataFrame({
        'Trial': range(1, n_trials + 1),
        'Mean': np.nanmean(data, axis=0),
        'SD': np.nanstd(data, axis=0, ddof=1),
        'SEM': np.nanstd(data, axis=0, ddof=1) / np.sqrt(data.shape[0]),
        'Min': np.nanmin(data, axis=0),
        'Max': np.nanmax(data, axis=0)
    })
    
    return stats_df

def run_repeated_measures_anova(long_df, dv_name):
    """Run repeated-measures ANOVA using pingouin."""
    if not HAS_PINGOUIN:
        print(f"  Skipping RM-ANOVA for {dv_name} (pingouin not installed)")
        return None
    
    rm_anova = pg.rm_anova(data=long_df, dv=dv_name, within='Trial', subject='Participant')
    return rm_anova

def run_paired_ttest(long_df, dv_name, trial_a=1, trial_b=10):
    """Run paired t-test between two trials."""
    data_a = long_df[long_df['Trial'] == trial_a][dv_name].values
    data_b = long_df[long_df['Trial'] == trial_b][dv_name].values
    
    t_stat, p_val = stats.ttest_rel(data_a, data_b)
    
    # Cohen's d for paired samples
    diff = data_a - data_b
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        'trial_a': trial_a,
        'trial_b': trial_b,
        'mean_a': np.mean(data_a),
        'sd_a': np.std(data_a, ddof=1),
        'mean_b': np.mean(data_b),
        'sd_b': np.std(data_b, ddof=1),
        't_stat': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'df': len(data_a) - 1
    }

def calculate_oneway_anova(long_df, dv='Difficulty', iv='Trial'):
    """
    Calculate one-way ANOVA manually to get SSA, SSE, SST like in lecture slides.
    
    Returns a dictionary with all ANOVA components.
    """
    
    # Get groups
    groups = long_df.groupby(iv)[dv]
    group_names = sorted(long_df[iv].unique())
    k = len(group_names)  # number of groups
    
    # Calculate group means and sizes
    group_means = groups.mean()
    group_sizes = groups.size()
    grand_mean = long_df[dv].mean()
    N = len(long_df)  # total observations
    
    # Calculate SSA (Sum of Squares Between/Alternatives)
    SSA = sum(group_sizes[g] * (group_means[g] - grand_mean)**2 for g in group_names)
    
    # Calculate SSE (Sum of Squares Within/Error)
    SSE = 0
    for g in group_names:
        group_data = long_df[long_df[iv] == g][dv]
        group_mean = group_means[g]
        SSE += sum((y - group_mean)**2 for y in group_data)
    
    # Calculate SST (Total Sum of Squares)
    SST = SSA + SSE
    
    # Degrees of freedom
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    
    # Mean squares
    MSA = SSA / df_between
    MSE = SSE / df_within
    
    # F-statistic
    F = MSA / MSE
    
    # p-value
    p_value = 1 - stats.f.cdf(F, df_between, df_within)
    
    # Effect size (eta-squared)
    eta_squared = SSA / SST
    
    # Critical F values
    F_crit_95 = stats.f.ppf(0.95, df_between, df_within)
    F_crit_99 = stats.f.ppf(0.99, df_between, df_within)
    F_crit_999 = stats.f.ppf(0.999, df_between, df_within)
    
    return {
        'SSA': SSA,
        'SSE': SSE,
        'SST': SST,
        'df_between': df_between,
        'df_within': df_within,
        'df_total': df_total,
        'MSA': MSA,
        'MSE': MSE,
        'F': F,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'F_crit_95': F_crit_95,
        'F_crit_99': F_crit_99,
        'F_crit_999': F_crit_999,
        'k': k,
        'N': N,
        'group_means': group_means,
        'grand_mean': grand_mean
    }

def print_anova_table(anova_results, dv_name, iv_name):
    """Print ANOVA results in a formatted table like lecture slides."""
    
    print(f"\n{'='*70}")
    print(f"ONE-WAY ANOVA: {dv_name} by {iv_name}")
    print(f"{'='*70}")
    
    # Print group means (first few and last few for brevity)
    print(f"\nGroup Means (ȳ.j):")
    group_means = anova_results['group_means']
    groups = sorted(group_means.index)
    
    if len(groups) <= 6:
        for g in groups:
            print(f"  {iv_name} {g}: ȳ = {group_means[g]:.4f}")
    else:
        for g in groups[:3]:
            print(f"  {iv_name} {g}: ȳ = {group_means[g]:.4f}")
        print(f"  ...")
        for g in groups[-3:]:
            print(f"  {iv_name} {g}: ȳ = {group_means[g]:.4f}")
    print(f"  Grand Mean (ȳ): {anova_results['grand_mean']:.4f}")
    
    # Print ANOVA table
    print(f"\n{'-'*70}")
    print(f"{'Variation':<20} {'Sum of Squares':>15} {'df':>8} {'Mean Square':>15} {'F':>10}")
    print(f"{'-'*70}")
    
    print(f"{'Between (SSA)':<20} {anova_results['SSA']:>15.4f} {anova_results['df_between']:>8} {anova_results['MSA']:>15.4f} {anova_results['F']:>10.4f}")
    print(f"{'Within (SSE)':<20} {anova_results['SSE']:>15.4f} {anova_results['df_within']:>8} {anova_results['MSE']:>15.4f}")
    print(f"{'Total (SST)':<20} {anova_results['SST']:>15.4f} {anova_results['df_total']:>8}")
    print(f"{'-'*70}")
    
    # Print F-test results
    print(f"\nF-test Results:")
    print(f"  Computed F = {anova_results['F']:.4f}")
    print(f"  p-value = {anova_results['p_value']:.6f}")
    print(f"  η² (effect size) = {anova_results['eta_squared']:.4f}")
    
    print(f"\nCritical F values (df1={anova_results['df_between']}, df2={anova_results['df_within']}):")
    print(f"  F[0.95] = {anova_results['F_crit_95']:.4f}")
    print(f"  F[0.99] = {anova_results['F_crit_99']:.4f}")
    print(f"  F[0.999] = {anova_results['F_crit_999']:.4f}")
    
    # Significance
    if anova_results['p_value'] < 0.001:
        sig = "*** (p < 0.001)"
    elif anova_results['p_value'] < 0.01:
        sig = "** (p < 0.01)"
    elif anova_results['p_value'] < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "ns (not significant)"
    
    print(f"\nSignificance: {sig}")
    
    return

def save_anova_table(anova_results, dv_name, iv_name, output_path):
    """Save ANOVA results to CSV files."""
    
    anova_table = pd.DataFrame({
        'Source': ['Between (SSA)', 'Within (SSE)', 'Total (SST)'],
        'Sum_of_Squares': [anova_results['SSA'], anova_results['SSE'], anova_results['SST']],
        'df': [anova_results['df_between'], anova_results['df_within'], anova_results['df_total']],
        'Mean_Square': [anova_results['MSA'], anova_results['MSE'], np.nan],
        'F': [anova_results['F'], np.nan, np.nan],
        'p_value': [anova_results['p_value'], np.nan, np.nan],
        'eta_squared': [anova_results['eta_squared'], np.nan, np.nan]
    })
    
    anova_table.to_csv(output_path, index=False)
    
    # Also save summary statistics
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary = pd.DataFrame({
        'Statistic': ['SSA', 'SSE', 'SST', 'df_between', 'df_within', 'df_total', 
                      'MSA', 'MSE', 'F', 'p_value', 'eta_squared',
                      'F_crit_95', 'F_crit_99', 'F_crit_999', 'k', 'N', 'grand_mean'],
        'Value': [anova_results['SSA'], anova_results['SSE'], anova_results['SST'],
                  anova_results['df_between'], anova_results['df_within'], anova_results['df_total'],
                  anova_results['MSA'], anova_results['MSE'], anova_results['F'],
                  anova_results['p_value'], anova_results['eta_squared'],
                  anova_results['F_crit_95'], anova_results['F_crit_99'], anova_results['F_crit_999'],
                  anova_results['k'], anova_results['N'], anova_results['grand_mean']]
    })
    summary.to_csv(summary_path, index=False)
    
    return anova_table

def tukey_hsd_posthoc(long_df, dv='Difficulty', iv='Trial'):
    """
    Perform Tukey HSD post-hoc test for pairwise comparisons.
    
    Returns a DataFrame with all pairwise comparisons.
    """
    from scipy.stats import studentized_range
    
    # Get groups
    groups = long_df.groupby(iv)[dv]
    group_names = sorted(long_df[iv].unique())
    k = len(group_names)
    
    # Calculate group statistics
    group_means = groups.mean()
    group_sizes = groups.size()
    group_vars = groups.var()
    
    # Calculate MSE (pooled variance)
    N = len(long_df)
    SSE = sum((group_sizes[g] - 1) * group_vars[g] for g in group_names)
    df_within = N - k
    MSE = SSE / df_within
    
    # Tukey HSD critical value
    try:
        q_crit = studentized_range.ppf(0.95, k, df_within)
    except:
        q_crit = 4.47  # Approximate for k=10, df=290
    
    # Calculate all pairwise comparisons
    results = []
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i < j:
                mean_diff = group_means[g1] - group_means[g2]
                n1, n2 = group_sizes[g1], group_sizes[g2]
                
                # Standard error for unequal sample sizes
                se = np.sqrt(MSE * (1/n1 + 1/n2) / 2)
                
                # q statistic
                q = abs(mean_diff) / se
                
                # HSD (minimum significant difference)
                hsd = q_crit * se
                
                # p-value
                try:
                    p_value = 1 - studentized_range.cdf(q, k, df_within)
                except:
                    p_value = 0.05 if q > q_crit else 0.10
                
                # Significance
                significant = abs(mean_diff) > hsd
                
                results.append({
                    'Group1': g1,
                    'Group2': g2,
                    'Mean1': group_means[g1],
                    'Mean2': group_means[g2],
                    'Mean_Diff': mean_diff,
                    'SE': se,
                    'q': q,
                    'HSD': hsd,
                    'p_value': p_value,
                    'Significant': significant
                })
    
    posthoc_df = pd.DataFrame(results)
    return posthoc_df, q_crit, MSE

def print_posthoc_table(posthoc_df, dv_name, iv_name, q_crit, show_all=False):
    """Print post-hoc results in a formatted table."""
    
    print(f"\n{'='*70}")
    print(f"TUKEY HSD POST-HOC: {dv_name} by {iv_name}")
    print(f"{'='*70}")
    print(f"Critical q value (α=0.05): {q_crit:.4f}")
    
    # For Trial comparisons, focus on key comparisons
    if not show_all and len(posthoc_df) > 20:
        # Show only comparisons with Trial 1 and Trial 10
        key_comparisons = posthoc_df[
            (posthoc_df['Group1'] == 1) | 
            (posthoc_df['Group2'] == 10) |
            (posthoc_df['Group1'] == 10)
        ].copy()
        print(f"\nKey comparisons (Trial 1 and Trial 10):")
    else:
        key_comparisons = posthoc_df
    
    print(f"\n{'-'*90}")
    print(f"{'Comparison':<20} {'Mean Diff':>12} {'SE':>10} {'q':>10} {'HSD':>10} {'p':>10} {'Sig':>8}")
    print(f"{'-'*90}")
    
    for _, row in key_comparisons.iterrows():
        comp = f"T{int(row['Group1'])} vs T{int(row['Group2'])}"
        sig = "*" if row['Significant'] else ""
        print(f"{comp:<20} {row['Mean_Diff']:>12.4f} {row['SE']:>10.4f} {row['q']:>10.4f} {row['HSD']:>10.4f} {row['p_value']:>10.4f} {sig:>8}")
    
    print(f"{'-'*90}")
    
    # Highlight significant differences
    sig_pairs = posthoc_df[posthoc_df['Significant']]
    if len(sig_pairs) > 0:
        print(f"\nSignificant pairwise differences (p < 0.05): {len(sig_pairs)} pairs")
        # Show first few
        for idx, (_, row) in enumerate(sig_pairs.iterrows()):
            if idx < 10:
                print(f"  • Trial {int(row['Group1'])} vs Trial {int(row['Group2'])}: Δ = {row['Mean_Diff']:.3f}")
            elif idx == 10:
                print(f"  ... and {len(sig_pairs) - 10} more")
                break
    else:
        print("\nNo significant pairwise differences found.")
    
    return

def create_plots(long_df, output_dir='.'):
    """Create visualization plots."""
    
    # Calculate trial means
    trial_means = long_df.groupby('Trial').agg({
        'Difficulty': ['mean', 'std', 'sem'],
        'Control': ['mean', 'std', 'sem']
    }).reset_index()
    trial_means.columns = ['Trial', 'Diff_Mean', 'Diff_SD', 'Diff_SEM', 
                           'Ctrl_Mean', 'Ctrl_SD', 'Ctrl_SEM']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    trials = trial_means['Trial'].values
    
    # Plot 1: Difficulty
    ax1 = axes[0]
    means = trial_means['Diff_Mean'].values
    sems = trial_means['Diff_SEM'].values
    
    ax1.plot(trials, means, 'o-', color='#E74C3C', linewidth=2, markersize=8, label='Mean')
    ax1.fill_between(trials, means - 1.96*sems, means + 1.96*sems, 
                     alpha=0.3, color='#E74C3C', label='95% CI')
    
    # Trend line
    slope, intercept, r, p, se = stats.linregress(trials, means)
    ax1.plot(trials, slope * trials + intercept, '--', color='gray', 
             label=f'Trend (slope={slope:.3f})')
    
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Perceived Difficulty (1-5)', fontsize=12)
    ax1.set_title('H3: Difficulty Across Trials\n(Lower = Easier)', fontsize=14)
    ax1.set_xticks(trials)
    ax1.set_ylim(1, 4)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control
    ax2 = axes[1]
    means_ctrl = trial_means['Ctrl_Mean'].values
    sems_ctrl = trial_means['Ctrl_SEM'].values
    
    ax2.plot(trials, means_ctrl, 'o-', color='#3498DB', linewidth=2, markersize=8, label='Mean')
    ax2.fill_between(trials, means_ctrl - 1.96*sems_ctrl, means_ctrl + 1.96*sems_ctrl, 
                     alpha=0.3, color='#3498DB', label='95% CI')
    
    slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(trials, means_ctrl)
    ax2.plot(trials, slope_c * trials + intercept_c, '--', color='gray', 
             label=f'Trend (slope={slope_c:.3f})')
    
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Control Feeling (1-5)', fontsize=12)
    ax2.set_title('H3: Control Across Trials\n(Higher = Better)', fontsize=14)
    ax2.set_xticks(trials)
    ax2.set_ylim(2.5, 5)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'H3_learning_effect_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    # Boxplot comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    trial1_diff = long_df[long_df['Trial'] == 1]['Difficulty'].values
    trial10_diff = long_df[long_df['Trial'] == 10]['Difficulty'].values
    trial1_ctrl = long_df[long_df['Trial'] == 1]['Control'].values
    trial10_ctrl = long_df[long_df['Trial'] == 10]['Control'].values
    
    ax3 = axes2[0]
    bp1 = ax3.boxplot([trial1_diff, trial10_diff], labels=['Trial 1', 'Trial 10'],
                      patch_artist=True)
    bp1['boxes'][0].set_facecolor('#FADBD8')
    bp1['boxes'][1].set_facecolor('#E74C3C')
    ax3.set_ylabel('Perceived Difficulty (1-5)', fontsize=12)
    ax3.set_title('Difficulty: First vs Last Trial', fontsize=12)
    
    ax4 = axes2[1]
    bp2 = ax4.boxplot([trial1_ctrl, trial10_ctrl], labels=['Trial 1', 'Trial 10'],
                      patch_artist=True)
    bp2['boxes'][0].set_facecolor('#D4E6F1')
    bp2['boxes'][1].set_facecolor('#3498DB')
    ax4.set_ylabel('Control Feeling (1-5)', fontsize=12)
    ax4.set_title('Control: First vs Last Trial', fontsize=12)
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'H3_boxplot_comparison.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {boxplot_path}")
    
    return trial_means

def main(xlsx_path):
    """Main analysis function."""
    
    print("="*70)
    print("H3 LEARNING EFFECT ANALYSIS")
    print("="*70)
    
    # Check file exists
    if not os.path.exists(xlsx_path):
        print(f"Error: File not found: {xlsx_path}")
        sys.exit(1)
    
    # Load data
    print("\n1. LOADING DATA")
    print("-"*70)
    long_df, difficulty_data, control_data, participant_ids = load_and_reshape_data(xlsx_path)
    
    # Descriptive statistics
    print("\n2. DESCRIPTIVE STATISTICS")
    print("-"*70)
    
    print("\nDifficulty by Trial:")
    diff_stats = calculate_descriptive_stats(difficulty_data, participant_ids)
    print(diff_stats.to_string(index=False))
    
    print("\nControl by Trial:")
    ctrl_stats = calculate_descriptive_stats(control_data, participant_ids)
    print(ctrl_stats.to_string(index=False))
    
    # Check preprocessing and assumptions
    if HAS_ASSUMPTION_TESTS:
        print("\n2b. PREPROCESSING CHECK")
        print("-"*70)
        missing = check_missing_values(long_df, ['Difficulty', 'Control', 'Trial'])
        print_missing_values_report(missing)
        
        print("\n2c. ANOVA ASSUMPTION TESTS")
        print("-"*70)
        assumption_results_diff = print_assumption_tests(
            long_df, 'Difficulty', 'Trial', 
            'Perceived Difficulty', 'Trial'
        )
        assumption_results_ctrl = print_assumption_tests(
            long_df, 'Control', 'Trial', 
            'Control Feeling', 'Trial'
        )
    
    # Repeated-measures ANOVA
    print("\n3. REPEATED-MEASURES ANOVA")
    print("-"*70)
    
    anova_table = None  # Initialize
    
    print("\nDifficulty:")
    anova_diff = run_repeated_measures_anova(long_df, 'Difficulty')
    if anova_diff is not None:
        print(anova_diff.to_string())
        p_diff = anova_diff['p-GG-corr'].values[0] if anova_diff['sphericity'].values[0] == False else anova_diff['p-unc'].values[0]
        eta_diff = anova_diff['ng2'].values[0]
    else:
        p_diff = None
        eta_diff = None
    
    print("\nControl:")
    anova_ctrl = run_repeated_measures_anova(long_df, 'Control')
    if anova_ctrl is not None:
        print(anova_ctrl.to_string())
        p_ctrl = anova_ctrl['p-GG-corr'].values[0] if anova_ctrl['sphericity'].values[0] == False else anova_ctrl['p-unc'].values[0]
        eta_ctrl = anova_ctrl['ng2'].values[0]
    else:
        p_ctrl = None
        eta_ctrl = None
    
    # Save ANOVA results to a readable table
    if anova_diff is not None and anova_ctrl is not None:
        anova_table = pd.DataFrame({
            'Variable': ['Difficulty', 'Control'],
            'F': [anova_diff['F'].values[0], anova_ctrl['F'].values[0]],
            'df1': [int(anova_diff['ddof1'].values[0]), int(anova_ctrl['ddof1'].values[0])],
            'df2': [int(anova_diff['ddof2'].values[0]), int(anova_ctrl['ddof2'].values[0])],
            'p_uncorrected': [anova_diff['p-unc'].values[0], anova_ctrl['p-unc'].values[0]],
            'p_GG_corrected': [anova_diff['p-GG-corr'].values[0], anova_ctrl['p-GG-corr'].values[0]],
            'eta_squared_g': [anova_diff['ng2'].values[0], anova_ctrl['ng2'].values[0]],
            'epsilon_GG': [anova_diff['eps'].values[0], anova_ctrl['eps'].values[0]],
            'sphericity_violated': [not anova_diff['sphericity'].values[0], not anova_ctrl['sphericity'].values[0]],
            'W_spher': [anova_diff['W-spher'].values[0], anova_ctrl['W-spher'].values[0]],
            'p_spher': [anova_diff['p-spher'].values[0], anova_ctrl['p-spher'].values[0]]
        })
    else:
        anova_table = None
    
    # One-way ANOVA with detailed output
    print("\n" + "="*70)
    print("3b. ONE-WAY ANOVA (Detailed)")
    print("="*70)
    
    # ANOVA for Difficulty by Trial
    anova_diff_detailed = calculate_oneway_anova(long_df, dv='Difficulty', iv='Trial')
    print_anova_table(anova_diff_detailed, 'Difficulty', 'Trial')
    
    # Post-hoc for Difficulty
    posthoc_diff_tukey, q_crit_diff, _ = tukey_hsd_posthoc(long_df, dv='Difficulty', iv='Trial')
    print_posthoc_table(posthoc_diff_tukey, 'Difficulty', 'Trial', q_crit_diff)
    
    # ANOVA for Control by Trial
    anova_ctrl_detailed = calculate_oneway_anova(long_df, dv='Control', iv='Trial')
    print_anova_table(anova_ctrl_detailed, 'Control', 'Trial')
    
    # Post-hoc for Control
    posthoc_ctrl_tukey, q_crit_ctrl, _ = tukey_hsd_posthoc(long_df, dv='Control', iv='Trial')
    print_posthoc_table(posthoc_ctrl_tukey, 'Control', 'Trial', q_crit_ctrl)
    
    # Post-hoc: First vs Last trial
    print("\n4. POST-HOC: TRIAL 1 vs TRIAL 10 (Paired t-test)")
    print("-"*70)
    
    ttest_diff = run_paired_ttest(long_df, 'Difficulty', 1, 10)
    print(f"\nDifficulty:")
    print(f"  Trial 1:  M = {ttest_diff['mean_a']:.2f}, SD = {ttest_diff['sd_a']:.2f}")
    print(f"  Trial 10: M = {ttest_diff['mean_b']:.2f}, SD = {ttest_diff['sd_b']:.2f}")
    print(f"  t({ttest_diff['df']}) = {ttest_diff['t_stat']:.3f}, p = {ttest_diff['p_value']:.4f}")
    print(f"  Cohen's d = {ttest_diff['cohens_d']:.3f}")
    
    ttest_ctrl = run_paired_ttest(long_df, 'Control', 1, 10)
    print(f"\nControl:")
    print(f"  Trial 1:  M = {ttest_ctrl['mean_a']:.2f}, SD = {ttest_ctrl['sd_a']:.2f}")
    print(f"  Trial 10: M = {ttest_ctrl['mean_b']:.2f}, SD = {ttest_ctrl['sd_b']:.2f}")
    print(f"  t({ttest_ctrl['df']}) = {ttest_ctrl['t_stat']:.3f}, p = {ttest_ctrl['p_value']:.4f}")
    print(f"  Cohen's d = {ttest_ctrl['cohens_d']:.3f}")
    
    # Create plots
    print("\n5. CREATING PLOTS")
    print("-"*70)
    # Use h3data directory for outputs
    output_dir = 'out/h3out'
    os.makedirs(output_dir, exist_ok=True)
    trial_means = create_plots(long_df, output_dir)
    
    # Save data
    print("\n6. SAVING RESULTS")
    print("-"*70)
    
    # Save trial means
    means_path = os.path.join(output_dir, 'H3_trial_means.csv')
    trial_means.to_csv(means_path, index=False)
    print(f"  Saved: {means_path}")
    
    # Save long-format data
    long_path = os.path.join(output_dir, 'H3_long_format_data.csv')
    long_df.to_csv(long_path, index=False)
    print(f"  Saved: {long_path}")
    
    # Save repeated-measures ANOVA results table
    if anova_table is not None:
        anova_path = os.path.join(output_dir, 'H3_rm_anova_results.csv')
        anova_table.to_csv(anova_path, index=False)
        print(f"  Saved: {anova_path}")
    
    # Save detailed one-way ANOVA tables
    anova_diff_path = os.path.join(output_dir, 'H3_anova_difficulty.csv')
    save_anova_table(anova_diff_detailed, 'Difficulty', 'Trial', anova_diff_path)
    print(f"  Saved: {anova_diff_path}")
    
    anova_ctrl_path = os.path.join(output_dir, 'H3_anova_control.csv')
    save_anova_table(anova_ctrl_detailed, 'Control', 'Trial', anova_ctrl_path)
    print(f"  Saved: {anova_ctrl_path}")
    
    # Save Tukey HSD post-hoc results
    posthoc_diff_path = os.path.join(output_dir, 'H3_posthoc_difficulty_tukey.csv')
    posthoc_diff_tukey.to_csv(posthoc_diff_path, index=False)
    print(f"  Saved: {posthoc_diff_path}")
    
    posthoc_ctrl_path = os.path.join(output_dir, 'H3_posthoc_control_tukey.csv')
    posthoc_ctrl_tukey.to_csv(posthoc_ctrl_path, index=False)
    print(f"  Saved: {posthoc_ctrl_path}")
    
    # Save paired t-test results table
    posthoc_table = pd.DataFrame({
        'Variable': ['Difficulty', 'Control'],
        'Trial_A': [ttest_diff['trial_a'], ttest_ctrl['trial_a']],
        'Trial_B': [ttest_diff['trial_b'], ttest_ctrl['trial_b']],
        'Mean_A': [ttest_diff['mean_a'], ttest_ctrl['mean_a']],
        'SD_A': [ttest_diff['sd_a'], ttest_ctrl['sd_a']],
        'Mean_B': [ttest_diff['mean_b'], ttest_ctrl['mean_b']],
        'SD_B': [ttest_diff['sd_b'], ttest_ctrl['sd_b']],
        't_statistic': [ttest_diff['t_stat'], ttest_ctrl['t_stat']],
        'df': [ttest_diff['df'], ttest_ctrl['df']],
        'p_value': [ttest_diff['p_value'], ttest_ctrl['p_value']],
        'cohens_d': [ttest_diff['cohens_d'], ttest_ctrl['cohens_d']],
        'significant': [ttest_diff['p_value'] < 0.05, ttest_ctrl['p_value'] < 0.05]
    })
    posthoc_path = os.path.join(output_dir, 'H3_ttest_trial1_vs_10.csv')
    posthoc_table.to_csv(posthoc_path, index=False)
    print(f"  Saved: {posthoc_path}")
    
    # Assumption tests - save to separate directory
    if HAS_ASSUMPTION_TESTS:
        assumption_dir = 'out/tests'
        os.makedirs(assumption_dir, exist_ok=True)
        
        assumption_diff_path = os.path.join(assumption_dir, 'H3_assumption_tests_difficulty.csv')
        save_assumption_tests(assumption_results_diff, assumption_diff_path)
        print(f"  Saved: {assumption_diff_path}")
        
        assumption_ctrl_path = os.path.join(assumption_dir, 'H3_assumption_tests_control.csv')
        save_assumption_tests(assumption_results_ctrl, assumption_ctrl_path)
        print(f"  Saved: {assumption_ctrl_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: H3 LEARNING EFFECT")
    print("="*70)
    
    print("\nDifficulty:")
    if p_diff is not None:
        sig_diff = "SIGNIFICANT" if p_diff < 0.05 else "NOT significant"
        print(f"  ANOVA: {sig_diff} (p = {p_diff:.4f}, η²g = {eta_diff:.4f})")
    sig_ttest_diff = "SIGNIFICANT" if ttest_diff['p_value'] < 0.05 else "NOT significant"
    print(f"  Trial 1 vs 10: {sig_ttest_diff} (p = {ttest_diff['p_value']:.4f})")
    direction_diff = "DECREASED" if ttest_diff['mean_a'] > ttest_diff['mean_b'] else "INCREASED"
    print(f"  Direction: {direction_diff} (supports learning)" if direction_diff == "DECREASED" else f"  Direction: {direction_diff}")
    
    print("\nControl:")
    if p_ctrl is not None:
        sig_ctrl = "SIGNIFICANT" if p_ctrl < 0.05 else "NOT significant"
        print(f"  ANOVA: {sig_ctrl} (p = {p_ctrl:.4f}, η²g = {eta_ctrl:.4f})")
    sig_ttest_ctrl = "SIGNIFICANT" if ttest_ctrl['p_value'] < 0.05 else "NOT significant"
    print(f"  Trial 1 vs 10: {sig_ttest_ctrl} (p = {ttest_ctrl['p_value']:.4f})")
    direction_ctrl = "DECREASED" if ttest_ctrl['mean_a'] > ttest_ctrl['mean_b'] else "INCREASED"
    print(f"  Direction: {direction_ctrl} (fatigue effect)" if direction_ctrl == "DECREASED" else f"  Direction: {direction_ctrl}")
    
    print("\n" + "="*70)
    print("H3 VERDICT:")
    print("="*70)
    
    # Determine verdict
    difficulty_supports = ttest_diff['p_value'] < 0.05 and ttest_diff['mean_a'] > ttest_diff['mean_b']
    control_supports = ttest_ctrl['p_value'] < 0.05 and ttest_ctrl['mean_a'] < ttest_ctrl['mean_b']
    
    if difficulty_supports and control_supports:
        print("  H3 is SUPPORTED - Clear learning effect detected")
    elif difficulty_supports or control_supports:
        print("  H3 is PARTIALLY SUPPORTED")
        if difficulty_supports:
            print("  - Difficulty decreased over trials (learning)")
        if ttest_ctrl['p_value'] < 0.05 and ttest_ctrl['mean_a'] > ttest_ctrl['mean_b']:
            print("  - Control feeling decreased over trials (fatigue/frustration)")
    else:
        print("  H3 is NOT SUPPORTED - No significant learning effect")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python H3_learning_analysis.py <xlsx_file>")
        print("\nExample:")
        print("  python H3_learning_analysis.py questionnaire_data-561422-2025-11-11-1622.xlsx")
        sys.exit(1)
    
    xlsx_file = sys.argv[1]
    main(xlsx_file)