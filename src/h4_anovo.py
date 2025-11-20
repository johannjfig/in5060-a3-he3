#!/usr/bin/env python3
"""
H4 Correlation Analysis Script
===============================
Analyzes whether subjective difficulty aligns with objective performance.

H4: Subjective difficulty aligns with objective performance

Usage:
    python H4_correlation_analysis.py <combined_csv_file>
    
Example:
    python H4_correlation_analysis.py combined.csv

Requirements:
    pip install pandas scipy numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

# Try to import seaborn for nicer plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def load_data(csv_path):
    """Load the combined CSV file with BBT data."""
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = df.columns.str.replace('&#39;', "'")
    
    return df

def extract_bbt_data(df):
    """Extract BBT performance and subjective ratings aligned by latency."""
    
    participants = df['Participant number'].values
    n_participants = len(participants)
    
    # Extract difficulty ratings for 5 BBT trials
    difficulty_cols = [
        'How difficult was it to perform the task?',
        'How difficult was it to perform the task?.1',
        'How difficult was it to perform the task?.2',
        'How difficult was it to perform the task?.3',
        'How difficult was it to perform the task?.4'
    ]
    
    # Extract control ratings
    control_cols = [
        'I felt like I was controlling the movement of the robot',
        'I felt like I was controlling the movement of the robot.1',
        'I felt like I was controlling the movement of the robot.2',
        'I felt like I was controlling the movement of the robot.3',
        'I felt like I was controlling the movement of the robot.4'
    ]
    
    # Extract delay perception
    delay_cols = [
        "Did you experience delays between your actions and the robot's movements?",
        "Did you experience delays between your actions and the robot's movements?.1",
        "Did you experience delays between your actions and the robot's movements?.2",
        "Did you experience delays between your actions and the robot's movements?.3",
        "Did you experience delays between your actions and the robot's movements?.4"
    ]
    
    # Extract blocks moved (objective performance)
    blocks_cols = ['moved blocks|0', 'moved blocks|1', 'moved blocks|2', 
                   'moved blocks|3', 'moved blocks|4']
    
    # Extract latency applied
    latency_cols = ['latency applied|0', 'latency applied|1', 'latency applied|2',
                    'latency applied|3', 'latency applied|4']
    
    # Get data
    difficulty_data = df[difficulty_cols].values
    control_data = df[control_cols].values
    delay_data = df[delay_cols].values
    blocks_data = df[blocks_cols].values
    latency_data = df[latency_cols].values
    
    # Create long-format dataframe
    rows = []
    for i in range(n_participants):
        for j in range(5):
            # Parse latency value (e.g., "150ms" -> 150)
            latency_str = str(latency_data[i, j])
            latency_val = int(latency_str.replace('ms', ''))
            
            rows.append({
                'Participant': participants[i],
                'Trial': j + 1,
                'Latency': latency_val,
                'Difficulty': difficulty_data[i, j],
                'Control': control_data[i, j],
                'Delay': delay_data[i, j],
                'Blocks': blocks_data[i, j]
            })
    
    long_df = pd.DataFrame(rows)
    
    return long_df

def run_correlation_analysis(long_df):
    """Run correlation analyses between subjective and objective measures."""
    
    results = {}
    
    # 1. Pearson correlation: Difficulty vs Blocks
    r_diff, p_diff = stats.pearsonr(long_df['Difficulty'], long_df['Blocks'])
    results['difficulty_blocks_pearson'] = {'r': r_diff, 'p': p_diff}
    
    # Spearman (for ordinal data)
    rho_diff, p_diff_s = stats.spearmanr(long_df['Difficulty'], long_df['Blocks'])
    results['difficulty_blocks_spearman'] = {'rho': rho_diff, 'p': p_diff_s}
    
    # 2. Control vs Blocks
    r_ctrl, p_ctrl = stats.pearsonr(long_df['Control'], long_df['Blocks'])
    results['control_blocks_pearson'] = {'r': r_ctrl, 'p': p_ctrl}
    
    rho_ctrl, p_ctrl_s = stats.spearmanr(long_df['Control'], long_df['Blocks'])
    results['control_blocks_spearman'] = {'rho': rho_ctrl, 'p': p_ctrl_s}
    
    # 3. Delay perception vs Blocks
    r_delay, p_delay = stats.pearsonr(long_df['Delay'], long_df['Blocks'])
    results['delay_blocks_pearson'] = {'r': r_delay, 'p': p_delay}
    
    rho_delay, p_delay_s = stats.spearmanr(long_df['Delay'], long_df['Blocks'])
    results['delay_blocks_spearman'] = {'rho': rho_delay, 'p': p_delay_s}
    
    # 4. Aggregate per participant (mean across trials)
    participant_means = long_df.groupby('Participant').agg({
        'Difficulty': 'mean',
        'Control': 'mean',
        'Delay': 'mean',
        'Blocks': 'mean'
    }).reset_index()
    
    r_agg, p_agg = stats.pearsonr(participant_means['Difficulty'], participant_means['Blocks'])
    results['difficulty_blocks_aggregated'] = {'r': r_agg, 'p': p_agg}
    
    return results, participant_means

def calculate_oneway_anova(long_df, dv='Blocks', iv='Latency'):
    """
    Calculate one-way ANOVA manually to get SSA, SSE, SST like in the lecture slides.
    
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
    # SSA = sum of n_j * (mean_j - grand_mean)^2
    SSA = sum(group_sizes[g] * (group_means[g] - grand_mean)**2 for g in group_names)
    
    # Calculate SSE (Sum of Squares Within/Error)
    # SSE = sum of (y_ij - mean_j)^2
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
    MSA = SSA / df_between  # Mean Square Between (Alternatives)
    MSE = SSE / df_within   # Mean Square Within (Error)
    
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
    """Print ANOVA results in a formatted table like the lecture slides."""
    
    print(f"\n{'='*70}")
    print(f"ONE-WAY ANOVA: {dv_name} by {iv_name}")
    print(f"{'='*70}")
    
    # Print group means
    print(f"\nGroup Means (ȳ.j):")
    for group, mean in sorted(anova_results['group_means'].items()):
        print(f"  {iv_name} {group}: ȳ = {mean:.4f}")
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
    """Save ANOVA results to a CSV file."""
    
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

def tukey_hsd_posthoc(long_df, dv='Blocks', iv='Latency'):
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
    # q_crit = studentized_range.ppf(0.95, k, df_within)
    # Using scipy's distribution
    try:
        q_crit = studentized_range.ppf(0.95, k, df_within)
    except:
        # Fallback approximation if studentized_range not available
        q_crit = 3.98  # Approximate for k=5, df=135
    
    # Calculate all pairwise comparisons
    results = []
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i < j:  # Only upper triangle
                mean_diff = group_means[g1] - group_means[g2]
                n1, n2 = group_sizes[g1], group_sizes[g2]
                
                # Standard error for unequal sample sizes
                se = np.sqrt(MSE * (1/n1 + 1/n2) / 2)
                
                # q statistic
                q = abs(mean_diff) / se
                
                # HSD (minimum significant difference)
                hsd = q_crit * se
                
                # p-value (approximate using studentized range distribution)
                try:
                    p_value = 1 - studentized_range.cdf(q, k, df_within)
                except:
                    # Rough approximation
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

def print_posthoc_table(posthoc_df, dv_name, iv_name, q_crit):
    """Print post-hoc results in a formatted table."""
    
    print(f"\n{'='*70}")
    print(f"TUKEY HSD POST-HOC: {dv_name} by {iv_name}")
    print(f"{'='*70}")
    print(f"Critical q value (α=0.05): {q_crit:.4f}")
    
    print(f"\n{'-'*90}")
    print(f"{'Comparison':<20} {'Mean Diff':>12} {'SE':>10} {'q':>10} {'HSD':>10} {'p':>10} {'Sig':>8}")
    print(f"{'-'*90}")
    
    for _, row in posthoc_df.iterrows():
        comp = f"{int(row['Group1'])}ms vs {int(row['Group2'])}ms"
        sig = "*" if row['Significant'] else ""
        print(f"{comp:<20} {row['Mean_Diff']:>12.4f} {row['SE']:>10.4f} {row['q']:>10.4f} {row['HSD']:>10.4f} {row['p_value']:>10.4f} {sig:>8}")
    
    print(f"{'-'*90}")
    
    # Highlight significant differences
    sig_pairs = posthoc_df[posthoc_df['Significant']]
    if len(sig_pairs) > 0:
        print("\nSignificant pairwise differences (p < 0.05):")
        for _, row in sig_pairs.iterrows():
            print(f"  • {int(row['Group1'])}ms vs {int(row['Group2'])}ms: Δ = {row['Mean_Diff']:.3f}")
    else:
        print("\nNo significant pairwise differences found.")
    
    return

def create_correlation_plots(long_df, participant_means, output_dir='.'):
    """Create scatter plots showing correlations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Difficulty vs Blocks (all trials)
    ax1 = axes[0, 0]
    if HAS_SEABORN:
        sns.regplot(x='Difficulty', y='Blocks', data=long_df, ax=ax1, 
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    else:
        ax1.scatter(long_df['Difficulty'], long_df['Blocks'], alpha=0.5)
        # Add regression line
        z = np.polyfit(long_df['Difficulty'], long_df['Blocks'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(long_df['Difficulty'].min(), long_df['Difficulty'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    r, p_val = stats.pearsonr(long_df['Difficulty'], long_df['Blocks'])
    ax1.set_xlabel('Perceived Difficulty (1-5)', fontsize=12)
    ax1.set_ylabel('Blocks Moved', fontsize=12)
    ax1.set_title(f'H4: Difficulty vs Performance\n(r = {r:.3f}, p = {p_val:.4f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control vs Blocks
    ax2 = axes[0, 1]
    if HAS_SEABORN:
        sns.regplot(x='Control', y='Blocks', data=long_df, ax=ax2,
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'blue'})
    else:
        ax2.scatter(long_df['Control'], long_df['Blocks'], alpha=0.5, color='blue')
        z = np.polyfit(long_df['Control'], long_df['Blocks'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(long_df['Control'].min(), long_df['Control'].max(), 100)
        ax2.plot(x_line, p(x_line), 'b-', linewidth=2)
    
    r_ctrl, p_ctrl = stats.pearsonr(long_df['Control'], long_df['Blocks'])
    ax2.set_xlabel('Control Feeling (1-5)', fontsize=12)
    ax2.set_ylabel('Blocks Moved', fontsize=12)
    ax2.set_title(f'Control vs Performance\n(r = {r_ctrl:.3f}, p = {p_ctrl:.4f})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Delay perception vs Blocks
    ax3 = axes[1, 0]
    if HAS_SEABORN:
        sns.regplot(x='Delay', y='Blocks', data=long_df, ax=ax3,
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'green'})
    else:
        ax3.scatter(long_df['Delay'], long_df['Blocks'], alpha=0.5, color='green')
        z = np.polyfit(long_df['Delay'], long_df['Blocks'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(long_df['Delay'].min(), long_df['Delay'].max(), 100)
        ax3.plot(x_line, p(x_line), 'g-', linewidth=2)
    
    r_delay, p_delay = stats.pearsonr(long_df['Delay'], long_df['Blocks'])
    ax3.set_xlabel('Perceived Delay (1-5)', fontsize=12)
    ax3.set_ylabel('Blocks Moved', fontsize=12)
    ax3.set_title(f'Delay Perception vs Performance\n(r = {r_delay:.3f}, p = {p_delay:.4f})', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Aggregated per participant
    ax4 = axes[1, 1]
    if HAS_SEABORN:
        sns.regplot(x='Difficulty', y='Blocks', data=participant_means, ax=ax4,
                    scatter_kws={'s': 80}, line_kws={'color': 'purple'})
    else:
        ax4.scatter(participant_means['Difficulty'], participant_means['Blocks'], s=80, color='purple')
        z = np.polyfit(participant_means['Difficulty'], participant_means['Blocks'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(participant_means['Difficulty'].min(), participant_means['Difficulty'].max(), 100)
        ax4.plot(x_line, p(x_line), 'purple', linewidth=2)
    
    r_agg, p_agg = stats.pearsonr(participant_means['Difficulty'], participant_means['Blocks'])
    ax4.set_xlabel('Mean Difficulty (1-5)', fontsize=12)
    ax4.set_ylabel('Mean Blocks Moved', fontsize=12)
    ax4.set_title(f'Aggregated by Participant\n(r = {r_agg:.3f}, p = {p_agg:.4f})', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'H4_correlation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    # Additional plot: By latency condition
    fig2, ax5 = plt.subplots(figsize=(10, 6))
    
    latency_means = long_df.groupby('Latency').agg({
        'Difficulty': ['mean', 'sem'],
        'Blocks': ['mean', 'sem']
    }).reset_index()
    latency_means.columns = ['Latency', 'Diff_Mean', 'Diff_SEM', 'Blocks_Mean', 'Blocks_SEM']
    latency_means = latency_means.sort_values('Latency')
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.errorbar(latency_means['Latency'], latency_means['Diff_Mean'], 
                         yerr=1.96*latency_means['Diff_SEM'], 
                         color='#E74C3C', marker='o', linewidth=2, capsize=5, label='Difficulty')
    line2 = ax5_twin.errorbar(latency_means['Latency'], latency_means['Blocks_Mean'],
                               yerr=1.96*latency_means['Blocks_SEM'],
                               color='#3498DB', marker='s', linewidth=2, capsize=5, label='Blocks')
    
    ax5.set_xlabel('Latency (ms)', fontsize=12)
    ax5.set_ylabel('Perceived Difficulty (1-5)', fontsize=12, color='#E74C3C')
    ax5_twin.set_ylabel('Blocks Moved', fontsize=12, color='#3498DB')
    ax5.set_title('Difficulty and Performance by Latency Condition', fontsize=14)
    
    # Set x-axis ticks to actual latency values
    ax5.set_xticks([0, 50, 100, 150, 200])
    
    # Combine legends
    lines = [line1, line2]
    labels = ['Difficulty', 'Blocks Moved']
    ax5.legend(lines, labels, loc='center right')
    
    ax5.tick_params(axis='y', labelcolor='#E74C3C')
    ax5_twin.tick_params(axis='y', labelcolor='#3498DB')
    
    plt.tight_layout()
    latency_plot_path = os.path.join(output_dir, 'H4_latency_comparison.png')
    plt.savefig(latency_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {latency_plot_path}")
    
    return latency_means

def interpret_correlation(r):
    """Interpret correlation strength."""
    r_abs = abs(r)
    if r_abs < 0.1:
        return "negligible"
    elif r_abs < 0.3:
        return "weak"
    elif r_abs < 0.5:
        return "moderate"
    elif r_abs < 0.7:
        return "strong"
    else:
        return "very strong"

def main(csv_path):
    """Main analysis function."""
    
    print("="*70)
    print("H4 CORRELATION ANALYSIS")
    print("Subjective Difficulty vs Objective Performance")
    print("="*70)
    
    # Check file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Load data
    print("\n1. LOADING DATA")
    print("-"*70)
    df = load_data(csv_path)
    long_df = extract_bbt_data(df)
    print(f"Loaded {len(df)} participants")
    print(f"Total observations: {len(long_df)}")
    
    # Descriptive statistics
    print("\n2. DESCRIPTIVE STATISTICS")
    print("-"*70)
    
    print("\nBy Latency Condition:")
    latency_stats = long_df.groupby('Latency').agg({
        'Difficulty': ['mean', 'std'],
        'Control': ['mean', 'std'],
        'Blocks': ['mean', 'std']
    }).round(2)
    print(latency_stats.to_string())
    
    # Run correlations
    print("\n3. CORRELATION ANALYSIS")
    print("-"*70)
    
    results, participant_means = run_correlation_analysis(long_df)
    
    print("\nDifficulty vs Blocks Moved:")
    r = results['difficulty_blocks_pearson']['r']
    p = results['difficulty_blocks_pearson']['p']
    strength = interpret_correlation(r)
    print(f"  Pearson r = {r:.3f}, p = {p:.4f} ({strength})")
    
    rho = results['difficulty_blocks_spearman']['rho']
    p_s = results['difficulty_blocks_spearman']['p']
    print(f"  Spearman ρ = {rho:.3f}, p = {p_s:.4f}")
    
    print("\nControl vs Blocks Moved:")
    r_ctrl = results['control_blocks_pearson']['r']
    p_ctrl = results['control_blocks_pearson']['p']
    strength_ctrl = interpret_correlation(r_ctrl)
    print(f"  Pearson r = {r_ctrl:.3f}, p = {p_ctrl:.4f} ({strength_ctrl})")
    
    print("\nDelay Perception vs Blocks Moved:")
    r_delay = results['delay_blocks_pearson']['r']
    p_delay = results['delay_blocks_pearson']['p']
    strength_delay = interpret_correlation(r_delay)
    print(f"  Pearson r = {r_delay:.3f}, p = {p_delay:.4f} ({strength_delay})")
    
    print("\nAggregated by Participant (Mean Difficulty vs Mean Blocks):")
    r_agg = results['difficulty_blocks_aggregated']['r']
    p_agg = results['difficulty_blocks_aggregated']['p']
    strength_agg = interpret_correlation(r_agg)
    print(f"  Pearson r = {r_agg:.3f}, p = {p_agg:.4f} ({strength_agg})")
    
    # Run one-way ANOVA: Blocks by Latency
    print("\n" + "="*70)
    print("4. ONE-WAY ANOVA ANALYSIS")
    print("="*70)
    
    # ANOVA for Blocks by Latency
    anova_blocks = calculate_oneway_anova(long_df, dv='Blocks', iv='Latency')
    print_anova_table(anova_blocks, 'Blocks Moved', 'Latency')
    
    # Post-hoc for Blocks
    posthoc_blocks, q_crit_blocks, _ = tukey_hsd_posthoc(long_df, dv='Blocks', iv='Latency')
    print_posthoc_table(posthoc_blocks, 'Blocks Moved', 'Latency', q_crit_blocks)
    
    # ANOVA for Difficulty by Latency
    anova_diff = calculate_oneway_anova(long_df, dv='Difficulty', iv='Latency')
    print_anova_table(anova_diff, 'Perceived Difficulty', 'Latency')
    
    # Post-hoc for Difficulty
    posthoc_diff, q_crit_diff, _ = tukey_hsd_posthoc(long_df, dv='Difficulty', iv='Latency')
    print_posthoc_table(posthoc_diff, 'Perceived Difficulty', 'Latency', q_crit_diff)
    
    # ANOVA for Control by Latency
    anova_ctrl = calculate_oneway_anova(long_df, dv='Control', iv='Latency')
    print_anova_table(anova_ctrl, 'Control Feeling', 'Latency')
    
    # Create plots
    print("\n5. CREATING PLOTS")
    print("-"*70)
    output_dir = 'out/h4out'
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    latency_means = create_correlation_plots(long_df, participant_means, output_dir)
    
    # Save results
    print("\n6. SAVING RESULTS")
    print("-"*70)
    
    # Save correlation results table
    corr_table = pd.DataFrame({
        'Comparison': [
            'Difficulty vs Blocks (all trials)',
            'Difficulty vs Blocks (Spearman)',
            'Control vs Blocks (all trials)',
            'Delay vs Blocks (all trials)',
            'Difficulty vs Blocks (aggregated)'
        ],
        'Statistic': ['r', 'ρ', 'r', 'r', 'r'],
        'Value': [
            results['difficulty_blocks_pearson']['r'],
            results['difficulty_blocks_spearman']['rho'],
            results['control_blocks_pearson']['r'],
            results['delay_blocks_pearson']['r'],
            results['difficulty_blocks_aggregated']['r']
        ],
        'p_value': [
            results['difficulty_blocks_pearson']['p'],
            results['difficulty_blocks_spearman']['p'],
            results['control_blocks_pearson']['p'],
            results['delay_blocks_pearson']['p'],
            results['difficulty_blocks_aggregated']['p']
        ],
        'Strength': [
            interpret_correlation(results['difficulty_blocks_pearson']['r']),
            interpret_correlation(results['difficulty_blocks_spearman']['rho']),
            interpret_correlation(results['control_blocks_pearson']['r']),
            interpret_correlation(results['delay_blocks_pearson']['r']),
            interpret_correlation(results['difficulty_blocks_aggregated']['r'])
        ],
        'Significant': [
            results['difficulty_blocks_pearson']['p'] < 0.05,
            results['difficulty_blocks_spearman']['p'] < 0.05,
            results['control_blocks_pearson']['p'] < 0.05,
            results['delay_blocks_pearson']['p'] < 0.05,
            results['difficulty_blocks_aggregated']['p'] < 0.05
        ]
    })
    
    corr_path = os.path.join(output_dir, 'H4_correlation_results.csv')
    corr_table.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path}")
    
    # Save ANOVA tables
    anova_blocks_path = os.path.join(output_dir, 'H4_anova_blocks.csv')
    save_anova_table(anova_blocks, 'Blocks', 'Latency', anova_blocks_path)
    print(f"  Saved: {anova_blocks_path}")
    
    anova_diff_path = os.path.join(output_dir, 'H4_anova_difficulty.csv')
    save_anova_table(anova_diff, 'Difficulty', 'Latency', anova_diff_path)
    print(f"  Saved: {anova_diff_path}")
    
    anova_ctrl_path = os.path.join(output_dir, 'H4_anova_control.csv')
    save_anova_table(anova_ctrl, 'Control', 'Latency', anova_ctrl_path)
    print(f"  Saved: {anova_ctrl_path}")
    
    # Save post-hoc results
    posthoc_blocks_path = os.path.join(output_dir, 'H4_posthoc_blocks.csv')
    posthoc_blocks.to_csv(posthoc_blocks_path, index=False)
    print(f"  Saved: {posthoc_blocks_path}")
    
    posthoc_diff_path = os.path.join(output_dir, 'H4_posthoc_difficulty.csv')
    posthoc_diff.to_csv(posthoc_diff_path, index=False)
    print(f"  Saved: {posthoc_diff_path}")
    
    # Save latency means
    latency_path = os.path.join(output_dir, 'H4_latency_means.csv')
    latency_means.to_csv(latency_path, index=False)
    print(f"  Saved: {latency_path}")
    
    # Save long-format data
    long_path = os.path.join(output_dir, 'H4_long_format_data.csv')
    long_df.to_csv(long_path, index=False)
    print(f"  Saved: {long_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: H4 CORRELATION ANALYSIS")
    print("="*70)
    
    print("\nKey Findings:")
    
    # Difficulty vs Performance
    r_main = results['difficulty_blocks_pearson']['r']
    p_main = results['difficulty_blocks_pearson']['p']
    if p_main < 0.05:
        if r_main < 0:
            print(f"  ✓ Difficulty negatively correlates with performance (r={r_main:.3f}, p={p_main:.4f})")
            print("    → Higher difficulty = fewer blocks (expected relationship)")
        else:
            print(f"  ✗ Difficulty positively correlates with performance (r={r_main:.3f}, p={p_main:.4f})")
            print("    → Unexpected: Higher difficulty = more blocks")
    else:
        print(f"  ✗ No significant correlation between difficulty and performance (r={r_main:.3f}, p={p_main:.4f})")
    
    # Control vs Performance
    r_ctrl = results['control_blocks_pearson']['r']
    p_ctrl = results['control_blocks_pearson']['p']
    if p_ctrl < 0.05:
        if r_ctrl > 0:
            print(f"  ✓ Control positively correlates with performance (r={r_ctrl:.3f}, p={p_ctrl:.4f})")
            print("    → Higher control feeling = more blocks (expected)")
        else:
            print(f"  ✗ Control negatively correlates with performance (r={r_ctrl:.3f}, p={p_ctrl:.4f})")
    else:
        print(f"  - No significant correlation between control and performance (r={r_ctrl:.3f}, p={p_ctrl:.4f})")
    
    print("\n" + "="*70)
    print("H4 VERDICT:")
    print("="*70)
    
    # Determine verdict
    # For H4 to be supported: difficulty should negatively correlate with performance
    # (higher difficulty = worse performance = fewer blocks)
    
    diff_supports = p_main < 0.05 and r_main < 0
    ctrl_supports = p_ctrl < 0.05 and r_ctrl > 0
    
    if diff_supports and ctrl_supports:
        print("  H4 is STRONGLY SUPPORTED")
        print("  - Perceived difficulty aligns with objective performance")
        print("  - Control feeling also aligns with performance")
    elif diff_supports or ctrl_supports:
        print("  H4 is PARTIALLY SUPPORTED")
        if diff_supports:
            print("  - Perceived difficulty aligns with objective performance")
        if ctrl_supports:
            print("  - Control feeling aligns with objective performance")
    else:
        print("  H4 is NOT SUPPORTED")
        print("  - No significant alignment between subjective ratings and objective performance")
        print("  - Participants' self-assessments do not reflect actual performance")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python H4_correlation_analysis.py <combined_csv_file>")
        print("\nExample:")
        print("  python H4_correlation_analysis.py combined.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    main(csv_file)