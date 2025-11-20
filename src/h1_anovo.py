#!/usr/bin/env python3
"""
H1 Latency-Difficulty Analysis Script
======================================
Analyzes whether increased latency leads to higher perceived difficulty.

H1: Increased latency leads to higher perceived difficulty

Usage:
    python H1_latency_difficulty.py <combined_csv_file>
    
Example:
    python H1_latency_difficulty.py combined.csv

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
    df.columns = df.columns.str.replace('&#39;', "'")
    return df

def extract_bbt_data(df):
    """Extract BBT subjective ratings aligned by latency."""
    
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
    
    # Extract latency applied
    latency_cols = ['latency applied|0', 'latency applied|1', 'latency applied|2',
                    'latency applied|3', 'latency applied|4']
    
    # Get data
    difficulty_data = df[difficulty_cols].values
    control_data = df[control_cols].values
    delay_data = df[delay_cols].values
    latency_data = df[latency_cols].values
    
    # Create long-format dataframe
    rows = []
    for i in range(n_participants):
        for j in range(5):
            latency_str = str(latency_data[i, j])
            latency_val = int(latency_str.replace('ms', ''))
            
            rows.append({
                'Participant': participants[i],
                'Trial': j + 1,
                'Latency': latency_val,
                'Difficulty': difficulty_data[i, j],
                'Control': control_data[i, j],
                'Delay': delay_data[i, j]
            })
    
    long_df = pd.DataFrame(rows)
    return long_df

def calculate_oneway_anova(long_df, dv='Difficulty', iv='Latency'):
    """Calculate one-way ANOVA with full statistics."""
    
    groups = long_df.groupby(iv)[dv]
    group_names = sorted(long_df[iv].unique())
    k = len(group_names)
    
    group_means = groups.mean()
    group_sizes = groups.size()
    grand_mean = long_df[dv].mean()
    N = len(long_df)
    
    # Sum of squares
    SSA = sum(group_sizes[g] * (group_means[g] - grand_mean)**2 for g in group_names)
    SSE = 0
    for g in group_names:
        group_data = long_df[long_df[iv] == g][dv]
        group_mean = group_means[g]
        SSE += sum((y - group_mean)**2 for y in group_data)
    SST = SSA + SSE
    
    # Degrees of freedom
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    
    # Mean squares and F
    MSA = SSA / df_between
    MSE = SSE / df_within
    F = MSA / MSE
    p_value = 1 - stats.f.cdf(F, df_between, df_within)
    eta_squared = SSA / SST
    
    # Critical F values
    F_crit_95 = stats.f.ppf(0.95, df_between, df_within)
    F_crit_99 = stats.f.ppf(0.99, df_between, df_within)
    F_crit_999 = stats.f.ppf(0.999, df_between, df_within)
    
    return {
        'SSA': SSA, 'SSE': SSE, 'SST': SST,
        'df_between': df_between, 'df_within': df_within, 'df_total': df_total,
        'MSA': MSA, 'MSE': MSE, 'F': F, 'p_value': p_value,
        'eta_squared': eta_squared,
        'F_crit_95': F_crit_95, 'F_crit_99': F_crit_99, 'F_crit_999': F_crit_999,
        'k': k, 'N': N, 'group_means': group_means, 'grand_mean': grand_mean
    }

def print_anova_table(anova_results, dv_name, iv_name):
    """Print ANOVA results in formatted table."""
    
    print(f"\n{'='*70}")
    print(f"ONE-WAY ANOVA: {dv_name} by {iv_name}")
    print(f"{'='*70}")
    
    print(f"\nGroup Means (ȳ.j):")
    for g, mean in sorted(anova_results['group_means'].items()):
        print(f"  {iv_name} {g}ms: ȳ = {mean:.4f}")
    print(f"  Grand Mean (ȳ): {anova_results['grand_mean']:.4f}")
    
    print(f"\n{'-'*70}")
    print(f"{'Variation':<20} {'Sum of Squares':>15} {'df':>8} {'Mean Square':>15} {'F':>10}")
    print(f"{'-'*70}")
    print(f"{'Between (SSA)':<20} {anova_results['SSA']:>15.4f} {anova_results['df_between']:>8} {anova_results['MSA']:>15.4f} {anova_results['F']:>10.4f}")
    print(f"{'Within (SSE)':<20} {anova_results['SSE']:>15.4f} {anova_results['df_within']:>8} {anova_results['MSE']:>15.4f}")
    print(f"{'Total (SST)':<20} {anova_results['SST']:>15.4f} {anova_results['df_total']:>8}")
    print(f"{'-'*70}")
    
    print(f"\nF-test Results:")
    print(f"  Computed F = {anova_results['F']:.4f}")
    print(f"  p-value = {anova_results['p_value']:.6f}")
    print(f"  η² (effect size) = {anova_results['eta_squared']:.4f}")
    
    print(f"\nCritical F values (df1={anova_results['df_between']}, df2={anova_results['df_within']}):")
    print(f"  F[0.95] = {anova_results['F_crit_95']:.4f}")
    print(f"  F[0.99] = {anova_results['F_crit_99']:.4f}")
    print(f"  F[0.999] = {anova_results['F_crit_999']:.4f}")
    
    if anova_results['p_value'] < 0.001:
        sig = "*** (p < 0.001)"
    elif anova_results['p_value'] < 0.01:
        sig = "** (p < 0.01)"
    elif anova_results['p_value'] < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "ns (not significant)"
    print(f"\nSignificance: {sig}")

def save_anova_table(anova_results, output_path):
    """Save ANOVA results to CSV."""
    
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
    
    # Summary file
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

def tukey_hsd_posthoc(long_df, dv='Difficulty', iv='Latency'):
    """Perform Tukey HSD post-hoc test."""
    from scipy.stats import studentized_range
    
    groups = long_df.groupby(iv)[dv]
    group_names = sorted(long_df[iv].unique())
    k = len(group_names)
    
    group_means = groups.mean()
    group_sizes = groups.size()
    group_vars = groups.var()
    
    N = len(long_df)
    SSE = sum((group_sizes[g] - 1) * group_vars[g] for g in group_names)
    df_within = N - k
    MSE = SSE / df_within
    
    try:
        q_crit = studentized_range.ppf(0.95, k, df_within)
    except:
        q_crit = 3.98
    
    results = []
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i < j:
                mean_diff = group_means[g1] - group_means[g2]
                n1, n2 = group_sizes[g1], group_sizes[g2]
                se = np.sqrt(MSE * (1/n1 + 1/n2) / 2)
                q = abs(mean_diff) / se
                hsd = q_crit * se
                
                try:
                    p_value = 1 - studentized_range.cdf(q, k, df_within)
                except:
                    p_value = 0.05 if q > q_crit else 0.10
                
                significant = abs(mean_diff) > hsd
                
                results.append({
                    'Group1': g1, 'Group2': g2,
                    'Mean1': group_means[g1], 'Mean2': group_means[g2],
                    'Mean_Diff': mean_diff, 'SE': se, 'q': q,
                    'HSD': hsd, 'p_value': p_value, 'Significant': significant
                })
    
    return pd.DataFrame(results), q_crit

def print_posthoc_table(posthoc_df, dv_name, q_crit):
    """Print post-hoc results."""
    
    print(f"\n{'='*70}")
    print(f"TUKEY HSD POST-HOC: {dv_name}")
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
    
    sig_pairs = posthoc_df[posthoc_df['Significant']]
    if len(sig_pairs) > 0:
        print("\nSignificant pairwise differences (p < 0.05):")
        for _, row in sig_pairs.iterrows():
            print(f"  • {int(row['Group1'])}ms vs {int(row['Group2'])}ms: Δ = {row['Mean_Diff']:.3f}")
    else:
        print("\nNo significant pairwise differences found.")

def calculate_trend(long_df, dv='Difficulty'):
    """Calculate linear trend (correlation with latency)."""
    r, p = stats.pearsonr(long_df['Latency'], long_df[dv])
    slope, intercept, r_val, p_val, se = stats.linregress(long_df['Latency'], long_df[dv])
    return {'r': r, 'p': p, 'slope': slope, 'intercept': intercept, 'se': se}

def create_plots(long_df, output_dir='.'):
    """Create visualization plots for H1."""
    
    # Calculate means by latency
    latency_stats = long_df.groupby('Latency').agg({
        'Difficulty': ['mean', 'std', 'sem'],
        'Control': ['mean', 'std', 'sem'],
        'Delay': ['mean', 'std', 'sem']
    }).reset_index()
    latency_stats.columns = ['Latency', 'Diff_Mean', 'Diff_SD', 'Diff_SEM',
                              'Ctrl_Mean', 'Ctrl_SD', 'Ctrl_SEM',
                              'Delay_Mean', 'Delay_SD', 'Delay_SEM']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Difficulty by Latency
    ax1 = axes[0]
    ax1.errorbar(latency_stats['Latency'], latency_stats['Diff_Mean'],
                 yerr=1.96*latency_stats['Diff_SEM'],
                 fmt='o-', color='#E74C3C', linewidth=2, markersize=8, capsize=5)
    
    # Trend line
    trend = calculate_trend(long_df, 'Difficulty')
    x_line = np.array([0, 50, 100, 150, 200])
    ax1.plot(x_line, trend['slope'] * x_line + trend['intercept'], '--', color='gray',
             label=f"r = {trend['r']:.3f}")
    
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_ylabel('Perceived Difficulty (1-5)', fontsize=12)
    ax1.set_title('H1: Difficulty by Latency', fontsize=14)
    ax1.set_xticks([0, 50, 100, 150, 200])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control by Latency
    ax2 = axes[1]
    ax2.errorbar(latency_stats['Latency'], latency_stats['Ctrl_Mean'],
                 yerr=1.96*latency_stats['Ctrl_SEM'],
                 fmt='s-', color='#3498DB', linewidth=2, markersize=8, capsize=5)
    
    trend_ctrl = calculate_trend(long_df, 'Control')
    ax2.plot(x_line, trend_ctrl['slope'] * x_line + trend_ctrl['intercept'], '--', color='gray',
             label=f"r = {trend_ctrl['r']:.3f}")
    
    ax2.set_xlabel('Latency (ms)', fontsize=12)
    ax2.set_ylabel('Control Feeling (1-5)', fontsize=12)
    ax2.set_title('Control by Latency', fontsize=14)
    ax2.set_xticks([0, 50, 100, 150, 200])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Delay Perception by Latency
    ax3 = axes[2]
    ax3.errorbar(latency_stats['Latency'], latency_stats['Delay_Mean'],
                 yerr=1.96*latency_stats['Delay_SEM'],
                 fmt='^-', color='#27AE60', linewidth=2, markersize=8, capsize=5)
    
    trend_delay = calculate_trend(long_df, 'Delay')
    ax3.plot(x_line, trend_delay['slope'] * x_line + trend_delay['intercept'], '--', color='gray',
             label=f"r = {trend_delay['r']:.3f}")
    
    ax3.set_xlabel('Latency (ms)', fontsize=12)
    ax3.set_ylabel('Perceived Delay (1-5)', fontsize=12)
    ax3.set_title('Delay Perception by Latency', fontsize=14)
    ax3.set_xticks([0, 50, 100, 150, 200])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'H1_latency_effects.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    # Box plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    latencies = sorted(long_df['Latency'].unique())
    diff_data = [long_df[long_df['Latency'] == lat]['Difficulty'].values for lat in latencies]
    ctrl_data = [long_df[long_df['Latency'] == lat]['Control'].values for lat in latencies]
    
    ax4 = axes2[0]
    bp1 = ax4.boxplot(diff_data, labels=[f'{lat}ms' for lat in latencies], patch_artist=True)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, 5))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    ax4.set_xlabel('Latency', fontsize=12)
    ax4.set_ylabel('Perceived Difficulty (1-5)', fontsize=12)
    ax4.set_title('Difficulty Distribution by Latency', fontsize=12)
    
    ax5 = axes2[1]
    bp2 = ax5.boxplot(ctrl_data, labels=[f'{lat}ms' for lat in latencies], patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 5))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    ax5.set_xlabel('Latency', fontsize=12)
    ax5.set_ylabel('Control Feeling (1-5)', fontsize=12)
    ax5.set_title('Control Distribution by Latency', fontsize=12)
    
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'H1_boxplots.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {boxplot_path}")
    
    return latency_stats

def main(csv_path):
    """Main analysis function."""
    
    print("="*70)
    print("H1 LATENCY-DIFFICULTY ANALYSIS")
    print("Increased latency → Higher perceived difficulty")
    print("="*70)
    
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
    
    latency_stats = long_df.groupby('Latency').agg({
        'Difficulty': ['mean', 'std', 'count'],
        'Control': ['mean', 'std'],
        'Delay': ['mean', 'std']
    }).round(3)
    print(latency_stats.to_string())
    
    # Trend analysis
    print("\n3. LINEAR TREND ANALYSIS")
    print("-"*70)
    
    trend_diff = calculate_trend(long_df, 'Difficulty')
    print(f"\nDifficulty vs Latency:")
    print(f"  Pearson r = {trend_diff['r']:.4f}, p = {trend_diff['p']:.6f}")
    print(f"  Slope = {trend_diff['slope']:.6f} per ms")
    
    trend_ctrl = calculate_trend(long_df, 'Control')
    print(f"\nControl vs Latency:")
    print(f"  Pearson r = {trend_ctrl['r']:.4f}, p = {trend_ctrl['p']:.6f}")
    
    trend_delay = calculate_trend(long_df, 'Delay')
    print(f"\nDelay Perception vs Latency:")
    print(f"  Pearson r = {trend_delay['r']:.4f}, p = {trend_delay['p']:.6f}")
    
    # ANOVA
    print("\n4. ONE-WAY ANOVA")
    print("-"*70)
    
    anova_diff = calculate_oneway_anova(long_df, dv='Difficulty', iv='Latency')
    print_anova_table(anova_diff, 'Perceived Difficulty', 'Latency')
    
    # Post-hoc
    posthoc_diff, q_crit = tukey_hsd_posthoc(long_df, dv='Difficulty', iv='Latency')
    print_posthoc_table(posthoc_diff, 'Perceived Difficulty', q_crit)
    
    # Also run for Control and Delay
    anova_ctrl = calculate_oneway_anova(long_df, dv='Control', iv='Latency')
    print_anova_table(anova_ctrl, 'Control Feeling', 'Latency')
    
    posthoc_ctrl, q_crit_ctrl = tukey_hsd_posthoc(long_df, dv='Control', iv='Latency')
    print_posthoc_table(posthoc_ctrl, 'Control Feeling', q_crit_ctrl)
    
    anova_delay = calculate_oneway_anova(long_df, dv='Delay', iv='Latency')
    print_anova_table(anova_delay, 'Delay Perception', 'Latency')
    
    # Create plots
    print("\n5. CREATING PLOTS")
    print("-"*70)
    output_dir = 'out/h1out'
    os.makedirs(output_dir, exist_ok=True)
    latency_means = create_plots(long_df, output_dir)
    
    # Save results
    print("\n6. SAVING RESULTS")
    print("-"*70)
    
    # ANOVA tables
    save_anova_table(anova_diff, os.path.join(output_dir, 'H1_anova_difficulty.csv'))
    print(f"  Saved: {output_dir}/H1_anova_difficulty.csv")
    
    save_anova_table(anova_ctrl, os.path.join(output_dir, 'H1_anova_control.csv'))
    print(f"  Saved: {output_dir}/H1_anova_control.csv")
    
    save_anova_table(anova_delay, os.path.join(output_dir, 'H1_anova_delay.csv'))
    print(f"  Saved: {output_dir}/H1_anova_delay.csv")
    
    # Post-hoc
    posthoc_diff.to_csv(os.path.join(output_dir, 'H1_posthoc_difficulty.csv'), index=False)
    print(f"  Saved: {output_dir}/H1_posthoc_difficulty.csv")
    
    posthoc_ctrl.to_csv(os.path.join(output_dir, 'H1_posthoc_control.csv'), index=False)
    print(f"  Saved: {output_dir}/H1_posthoc_control.csv")
    
    # Trend results
    trend_table = pd.DataFrame({
        'Variable': ['Difficulty', 'Control', 'Delay'],
        'r': [trend_diff['r'], trend_ctrl['r'], trend_delay['r']],
        'p_value': [trend_diff['p'], trend_ctrl['p'], trend_delay['p']],
        'slope': [trend_diff['slope'], trend_ctrl['slope'], trend_delay['slope']],
        'significant': [trend_diff['p'] < 0.05, trend_ctrl['p'] < 0.05, trend_delay['p'] < 0.05]
    })
    trend_table.to_csv(os.path.join(output_dir, 'H1_trend_analysis.csv'), index=False)
    print(f"  Saved: {output_dir}/H1_trend_analysis.csv")
    
    # Latency means
    latency_means.to_csv(os.path.join(output_dir, 'H1_latency_means.csv'), index=False)
    print(f"  Saved: {output_dir}/H1_latency_means.csv")
    
    # Long format data
    long_df.to_csv(os.path.join(output_dir, 'H1_long_format_data.csv'), index=False)
    print(f"  Saved: {output_dir}/H1_long_format_data.csv")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: H1 LATENCY-DIFFICULTY ANALYSIS")
    print("="*70)
    
    print("\nKey Findings:")
    
    # Difficulty
    if anova_diff['p_value'] < 0.05:
        print(f"  ✓ ANOVA significant: F({anova_diff['df_between']},{anova_diff['df_within']}) = {anova_diff['F']:.2f}, p = {anova_diff['p_value']:.4f}")
        print(f"    Effect size: η² = {anova_diff['eta_squared']:.3f}")
    else:
        print(f"  ✗ ANOVA not significant: p = {anova_diff['p_value']:.4f}")
    
    if trend_diff['p'] < 0.05:
        direction = "increases" if trend_diff['r'] > 0 else "decreases"
        print(f"  ✓ Significant linear trend: r = {trend_diff['r']:.3f}, p = {trend_diff['p']:.4f}")
        print(f"    Difficulty {direction} with latency")
    else:
        print(f"  ✗ No significant linear trend: r = {trend_diff['r']:.3f}, p = {trend_diff['p']:.4f}")
    
    # Significant post-hoc
    sig_pairs = posthoc_diff[posthoc_diff['Significant']]
    if len(sig_pairs) > 0:
        print(f"\n  Significant pairwise differences:")
        for _, row in sig_pairs.iterrows():
            print(f"    • {int(row['Group1'])}ms vs {int(row['Group2'])}ms (p = {row['p_value']:.4f})")
    
    print("\n" + "="*70)
    print("H1 VERDICT:")
    print("="*70)
    
    # H1 is supported if difficulty increases with latency
    h1_supported = anova_diff['p_value'] < 0.05 and trend_diff['r'] > 0
    
    if h1_supported:
        print("  H1 is SUPPORTED")
        print("  - Latency significantly affects perceived difficulty")
        print("  - Higher latency leads to higher perceived difficulty")
    elif anova_diff['p_value'] < 0.05:
        print("  H1 is PARTIALLY SUPPORTED")
        print("  - Latency affects difficulty, but relationship may not be linear")
    else:
        print("  H1 is NOT SUPPORTED")
        print("  - No significant effect of latency on perceived difficulty")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python H1_latency_difficulty.py <combined_csv_file>")
        sys.exit(1)
    
    main(sys.argv[1])