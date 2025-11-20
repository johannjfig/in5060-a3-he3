#!/usr/bin/env python3
"""
H2 Latency-Performance Analysis Script
=======================================
Analyzes whether increased latency leads to worse objective performance.

H2: Increased latency leads to worse objective performance (fewer blocks moved)

Usage:
    python H2_latency_performance.py <combined_csv_file>
    
Example:
    python H2_latency_performance.py combined.csv

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
    """Extract BBT performance data aligned by latency."""
    
    participants = df['Participant number'].values
    n_participants = len(participants)
    
    # Extract blocks moved (objective performance)
    blocks_cols = ['moved blocks|0', 'moved blocks|1', 'moved blocks|2',
                   'moved blocks|3', 'moved blocks|4']
    
    # Extract latency applied
    latency_cols = ['latency applied|0', 'latency applied|1', 'latency applied|2',
                    'latency applied|3', 'latency applied|4']
    
    # Get data
    blocks_data = df[blocks_cols].values
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
                'Blocks': blocks_data[i, j]
            })
    
    long_df = pd.DataFrame(rows)
    return long_df

def calculate_oneway_anova(long_df, dv='Blocks', iv='Latency'):
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

def tukey_hsd_posthoc(long_df, dv='Blocks', iv='Latency'):
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
            print(f"  • {int(row['Group1'])}ms vs {int(row['Group2'])}ms: Δ = {row['Mean_Diff']:.3f} blocks")
    else:
        print("\nNo significant pairwise differences found.")

def calculate_trend(long_df):
    """Calculate linear trend (correlation with latency)."""
    r, p = stats.pearsonr(long_df['Latency'], long_df['Blocks'])
    slope, intercept, r_val, p_val, se = stats.linregress(long_df['Latency'], long_df['Blocks'])
    return {'r': r, 'p': p, 'slope': slope, 'intercept': intercept, 'se': se}

def create_plots(long_df, output_dir='.'):
    """Create visualization plots for H2."""
    
    # Calculate means by latency
    latency_stats = long_df.groupby('Latency').agg({
        'Blocks': ['mean', 'std', 'sem', 'count']
    }).reset_index()
    latency_stats.columns = ['Latency', 'Blocks_Mean', 'Blocks_SD', 'Blocks_SEM', 'N']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Line plot with error bars
    ax1 = axes[0]
    ax1.errorbar(latency_stats['Latency'], latency_stats['Blocks_Mean'],
                 yerr=1.96*latency_stats['Blocks_SEM'],
                 fmt='o-', color='#3498DB', linewidth=2, markersize=10, capsize=5)
    
    # Trend line
    trend = calculate_trend(long_df)
    x_line = np.array([0, 50, 100, 150, 200])
    ax1.plot(x_line, trend['slope'] * x_line + trend['intercept'], '--', color='red',
             linewidth=2, label=f"r = {trend['r']:.3f}, p = {trend['p']:.4f}")
    
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_ylabel('Blocks Moved', fontsize=12)
    ax1.set_title('H2: Objective Performance by Latency\n(95% CI)', fontsize=14)
    ax1.set_xticks([0, 50, 100, 150, 200])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot
    ax2 = axes[1]
    latencies = sorted(long_df['Latency'].unique())
    blocks_data = [long_df[long_df['Latency'] == lat]['Blocks'].values for lat in latencies]
    
    bp = ax2.boxplot(blocks_data, labels=[f'{lat}ms' for lat in latencies], patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('Latency', fontsize=12)
    ax2.set_ylabel('Blocks Moved', fontsize=12)
    ax2.set_title('Performance Distribution by Latency', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'H2_performance_by_latency.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    # Individual participant plot
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    
    # Plot each participant's data
    participants = long_df['Participant'].unique()
    for p in participants:
        p_data = long_df[long_df['Participant'] == p].sort_values('Latency')
        ax3.plot(p_data['Latency'], p_data['Blocks'], 'o-', alpha=0.3, linewidth=1)
    
    # Overlay mean
    ax3.errorbar(latency_stats['Latency'], latency_stats['Blocks_Mean'],
                 yerr=latency_stats['Blocks_SEM'],
                 fmt='s-', color='red', linewidth=3, markersize=10, capsize=5,
                 label='Mean ± SEM')
    
    ax3.set_xlabel('Latency (ms)', fontsize=12)
    ax3.set_ylabel('Blocks Moved', fontsize=12)
    ax3.set_title('Individual Performance Trajectories', fontsize=14)
    ax3.set_xticks([0, 50, 100, 150, 200])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    individual_path = os.path.join(output_dir, 'H2_individual_trajectories.png')
    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {individual_path}")
    
    # Bar chart with individual points
    fig3, ax4 = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(latencies))
    bars = ax4.bar(x_pos, latency_stats['Blocks_Mean'], 
                   yerr=1.96*latency_stats['Blocks_SEM'],
                   capsize=5, color=plt.cm.Blues(0.6), edgecolor='black')
    
    # Add individual points
    for i, lat in enumerate(latencies):
        jitter = np.random.normal(0, 0.1, size=len(blocks_data[i]))
        ax4.scatter(x_pos[i] + jitter, blocks_data[i], alpha=0.5, color='darkblue', s=30)
    
    ax4.set_xlabel('Latency (ms)', fontsize=12)
    ax4.set_ylabel('Blocks Moved', fontsize=12)
    ax4.set_title('Mean Performance by Latency (with individual data)', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{lat}ms' for lat in latencies])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'H2_bar_chart.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {bar_path}")
    
    return latency_stats

def main(csv_path):
    """Main analysis function."""
    
    print("="*70)
    print("H2 LATENCY-PERFORMANCE ANALYSIS")
    print("Increased latency → Worse objective performance")
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
        'Blocks': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    latency_stats.columns = ['Mean', 'SD', 'Min', 'Max', 'N']
    print("\nBlocks Moved by Latency:")
    print(latency_stats.to_string())
    
    # Overall statistics
    print(f"\nOverall: M = {long_df['Blocks'].mean():.2f}, SD = {long_df['Blocks'].std():.2f}")
    print(f"Range: {long_df['Blocks'].min()} - {long_df['Blocks'].max()}")
    
    # Trend analysis
    print("\n3. LINEAR TREND ANALYSIS")
    print("-"*70)
    
    trend = calculate_trend(long_df)
    print(f"\nBlocks vs Latency:")
    print(f"  Pearson r = {trend['r']:.4f}, p = {trend['p']:.6f}")
    print(f"  Slope = {trend['slope']:.4f} blocks per ms")
    print(f"  Interpretation: {abs(trend['slope']*100):.2f} fewer blocks per 100ms increase")
    
    if trend['r'] < 0 and trend['p'] < 0.05:
        print("  → Significant negative relationship (as expected)")
    elif trend['r'] < 0:
        print("  → Negative trend but not significant")
    else:
        print("  → Unexpected: positive or no relationship")
    
    # ANOVA
    print("\n4. ONE-WAY ANOVA")
    print("-"*70)
    
    anova = calculate_oneway_anova(long_df, dv='Blocks', iv='Latency')
    print_anova_table(anova, 'Blocks Moved', 'Latency')
    
    # Post-hoc
    posthoc, q_crit = tukey_hsd_posthoc(long_df, dv='Blocks', iv='Latency')
    print_posthoc_table(posthoc, 'Blocks Moved', q_crit)
    
    # Create plots
    print("\n5. CREATING PLOTS")
    print("-"*70)
    output_dir = 'out/h2out'
    os.makedirs(output_dir, exist_ok=True)
    latency_means = create_plots(long_df, output_dir)
    
    # Save results
    print("\n6. SAVING RESULTS")
    print("-"*70)
    
    # ANOVA table
    save_anova_table(anova, os.path.join(output_dir, 'H2_anova_blocks.csv'))
    print(f"  Saved: {output_dir}/H2_anova_blocks.csv")
    
    # Post-hoc
    posthoc.to_csv(os.path.join(output_dir, 'H2_posthoc_blocks.csv'), index=False)
    print(f"  Saved: {output_dir}/H2_posthoc_blocks.csv")
    
    # Trend results
    trend_table = pd.DataFrame({
        'Variable': ['Blocks'],
        'r': [trend['r']],
        'p_value': [trend['p']],
        'slope': [trend['slope']],
        'intercept': [trend['intercept']],
        'blocks_per_100ms': [trend['slope'] * 100],
        'significant': [trend['p'] < 0.05]
    })
    trend_table.to_csv(os.path.join(output_dir, 'H2_trend_analysis.csv'), index=False)
    print(f"  Saved: {output_dir}/H2_trend_analysis.csv")
    
    # Latency means
    latency_means.to_csv(os.path.join(output_dir, 'H2_latency_means.csv'), index=False)
    print(f"  Saved: {output_dir}/H2_latency_means.csv")
    
    # Long format data
    long_df.to_csv(os.path.join(output_dir, 'H2_long_format_data.csv'), index=False)
    print(f"  Saved: {output_dir}/H2_long_format_data.csv")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: H2 LATENCY-PERFORMANCE ANALYSIS")
    print("="*70)
    
    print("\nKey Findings:")
    
    # ANOVA result
    if anova['p_value'] < 0.05:
        print(f"  ✓ ANOVA significant: F({anova['df_between']},{anova['df_within']}) = {anova['F']:.2f}, p = {anova['p_value']:.4f}")
        print(f"    Effect size: η² = {anova['eta_squared']:.3f} ({interpret_eta(anova['eta_squared'])})")
    else:
        print(f"  ✗ ANOVA not significant: p = {anova['p_value']:.4f}")
    
    # Trend result
    if trend['p'] < 0.05:
        print(f"  ✓ Significant linear trend: r = {trend['r']:.3f}, p = {trend['p']:.4f}")
        print(f"    Performance decreases {abs(trend['slope']*100):.2f} blocks per 100ms")
    else:
        print(f"  ✗ No significant linear trend: r = {trend['r']:.3f}, p = {trend['p']:.4f}")
    
    # Performance drop
    mean_0 = anova['group_means'][0]
    mean_200 = anova['group_means'][200]
    drop = mean_0 - mean_200
    drop_pct = (drop / mean_0) * 100
    print(f"\n  Performance drop (0ms → 200ms):")
    print(f"    {mean_0:.2f} → {mean_200:.2f} blocks")
    print(f"    Absolute: {drop:.2f} blocks")
    print(f"    Relative: {drop_pct:.1f}% decrease")
    
    # Significant post-hoc
    sig_pairs = posthoc[posthoc['Significant']]
    if len(sig_pairs) > 0:
        print(f"\n  Significant pairwise differences:")
        for _, row in sig_pairs.iterrows():
            print(f"    • {int(row['Group1'])}ms vs {int(row['Group2'])}ms: Δ = {row['Mean_Diff']:.2f} blocks")
    
    print("\n" + "="*70)
    print("H2 VERDICT:")
    print("="*70)
    
    # H2 is supported if performance decreases with latency
    h2_supported = anova['p_value'] < 0.05 and trend['r'] < 0
    
    if h2_supported:
        print("  H2 is SUPPORTED")
        print("  - Latency significantly affects objective performance")
        print("  - Higher latency leads to fewer blocks moved")
        print(f"  - {drop_pct:.1f}% performance decrease from 0ms to 200ms")
    elif anova['p_value'] < 0.05:
        print("  H2 is PARTIALLY SUPPORTED")
        print("  - Latency affects performance, but relationship may not be linear")
    else:
        print("  H2 is NOT SUPPORTED")
        print("  - No significant effect of latency on objective performance")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

def interpret_eta(eta):
    """Interpret eta-squared effect size."""
    if eta < 0.01:
        return "negligible"
    elif eta < 0.06:
        return "small"
    elif eta < 0.14:
        return "medium"
    else:
        return "large"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python H2_latency_performance.py <combined_csv_file>")
        sys.exit(1)
    
    main(sys.argv[1])