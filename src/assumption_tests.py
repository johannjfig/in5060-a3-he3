#!/usr/bin/env python3
"""
ANOVA Assumption Tests Module
==============================
Contains functions for testing ANOVA assumptions:
- Preprocessing checks (missing values)
- Levene's test (homogeneity of variance)
- Shapiro-Wilk test (normality)

Import this in your H1, H2, H3, H4 scripts.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def create_assumption_plots(long_df, dv, iv, output_dir, prefix, dv_name=None, iv_name=None):
    """
    Create diagnostic plots for ANOVA assumptions.
    
    Generates:
        - Histogram of residuals with normal curve
        - Q-Q plot of residuals
        - Box plots by group (for homogeneity check)
        - Histogram per group
    
    Parameters:
        long_df: DataFrame in long format
        dv: Dependent variable column name
        iv: Independent variable (grouping) column name
        output_dir: Directory to save plots
        prefix: Filename prefix (e.g., 'H1_difficulty')
        dv_name: Display name for DV (optional)
        iv_name: Display name for IV (optional)
    """
    import matplotlib.pyplot as plt
    
    if dv_name is None:
        dv_name = dv
    if iv_name is None:
        iv_name = iv
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate residuals
    group_means = long_df.groupby(iv)[dv].transform('mean')
    residuals = long_df[dv] - group_means
    
    # Get groups
    groups = sorted(long_df[iv].unique())
    n_groups = len(groups)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histogram of residuals with normal curve
    ax1 = fig.add_subplot(2, 3, 1)
    n, bins, patches = ax1.hist(residuals, bins=20, density=True, alpha=0.7, 
                                 color='steelblue', edgecolor='black')
    
    # Overlay normal distribution
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal')
    
    ax1.set_xlabel('Residuals', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title(f'Histogram of Residuals\n{dv_name}', fontsize=12)
    ax1.legend()
    
    # Add Shapiro-Wilk result
    sw_stat, sw_p = stats.shapiro(residuals)
    ax1.text(0.05, 0.95, f'Shapiro-Wilk:\nW = {sw_stat:.3f}\np = {sw_p:.4f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Q-Q Plot
    ax2 = fig.add_subplot(2, 3, 2)
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot of Residuals\n{dv_name}', fontsize=12)
    ax2.get_lines()[0].set_markerfacecolor('steelblue')
    ax2.get_lines()[0].set_markersize(5)
    
    # 3. Box plots by group (homogeneity of variance)
    ax3 = fig.add_subplot(2, 3, 3)
    group_data = [long_df[long_df[iv] == g][dv].values for g in groups]
    bp = ax3.boxplot(group_data, labels=[str(g) for g in groups], patch_artist=True)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_groups))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_xlabel(iv_name, fontsize=10)
    ax3.set_ylabel(dv_name, fontsize=10)
    ax3.set_title(f'Distribution by {iv_name}\n(Homogeneity Check)', fontsize=12)
    
    # Add Levene's result
    levene_stat, levene_p = stats.levene(*group_data, center='median')
    ax3.text(0.05, 0.95, f"Levene's Test:\nW = {levene_stat:.3f}\np = {levene_p:.4f}",
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Histograms per group
    ax4 = fig.add_subplot(2, 3, 4)
    for i, g in enumerate(groups):
        data = long_df[long_df[iv] == g][dv].values
        ax4.hist(data, bins=10, alpha=0.5, label=f'{iv_name} {g}', 
                 color=colors[i], edgecolor='black')
    
    ax4.set_xlabel(dv_name, fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title(f'Distributions by {iv_name}', fontsize=12)
    ax4.legend(fontsize=8)
    
    # 5. Residuals vs Group (spread check)
    ax5 = fig.add_subplot(2, 3, 5)
    for i, g in enumerate(groups):
        group_residuals = residuals[long_df[iv] == g]
        x_jitter = np.random.normal(i, 0.1, len(group_residuals))
        ax5.scatter(x_jitter, group_residuals, alpha=0.5, color=colors[i], s=30)
    
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax5.set_xticks(range(n_groups))
    ax5.set_xticklabels([str(g) for g in groups])
    ax5.set_xlabel(iv_name, fontsize=10)
    ax5.set_ylabel('Residuals', fontsize=10)
    ax5.set_title('Residuals by Group\n(Variance Homogeneity)', fontsize=12)
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for g in groups:
        data = long_df[long_df[iv] == g][dv]
        sw_stat_g, sw_p_g = stats.shapiro(data)
        summary_data.append([
            str(g),
            f'{data.mean():.2f}',
            f'{data.std():.2f}',
            f'{len(data)}',
            f'{sw_stat_g:.3f}',
            f'{sw_p_g:.4f}'
        ])
    
    table = ax6.table(
        cellText=summary_data,
        colLabels=[iv_name, 'Mean', 'SD', 'N', 'SW W', 'SW p'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title('Group Statistics & Normality Tests', fontsize=12, pad=20)
    
    plt.suptitle(f'ANOVA Assumption Diagnostics: {dv_name} by {iv_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{prefix}_assumption_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def check_missing_values(df, columns=None):
    """
    Check for missing values in the dataframe.
    
    Returns dict with missing value info.
    """
    if columns is None:
        columns = df.columns
    
    missing_info = {}
    total_missing = 0
    
    for col in columns:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100
            missing_info[col] = {'n_missing': n_missing, 'pct': pct_missing}
            total_missing += n_missing
    
    return {
        'by_column': missing_info,
        'total_missing': total_missing,
        'total_cells': len(df) * len(columns),
        'any_missing': total_missing > 0
    }

def print_missing_values_report(missing_info, title="Missing Values Check"):
    """Print a formatted missing values report."""
    
    print(f"\n{title}")
    print("-" * 50)
    
    if not missing_info['any_missing']:
        print("  ✓ No missing values found")
        return
    
    print(f"  Total missing: {missing_info['total_missing']} / {missing_info['total_cells']}")
    print("\n  By column:")
    for col, info in missing_info['by_column'].items():
        if info['n_missing'] > 0:
            print(f"    {col}: {info['n_missing']} ({info['pct']:.1f}%)")

def levenes_test(long_df, dv, iv):
    """
    Perform Levene's test for homogeneity of variance.
    
    Parameters:
        long_df: DataFrame in long format
        dv: Dependent variable column name
        iv: Independent variable (grouping) column name
    
    Returns:
        dict with test statistic, p-value, and interpretation
    """
    groups = long_df.groupby(iv)[dv]
    group_data = [group.values for name, group in groups]
    
    # Levene's test (using median, more robust)
    statistic, p_value = stats.levene(*group_data, center='median')
    
    # Interpretation
    if p_value >= 0.05:
        interpretation = "PASSED - Variances are homogeneous (p >= 0.05)"
        passed = True
    else:
        interpretation = "FAILED - Variances are NOT homogeneous (p < 0.05)"
        passed = False
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'interpretation': interpretation,
        'passed': passed
    }

def shapiro_wilk_test(data, group_name=None):
    """
    Perform Shapiro-Wilk test for normality on a single group.
    
    Parameters:
        data: Array-like of values
        group_name: Optional name for the group
    
    Returns:
        dict with test statistic, p-value, and interpretation
    """
    # Shapiro-Wilk requires 3-5000 observations
    if len(data) < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'interpretation': "Cannot test - too few observations",
            'passed': None,
            'group': group_name
        }
    
    if len(data) > 5000:
        # Sample if too large
        data = np.random.choice(data, 5000, replace=False)
    
    statistic, p_value = stats.shapiro(data)
    
    if p_value >= 0.05:
        interpretation = "PASSED - Data is normally distributed (p >= 0.05)"
        passed = True
    else:
        interpretation = "FAILED - Data is NOT normally distributed (p < 0.05)"
        passed = False
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'interpretation': interpretation,
        'passed': passed,
        'group': group_name
    }

def shapiro_wilk_by_group(long_df, dv, iv):
    """
    Perform Shapiro-Wilk test for each group.
    
    Returns list of test results for each group.
    """
    results = []
    
    for group_name, group_data in long_df.groupby(iv)[dv]:
        result = shapiro_wilk_test(group_data.values, group_name)
        results.append(result)
    
    return results

def shapiro_wilk_residuals(long_df, dv, iv):
    """
    Perform Shapiro-Wilk test on ANOVA residuals.
    
    This is often preferred over testing each group separately.
    """
    # Calculate residuals (observation - group mean)
    group_means = long_df.groupby(iv)[dv].transform('mean')
    residuals = long_df[dv] - group_means
    
    return shapiro_wilk_test(residuals.values, "Residuals")

def print_assumption_tests(long_df, dv, iv, dv_name=None, iv_name=None):
    """
    Print all ANOVA assumption tests in a formatted report.
    
    Parameters:
        long_df: DataFrame in long format
        dv: Dependent variable column name
        iv: Independent variable column name
        dv_name: Display name for DV (optional)
        iv_name: Display name for IV (optional)
    """
    if dv_name is None:
        dv_name = dv
    if iv_name is None:
        iv_name = iv
    
    print(f"\n{'='*70}")
    print(f"ANOVA ASSUMPTION TESTS: {dv_name} by {iv_name}")
    print(f"{'='*70}")
    
    # 1. Sample sizes
    print("\n1. SAMPLE SIZES PER GROUP")
    print("-" * 50)
    group_sizes = long_df.groupby(iv)[dv].count()
    for group, size in group_sizes.items():
        print(f"  {iv_name} {group}: n = {size}")
    print(f"  Total N = {len(long_df)}")
    
    # 2. Levene's test
    print("\n2. LEVENE'S TEST (Homogeneity of Variance)")
    print("-" * 50)
    levene_result = levenes_test(long_df, dv, iv)
    print(f"  W = {levene_result['statistic']:.4f}")
    print(f"  p = {levene_result['p_value']:.4f}")
    print(f"  {levene_result['interpretation']}")
    
    # 3. Shapiro-Wilk on residuals
    print("\n3. SHAPIRO-WILK TEST (Normality of Residuals)")
    print("-" * 50)
    sw_residuals = shapiro_wilk_residuals(long_df, dv, iv)
    print(f"  W = {sw_residuals['statistic']:.4f}")
    print(f"  p = {sw_residuals['p_value']:.4f}")
    print(f"  {sw_residuals['interpretation']}")
    
    # 4. Shapiro-Wilk by group (additional info)
    print("\n  By group:")
    sw_by_group = shapiro_wilk_by_group(long_df, dv, iv)
    all_normal = True
    for result in sw_by_group:
        status = "✓" if result['passed'] else "✗"
        print(f"    {status} {iv_name} {result['group']}: W = {result['statistic']:.4f}, p = {result['p_value']:.4f}")
        if not result['passed']:
            all_normal = False
    
    # Summary
    print("\n" + "-" * 50)
    print("ASSUMPTION SUMMARY:")
    
    if levene_result['passed'] and sw_residuals['passed']:
        print("  ✓ All assumptions met - ANOVA is appropriate")
    else:
        issues = []
        if not levene_result['passed']:
            issues.append("heterogeneous variances")
        if not sw_residuals['passed']:
            issues.append("non-normal residuals")
        print(f"  ⚠ Assumptions violated: {', '.join(issues)}")
        print("  → Consider: Welch's ANOVA or non-parametric alternatives")
    
    return {
        'levene': levene_result,
        'shapiro_residuals': sw_residuals,
        'shapiro_by_group': sw_by_group,
        'all_passed': levene_result['passed'] and sw_residuals['passed']
    }

def save_assumption_tests(results, output_path):
    """
    Save assumption test results to CSV.
    
    Parameters:
        results: dict from print_assumption_tests()
        output_path: Full path for output CSV file
    """
    
    rows = [
        {
            'Test': "Levene's Test",
            'Statistic': results['levene']['statistic'],
            'p_value': results['levene']['p_value'],
            'Passed': results['levene']['passed'],
            'Interpretation': results['levene']['interpretation']
        },
        {
            'Test': 'Shapiro-Wilk (Residuals)',
            'Statistic': results['shapiro_residuals']['statistic'],
            'p_value': results['shapiro_residuals']['p_value'],
            'Passed': results['shapiro_residuals']['passed'],
            'Interpretation': results['shapiro_residuals']['interpretation']
        }
    ]
    
    # Add per-group Shapiro-Wilk
    for sw in results['shapiro_by_group']:
        rows.append({
            'Test': f"Shapiro-Wilk (Group {sw['group']})",
            'Statistic': sw['statistic'],
            'p_value': sw['p_value'],
            'Passed': sw['passed'],
            'Interpretation': sw['interpretation']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

# Example usage
if __name__ == "__main__":
    # Demo with sample data
    np.random.seed(42)
    
    # Create sample data
    data = {
        'Latency': np.repeat([0, 50, 100, 150, 200], 20),
        'Difficulty': np.concatenate([
            np.random.normal(2.5, 0.8, 20),
            np.random.normal(2.7, 0.9, 20),
            np.random.normal(2.9, 0.7, 20),
            np.random.normal(3.0, 0.85, 20),
            np.random.normal(3.3, 0.9, 20)
        ])
    }
    df = pd.DataFrame(data)
    
    # Run tests
    results = print_assumption_tests(df, 'Difficulty', 'Latency', 
                                      'Perceived Difficulty', 'Latency')