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
    """Save assumption test results to CSV."""
    
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