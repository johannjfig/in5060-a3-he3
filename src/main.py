#!/usr/bin/env python3
"""
Main Analysis Runner
====================
Runs all hypothesis analysis scripts (H1, H2, H3, H4) for the VR telepresence study.

Usage (from project root directory):
    python3 src/main.py out/combined.csv data/questionnaire_data.xlsx

This will run:
    - H1: Latency → Perceived Difficulty
    - H2: Latency → Objective Performance
    - H3: Learning Effect (Trial 1 vs Trial 10)
    - H4: Subjective-Objective Correlation

Output will be saved to:
    - out/h1out/
    - out/h2out/
    - out/h3out/
    - out/h4out/
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_path, args, description, working_dir):
    """Run a Python script and capture output."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, script_path] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=working_dir  # Run from project root
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("WARNINGS/ERRORS:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False
            
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_path}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {script_path}: {e}")
        return False

def main():
    """Main function to run all analyses."""
    
    print("="*70)
    print("VR TELEPRESENCE LATENCY STUDY - COMPLETE ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check arguments
    if len(sys.argv) < 3:
        print("\nUsage: python3 src/main.py <combined_csv> <questionnaire_xlsx>")
        print("\nExample (from project root):")
        print("  python3 src/main.py out/combined.csv data/questionnaire_data.xlsx")
        sys.exit(1)
    
    combined_csv = sys.argv[1]
    questionnaire_xlsx = sys.argv[2]
    
    # Determine project root (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(script_dir)  # parent of src/
    
    # Convert relative paths to absolute (relative to where command was run)
    if not os.path.isabs(combined_csv):
        combined_csv_abs = os.path.abspath(combined_csv)
    else:
        combined_csv_abs = combined_csv
        
    if not os.path.isabs(questionnaire_xlsx):
        questionnaire_xlsx_abs = os.path.abspath(questionnaire_xlsx)
    else:
        questionnaire_xlsx_abs = questionnaire_xlsx
    
    # Check files exist
    if not os.path.exists(combined_csv_abs):
        print(f"\n✗ Error: File not found: {combined_csv_abs}")
        print(f"  (Looking for: {combined_csv})")
        sys.exit(1)
    
    if not os.path.exists(questionnaire_xlsx_abs):
        print(f"\n✗ Error: File not found: {questionnaire_xlsx_abs}")
        print(f"  (Looking for: {questionnaire_xlsx})")
        sys.exit(1)
    
    print(f"\nProject root: {project_root}")
    print(f"Script directory: {script_dir}")
    print(f"\nInput files:")
    print(f"  Combined CSV: {combined_csv_abs}")
    print(f"  Questionnaire: {questionnaire_xlsx_abs}")
    
    # Define analyses to run
    # Scripts are in src/ directory, named h1_anovo.py, h2_anovo.py, etc.
    analyses = [
        {
            'script': os.path.join(script_dir, 'h1_anovo.py'),
            'args': [combined_csv_abs],
            'description': 'H1: Latency → Perceived Difficulty',
            'output_dir': 'out/h1out'
        },
        {
            'script': os.path.join(script_dir, 'h2_anovo.py'),
            'args': [combined_csv_abs],
            'description': 'H2: Latency → Objective Performance',
            'output_dir': 'out/h2out'
        },
        {
            'script': os.path.join(script_dir, 'h3_anovo.py'),
            'args': [questionnaire_xlsx_abs],
            'description': 'H3: Learning Effect Across Trials',
            'output_dir': 'out/h3out'
        },
        {
            'script': os.path.join(script_dir, 'h4_anovo.py'),
            'args': [combined_csv_abs],
            'description': 'H4: Subjective-Objective Correlation',
            'output_dir': 'out/h4out'
        }
    ]
    
    # Check all scripts exist
    print("\nChecking scripts:")
    for analysis in analyses:
        if os.path.exists(analysis['script']):
            print(f"  ✓ {os.path.basename(analysis['script'])}")
        else:
            print(f"  ✗ {analysis['script']} NOT FOUND")
    
    # Run each analysis from project root directory
    results = {}
    for analysis in analyses:
        if not os.path.exists(analysis['script']):
            print(f"\n✗ Skipping {analysis['description']} - script not found")
            results[analysis['description']] = False
            continue
            
        success = run_script(
            analysis['script'],
            analysis['args'],
            analysis['description'],
            project_root  # Run from project root so output goes to out/
        )
        results[analysis['description']] = success
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    all_success = True
    for description, success in results.items():
        status = "✓ COMPLETE" if success else "✗ FAILED"
        print(f"  {status}: {description}")
        if not success:
            all_success = False
    
    # List output directories
    print("\nOutput directories:")
    for analysis in analyses:
        output_dir = os.path.join(project_root, analysis['output_dir'])
        if os.path.exists(output_dir):
            n_files = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
            print(f"  {analysis['output_dir']}/: {n_files} files")
        else:
            print(f"  {analysis['output_dir']}/: (not created)")
    
    print("\n" + "="*70)
    if all_success:
        print("ALL ANALYSES COMPLETED SUCCESSFULLY")
    else:
        print("SOME ANALYSES FAILED - Check output above for errors")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())