"""
Script to train the career recommender model with synthetic data.
This script generates synthetic data and trains the model in one step.
"""

import argparse
import os
from utils.model_trainer import initial_model_training
from utils.data_generator import SyntheticDataGenerator

def main():
    """Main function to generate data and train the model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate data and train the career recommender model')
    parser.add_argument('--employee-count', type=int, default=1000, 
                        help='Number of employee records to generate')
    parser.add_argument('--career-path-count', type=int, default=800, 
                        help='Number of career path records to generate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--replace', action='store_true', default=True,
                        help='Replace existing data files')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    employee_file = os.path.join(data_dir, "synthetic_employee_data.json")
    career_file = os.path.join(data_dir, "synthetic_career_path_data.json")
    
    # Generate synthetic data
    if verbose:
        print("=== Generating Synthetic Data ===")
        print(f"Initializing data generator with seed {args.seed}...")
    
    generator = SyntheticDataGenerator(seed=args.seed)
    
    mode_str = "Creating new" if args.replace else "Appending to"
    if verbose:
        print(f"{mode_str} data files...")
        print(f"Generating {args.employee_count} employee records and {args.career_path_count} career path records...")
    
    generator.generate_datasets(
        employee_count=args.employee_count,
        career_path_count=args.career_path_count,
        employee_file=employee_file,
        career_file=career_file,
        append=not args.replace
    )
    
    if verbose:
        print("Data generation complete!")
        print("\n=== Training Model ===")
    
    # Train the model
    success = initial_model_training(verbose=verbose)
    
    if success:
        if verbose:
            print("\nModel training completed successfully!")
    else:
        if verbose:
            print("\nModel training failed. Check logs for details.")

if __name__ == "__main__":
    main() 