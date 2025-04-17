"""
Script to generate synthetic data for the HR career recommendation system.
"""

import os
import argparse
from utils.data_generator import SyntheticDataGenerator

def main():
    """Main function to generate synthetic data."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data for HR career recommendations')
    parser.add_argument('--employee-count', type=int, default=200, 
                        help='Number of employee records to generate')
    parser.add_argument('--career-path-count', type=int, default=150, 
                        help='Number of career path records to generate')
    parser.add_argument('--employee-file', type=str, default=None, 
                        help='Output file for employee data (default: data/synthetic_employee_data.json)')
    parser.add_argument('--career-file', type=str, default=None, 
                        help='Output file for career path data (default: data/synthetic_career_path_data.json)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--keep-existing', action='store_true', 
                        help='Do not overwrite existing files')
    parser.add_argument('--append', action='store_true', 
                        help='Append to existing files instead of replacing them')
    parser.add_argument('--replace', action='store_true', 
                        help='Replace existing files instead of appending (overrides --append)')
    parser.add_argument('--train', action='store_true', 
                        help='Train the model after generating data')
    
    args = parser.parse_args()
    
    # Set default file paths if not provided
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if args.employee_file is None:
        args.employee_file = os.path.join(data_dir, "synthetic_employee_data.json")
        
    if args.career_file is None:
        args.career_file = os.path.join(data_dir, "synthetic_career_path_data.json")
    
    # Determine append mode - default is to append if files exist unless replace is specified
    should_append = args.append or (not args.replace and not args.keep_existing)
    
    # Check if files exist and handle based on flags
    if args.keep_existing:
        if os.path.exists(args.employee_file):
            print(f"Employee file {args.employee_file} exists. Skipping generation.")
            employee_generated = False
        else:
            employee_generated = True
            
        if os.path.exists(args.career_file):
            print(f"Career path file {args.career_file} exists. Skipping generation.")
            career_generated = False
        else:
            career_generated = True
            
        # If both files exist, exit
        if not employee_generated and not career_generated:
            print("Both files exist and --keep-existing is set. Nothing to do.")
            return
    else:
        employee_generated = True
        career_generated = True
    
    # Initialize the data generator
    print(f"Initializing data generator with seed {args.seed}...")
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate the datasets
    if employee_generated and career_generated:
        # Generate both datasets
        append_mode = should_append and os.path.exists(args.employee_file) and os.path.exists(args.career_file)
        mode_str = "append to" if append_mode else "create new"
        print(f"Will {mode_str} data files...")
        print(f"Generating {args.employee_count} employee records and {args.career_path_count} career path records...")
        generator.generate_datasets(
            employee_count=args.employee_count,
            career_path_count=args.career_path_count,
            employee_file=args.employee_file,
            career_file=args.career_file,
            append=append_mode
        )
    else:
        # Generate only the needed datasets
        if employee_generated:
            append_mode = should_append and os.path.exists(args.employee_file)
            mode_str = "append to" if append_mode else "create new"
            print(f"Will {mode_str} employee data file...")
            print(f"Generating {args.employee_count} employee records...")
            generator.generate_employee_data(
                num_entries=args.employee_count,
                output_file=args.employee_file,
                append=append_mode
            )
            
        if career_generated:
            append_mode = should_append and os.path.exists(args.career_file)
            mode_str = "append to" if append_mode else "create new"
            print(f"Will {mode_str} career path data file...")
            print(f"Generating {args.career_path_count} career path records...")
            generator.generate_career_path_data(
                num_entries=args.career_path_count,
                output_file=args.career_file,
                append=append_mode
            )
    
    print("Data generation complete!")
    
    # Train the model if requested
    if args.train:
        print("\nTraining model with newly generated data...")
        from utils.model_trainer import initial_model_training
        success = initial_model_training(verbose=True)
        
        if success:
            print("Model training completed successfully!")
        else:
            print("Model training failed. Check logs for details.")

if __name__ == "__main__":
    main() 