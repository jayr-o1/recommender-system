#!/usr/bin/env python3
"""
Script to schedule regular updates of the career recommender model.
This script runs on a scheduled basis to update the model with new feedback.
"""

import os
import sys
import time
import logging
import argparse
import schedule
import subprocess
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommender_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recommender_scheduler")

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPDATE_SCRIPT = os.path.join(SCRIPT_DIR, "update_from_feedback.py")

def run_update(verbose=False, max_entries=None):
    """
    Run the update script as a subprocess.
    
    Args:
        verbose (bool): Whether to run in verbose mode
        max_entries (int): Maximum number of feedback entries to process
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        logger.info("Starting scheduled model update")
        
        # Build command
        cmd = [sys.executable, UPDATE_SCRIPT]
        
        if verbose:
            cmd.append("--verbose")
            
        if max_entries:
            cmd.extend(["--max-entries", str(max_entries)])
            
        # Run the update script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Capture output
        stdout, stderr = process.communicate()
        
        # Check if successful
        if process.returncode == 0:
            logger.info("Scheduled update completed successfully")
            if verbose:
                logger.info(f"Output: {stdout}")
            return True
        else:
            logger.error(f"Scheduled update failed with exit code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running scheduled update: {str(e)}")
        return False
        
def main():
    """
    Main function to schedule regular model updates.
    """
    parser = argparse.ArgumentParser(description='Schedule regular model updates')
    parser.add_argument('--interval', type=str, default='daily',
                       choices=['hourly', 'daily', 'weekly'],
                       help='Update interval (hourly, daily, weekly)')
    parser.add_argument('--time', type=str, default='03:00',
                       help='Time for daily/weekly updates (HH:MM)')
    parser.add_argument('--weekday', type=str, default='monday',
                       choices=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                       help='Day of week for weekly updates')
    parser.add_argument('--run-now', action='store_true',
                       help='Run an update immediately')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--max-entries', type=int, default=None,
                       help='Maximum number of feedback entries to process per update')
    
    args = parser.parse_args()
    
    # Schedule updates based on interval
    if args.interval == 'hourly':
        schedule.every().hour.do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        logger.info("Scheduled hourly updates")
    elif args.interval == 'daily':
        hour, minute = map(int, args.time.split(':'))
        schedule.every().day.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        logger.info(f"Scheduled daily updates at {args.time}")
    elif args.interval == 'weekly':
        hour, minute = map(int, args.time.split(':'))
        if args.weekday == 'monday':
            schedule.every().monday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        elif args.weekday == 'tuesday':
            schedule.every().tuesday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        elif args.weekday == 'wednesday':
            schedule.every().wednesday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        elif args.weekday == 'thursday':
            schedule.every().thursday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        elif args.weekday == 'friday':
            schedule.every().friday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        elif args.weekday == 'saturday':
            schedule.every().saturday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
        elif args.weekday == 'sunday':
            schedule.every().sunday.at(args.time).do(run_update, verbose=args.verbose, max_entries=args.max_entries)
            
        logger.info(f"Scheduled weekly updates on {args.weekday} at {args.time}")
        
    # Run an update immediately if requested
    if args.run_now:
        logger.info("Running immediate update as requested")
        run_update(verbose=args.verbose, max_entries=args.max_entries)
        
    # Keep the script running to execute scheduled tasks
    logger.info("Scheduler running, press Ctrl+C to exit")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        
if __name__ == "__main__":
    main() 