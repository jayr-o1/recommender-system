#!/usr/bin/env python3
"""
Script to update the career recommender model with user feedback.
This script can be run periodically to incorporate user feedback into the model.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommender_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("recommender_update")

# Add parent directory to path to handle imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
from utils.feedback_handler import get_feedback_for_update, mark_feedback_as_processed, get_feedback_stats
from utils.model_trainer import update_model_with_feedback

def update_model():
    """
    Main function to update the model with user feedback.
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    parser = argparse.ArgumentParser(description='Update recommender model with user feedback')
    parser.add_argument('--min-score', type=int, default=None,
                       help='Minimum feedback score to include (1-5)')
    parser.add_argument('--max-entries', type=int, default=1000,
                       help='Maximum number of feedback entries to process')
    parser.add_argument('--include-processed', action='store_true',
                       help='Include previously processed feedback entries')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Get statistics before update
        before_stats = get_feedback_stats()
        logger.info(f"Current statistics: {before_stats['total_feedback']} total feedback entries, "
                   f"{before_stats['average_score']} average score, "
                   f"{before_stats['field_agreement']}% field agreement")
        
        # Get feedback for update
        exclude_processed = not args.include_processed
        feedback_entries = get_feedback_for_update(
            min_score=args.min_score,
            max_entries=args.max_entries,
            exclude_processed=exclude_processed
        )
        
        if not feedback_entries:
            logger.info("No feedback entries found for model update")
            return True
            
        logger.info(f"Found {len(feedback_entries)} feedback entries for model update")
        
        if args.dry_run:
            logger.info("Dry run mode - showing entries that would be processed:")
            for i, entry in enumerate(feedback_entries[:10]):  # Show first 10
                logger.info(f"  {i+1}. User: {entry.get('user_id')}, "
                           f"Score: {entry.get('feedback_score', 'N/A')}, "
                           f"Timestamp: {entry.get('timestamp', 'N/A')}")
                
            if len(feedback_entries) > 10:
                logger.info(f"  ... and {len(feedback_entries) - 10} more entries")
                
            logger.info("Dry run completed - no changes made")
            return True
            
        # Update the model with feedback
        success = update_model_with_feedback(feedback_entries, verbose=args.verbose)
        
        if not success:
            logger.error("Failed to update model with feedback")
            return False
            
        # Mark entries as processed
        user_ids = [entry.get('user_id') for entry in feedback_entries if 'user_id' in entry]
        mark_result = mark_feedback_as_processed(user_ids)
        
        if mark_result:
            logger.info(f"Marked {len(user_ids)} feedback entries as processed")
        else:
            logger.warning("Failed to mark feedback entries as processed")
            
        # Get statistics after update
        after_stats = get_feedback_stats()
        logger.info(f"Updated statistics: {after_stats['total_feedback']} total feedback entries, "
                   f"{after_stats['average_score']} average score, "
                   f"{after_stats['field_agreement']}% field agreement")
        
        return True
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting model update at {start_time.isoformat()}")
    
    success = update_model()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if success:
        logger.info(f"Model update completed successfully in {duration.total_seconds():.2f} seconds")
    else:
        logger.error(f"Model update failed after {duration.total_seconds():.2f} seconds")
        sys.exit(1)  # Exit with error code for scripts
        
    sys.exit(0)  # Exit with success code 