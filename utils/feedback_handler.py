#!/usr/bin/env python3
"""
Feedback handler for career recommender system.
Collects and processes user feedback on recommendations.
"""

import os
import json
import datetime
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the feedback data file
FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "user_feedback.json")

def save_recommendation_feedback(feedback_data):
    """
    Save user feedback on recommendations.
    
    Args:
        feedback_data (dict): Dictionary containing feedback information
            {
                'user_id': str,
                'skills': list,
                'recommended_field': str,
                'recommended_specialization': str,
                'user_selected_field': str,
                'user_selected_specialization': str,
                'feedback_score': int,  # 1-5 rating
                'additional_comments': str
            }
            
    Returns:
        bool: True if feedback was saved successfully, False otherwise
    """
    try:
        # Validate feedback data
        required_fields = ['user_id', 'skills', 'recommended_field', 'recommended_specialization']
        for field in required_fields:
            if field not in feedback_data:
                logger.error(f"Missing required field in feedback data: {field}")
                return False
                
        # Add timestamp
        feedback_data['timestamp'] = datetime.datetime.now().isoformat()
        
        # Load existing feedback if file exists
        all_feedback = []
        if os.path.exists(FEEDBACK_PATH):
            try:
                with open(FEEDBACK_PATH, 'r') as f:
                    all_feedback = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error parsing existing feedback file, creating new file")
                all_feedback = []
                
        # Append new feedback
        all_feedback.append(feedback_data)
        
        # Save updated feedback
        with open(FEEDBACK_PATH, 'w') as f:
            json.dump(all_feedback, f, indent=4)
            
        logger.info(f"Saved feedback from user {feedback_data.get('user_id')}")
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return False
        
def get_feedback_stats(min_date=None, max_date=None):
    """
    Get statistics on collected feedback.
    
    Args:
        min_date (str): ISO formatted date string for start date (inclusive)
        max_date (str): ISO formatted date string for end date (inclusive)
        
    Returns:
        dict: Statistics on feedback
    """
    try:
        if not os.path.exists(FEEDBACK_PATH):
            return {
                'total_feedback': 0,
                'average_score': 0,
                'field_agreement': 0,
                'specialization_agreement': 0,
                'field_distribution': {},
                'specialization_distribution': {}
            }
            
        with open(FEEDBACK_PATH, 'r') as f:
            all_feedback = json.load(f)
            
        # Filter by date if specified
        filtered_feedback = all_feedback
        if min_date or max_date:
            filtered_feedback = []
            min_date_obj = datetime.datetime.fromisoformat(min_date) if min_date else None
            max_date_obj = datetime.datetime.fromisoformat(max_date) if max_date else None
            
            for item in all_feedback:
                item_date = datetime.datetime.fromisoformat(item.get('timestamp', '2000-01-01T00:00:00'))
                
                if min_date_obj and item_date < min_date_obj:
                    continue
                    
                if max_date_obj and item_date > max_date_obj:
                    continue
                    
                filtered_feedback.append(item)
                
        # Calculate statistics
        total = len(filtered_feedback)
        
        if total == 0:
            return {
                'total_feedback': 0,
                'average_score': 0,
                'field_agreement': 0,
                'specialization_agreement': 0,
                'field_distribution': {},
                'specialization_distribution': {}
            }
            
        # Average score
        scores = [item.get('feedback_score', 0) for item in filtered_feedback if 'feedback_score' in item]
        average_score = sum(scores) / len(scores) if scores else 0
        
        # Field agreement rate
        field_matches = 0
        for item in filtered_feedback:
            rec_field = item.get('recommended_field', '')
            user_field = item.get('user_selected_field', '')
            
            if user_field and rec_field and user_field == rec_field:
                field_matches += 1
                
        field_agreement = field_matches / total if total > 0 else 0
        
        # Specialization agreement rate
        spec_matches = 0
        for item in filtered_feedback:
            rec_spec = item.get('recommended_specialization', '')
            user_spec = item.get('user_selected_specialization', '')
            
            if user_spec and rec_spec and user_spec == rec_spec:
                spec_matches += 1
                
        spec_agreement = spec_matches / total if total > 0 else 0
        
        # Field distribution
        field_dist = defaultdict(int)
        for item in filtered_feedback:
            field = item.get('user_selected_field', item.get('recommended_field', 'Unknown'))
            field_dist[field] += 1
            
        # Specialization distribution
        spec_dist = defaultdict(int)
        for item in filtered_feedback:
            spec = item.get('user_selected_specialization', item.get('recommended_specialization', 'Unknown'))
            spec_dist[spec] += 1
            
        return {
            'total_feedback': total,
            'average_score': round(average_score, 2),
            'field_agreement': round(field_agreement * 100, 2),
            'specialization_agreement': round(spec_agreement * 100, 2),
            'field_distribution': dict(field_dist),
            'specialization_distribution': dict(spec_dist)
        }
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        return {
            'error': str(e),
            'total_feedback': 0
        }
        
def get_recent_feedback(count=10):
    """
    Get most recent feedback entries.
    
    Args:
        count (int): Number of entries to return
        
    Returns:
        list: List of feedback entries
    """
    try:
        if not os.path.exists(FEEDBACK_PATH):
            return []
            
        with open(FEEDBACK_PATH, 'r') as f:
            all_feedback = json.load(f)
            
        # Sort by timestamp descending
        sorted_feedback = sorted(
            all_feedback,
            key=lambda x: x.get('timestamp', '2000-01-01T00:00:00'),
            reverse=True
        )
        
        # Return requested number of entries
        return sorted_feedback[:count]
    except Exception as e:
        logger.error(f"Error getting recent feedback: {str(e)}")
        return []
        
def get_feedback_for_update(min_score=None, max_entries=1000, exclude_processed=True):
    """
    Get feedback entries for model update.
    
    Args:
        min_score (int): Minimum feedback score to include (None for all)
        max_entries (int): Maximum number of entries to return
        exclude_processed (bool): Whether to exclude already processed entries
        
    Returns:
        list: List of feedback entries for update
    """
    try:
        if not os.path.exists(FEEDBACK_PATH):
            return []
            
        with open(FEEDBACK_PATH, 'r') as f:
            all_feedback = json.load(f)
            
        # Filter entries
        filtered_feedback = []
        
        for item in all_feedback:
            # Skip if already processed and exclude_processed is True
            if exclude_processed and item.get('processed_for_update', False):
                continue
                
            # Skip if score is below minimum (if specified)
            if min_score is not None and item.get('feedback_score', 0) < min_score:
                continue
                
            # Include this item
            filtered_feedback.append(item)
            
        # Sort by timestamp (oldest first to maintain chronological processing)
        sorted_feedback = sorted(
            filtered_feedback,
            key=lambda x: x.get('timestamp', '2000-01-01T00:00:00')
        )
        
        # Return limited number of entries
        return sorted_feedback[:max_entries]
    except Exception as e:
        logger.error(f"Error getting feedback for update: {str(e)}")
        return []
        
def mark_feedback_as_processed(feedback_ids):
    """
    Mark feedback entries as processed for update.
    
    Args:
        feedback_ids (list): List of feedback entry IDs to mark
        
    Returns:
        bool: True if marking was successful, False otherwise
    """
    try:
        if not os.path.exists(FEEDBACK_PATH) or not feedback_ids:
            return False
            
        with open(FEEDBACK_PATH, 'r') as f:
            all_feedback = json.load(f)
            
        # Mark entries as processed
        updated = False
        for item in all_feedback:
            if item.get('user_id') in feedback_ids:
                item['processed_for_update'] = True
                item['processed_at'] = datetime.datetime.now().isoformat()
                updated = True
                
        # Save updates if any were made
        if updated:
            with open(FEEDBACK_PATH, 'w') as f:
                json.dump(all_feedback, f, indent=4)
                
        return updated
    except Exception as e:
        logger.error(f"Error marking feedback as processed: {str(e)}")
        return False 