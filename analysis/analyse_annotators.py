#!/usr/bin/env python3
"""
Script to analyze annotator statistics from Label Studio JSON export.
Sorts annotators by number of annotated examples and shows total time spent.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime

def parse_time_spent(lead_time_seconds):
    """Convert lead time from seconds to readable format"""
    if lead_time_seconds is None or lead_time_seconds == 0:
        return 0
    
    hours = int(lead_time_seconds // 3600)
    minutes = int((lead_time_seconds % 3600) // 60)
    seconds = int(lead_time_seconds % 60)
    
    return lead_time_seconds

def format_time(total_seconds):
    """Convert seconds to readable format"""
    if total_seconds == 0:
        return "0s"
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def get_annotator_name(completed_by_id, user_mapping=None):
    """Get annotator name from completed_by ID"""
    if user_mapping and completed_by_id in user_mapping:
        return user_mapping[completed_by_id]
    return f"User_{completed_by_id}"

def analyze_annotations(json_file):
    """Analyze annotations and generate statistics"""
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return
    
    # Create user mapping if available (from drafts or other sources)
    user_mapping = {
        12: 'pawel.borsukiewicz@uni.lu',
        14: 'tailia.malloy@uni.lu'
    }
    
    # Collect statistics
    annotator_stats = defaultdict(lambda: {
        'count': 0,
        'total_time': 0,
        'annotations': []
    })
    
    total_tasks = len(data)
    tasks_with_annotations = 0
    
    print(f"Processing {total_tasks} tasks...")
    
    for task in data:
        task_id = task.get('id', 'unknown')
        annotations = task.get('annotations', [])
        
        if annotations:
            tasks_with_annotations += 1
            
        for annotation in annotations:
            completed_by = annotation.get('completed_by')
            lead_time = annotation.get('lead_time', 0)
            created_at = annotation.get('created_at', '')
            annotation_id = annotation.get('id', 'unknown')
            
            if completed_by:
                annotator_name = get_annotator_name(completed_by, user_mapping)
                
                annotator_stats[annotator_name]['count'] += 1
                annotator_stats[annotator_name]['total_time'] += lead_time if lead_time else 0
                annotator_stats[annotator_name]['annotations'].append({
                    'task_id': task_id,
                    'annotation_id': annotation_id,
                    'lead_time': lead_time,
                    'created_at': created_at
                })
        
        # Also check drafts for additional user information
        drafts = task.get('drafts', [])
        for draft in drafts:
            user_info = draft.get('created_username', '')
            if user_info and ',' in user_info:
                username = user_info.split(',')[0].strip()
                user_id = user_info.split(',')[1].strip() if ',' in user_info else None
                if user_id and user_id.isdigit():
                    user_mapping[int(user_id)] = username
    
    # Re-process with updated user mapping
    if user_mapping:
        print(f"Found user mapping: {user_mapping}")
        # Update annotator names
        updated_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'annotations': []
        })
        
        for old_name, stats in annotator_stats.items():
            if old_name.startswith('User_'):
                user_id = int(old_name.split('_')[1])
                new_name = user_mapping.get(user_id, old_name)
            else:
                new_name = old_name
            
            updated_stats[new_name] = stats
        
        annotator_stats = updated_stats
    
    # Sort by annotation count (descending)
    sorted_annotators = sorted(
        annotator_stats.items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("ANNOTATOR STATISTICS (Sorted by Number of Annotations)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Annotator':<30} {'Count':<8} {'Total Time':<15} {'Avg Time/Task':<15}")
    print("-" * 80)
    
    total_annotations = 0
    total_time = 0
    
    for rank, (annotator, stats) in enumerate(sorted_annotators, 1):
        count = stats['count']
        time_spent = stats['total_time']
        avg_time = time_spent / count if count > 0 else 0
        
        formatted_total_time = format_time(time_spent)
        formatted_avg_time = format_time(avg_time)
        
        print(f"{rank:<5} {annotator:<30} {count:<8} {formatted_total_time:<15} {formatted_avg_time:<15}")
        
        total_annotations += count
        total_time += time_spent
    
    # Summary statistics
    print("-" * 80)
    print(f"{'TOTAL':<5} {'':<30} {total_annotations:<8} {format_time(total_time):<15}")
    print(f"\nSummary:")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Tasks with Annotations: {tasks_with_annotations}")
    print(f"  Total Annotators: {len(annotator_stats)}")
    print(f"  Total Annotations: {total_annotations}")
    print(f"  Average Annotations per Annotator: {total_annotations / len(annotator_stats) if annotator_stats else 0:.1f}")
    print(f"  Average Time per Annotator: {format_time(total_time / len(annotator_stats)) if annotator_stats else '0s'}")
    print(f"  Average Time per Annotation: {format_time(total_time / total_annotations) if total_annotations > 0 else '0s'}")
    
    # Additional detailed stats
    if len(sorted_annotators) > 0:
        print(f"\nTop Annotator Details:")
        top_annotator, top_stats = sorted_annotators[0]
        print(f"  {top_annotator}: {top_stats['count']} annotations, {format_time(top_stats['total_time'])} total")
        
        # Show date range for top annotator
        if top_stats['annotations']:
            dates = [ann['created_at'] for ann in top_stats['annotations'] if ann['created_at']]
            if dates:
                dates.sort()
                print(f"  Date range: {dates[0][:10]} to {dates[-1][:10]}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_annotators.py <json_file>")
        print("Example: python analyze_annotators.py full_annotations.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    analyze_annotations(json_file)

if __name__ == "__main__":
    main()
