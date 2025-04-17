# Career Recommender System

This system recommends career paths based on a user's skills and experience from their resume. It includes a machine learning model trained on synthetic data and can personalize recommendations based on user feedback.

## Project Structure

```
recommender/
├── data/                     # Data files
│   ├── resume_1.txt          # Sample resume
│   ├── synthetic_career_path_data.csv
│   ├── synthetic_employee_data.csv
│   └── user_feedback.json    # User feedback database
│
├── models/                   # Machine learning models
│   └── career_path_recommendation_model.pkl
│
├── scripts/                  # Runner scripts
│   ├── main.py               # Main logic
│   ├── run_recommender.bat   # Windows batch file
│   ├── run_recommender.py    # Package installer/runner
│   └── run_recommender.sh    # Unix shell script
│
├── utils/                    # Utility modules
│   ├── data_processing.py    # Resume parsing utilities
│   ├── feedback.py           # Feedback handling
│   └── requirements.txt      # Required Python packages
│
├── docs/                     # Documentation
│   ├── README.md             # This file
│   └── HOW_TO_RUN.txt        # Quick start guide
│
├── __init__.py               # Package initialization
├── recommender.py            # Core recommendation module
└── run_recommender.py        # Easy entry point
```

## Installation

Before running the system, you need to install the required Python packages:

```bash
pip install -r utils/requirements.txt
```

Alternatively, the system will automatically check and install required packages when you run it.

## How to Run

### Windows:
1. Double-click on `run_recommender.py`
   OR
2. Open a command prompt in this directory and run:
   ```
   python run_recommender.py
   ```

### Linux/Mac:
1. Make the script executable (one-time setup):
   ```
   chmod +x run_recommender.py
   ```
   
2. Run the script:
   ```
   ./run_recommender.py
   ```
   OR
   
3. Run with Python directly:
   ```
   python3 run_recommender.py
   ```

## Features

- **Resume Parsing**: Extracts skills and experience from resume text files
- **Field Classification**: Identifies the most suitable career field (e.g., Computer Science, Engineering)
- **Career Path Recommendations**: Suggests specific roles within the field
- **Skills Gap Analysis**: Identifies missing skills for recommended career paths
- **Training Recommendations**: Suggests courses to acquire missing skills
- **Feedback Learning**: Improves future recommendations based on user feedback

## Memory and Learning

The system remembers user feedback through a JSON database file. When you provide feedback about recommendations:

1. It stores your preferences and ratings
2. For future recommendations with the same user ID, it prioritizes career paths you've shown interest in
3. The more you use the system, the more personalized your recommendations become

To see this in action, run the system multiple times with the same user ID and provide feedback each time. 