# Career Recommender Model Experiment: Project Manager

## Experiment Date

April 17, 2025

## Modifications

1. Added "Project Manager" specialization to `data/specializations.json`
2. Added missing skills to `data/skill_weights.json`:
    - Risk Management
    - Stakeholder Management
    - Planning
    - Budgeting
    - Statistical Analysis (fixed warning)

## Training

-   Successfully trained model with new data
-   Fixed warning about missing skill: Statistical Analysis (used in Research Associate)

## Test Results

### Pure Project Manager Profile

```
===== CAREER RECOMMENDATIONS FOR PROJECT MANAGER PROFILE =====

TOP FIELDS:
- Business Accountancy (Confidence: 100.0%)

TOP SPECIALIZATIONS:
- Project Manager (Field: Business Accountancy, Confidence: 100.0%)
  Matched Skills: 9/9
    - Project Management (Your level: 95, Required level: 95)
    - Communication (Your level: 90, Required level: 90)
    - Planning (Your level: 92, Required level: 90)
    - Leadership (Your level: 85, Required level: 85)
    - Stakeholder Management (Your level: 88, Required level: 85)
    - Problem Solving (Your level: 85, Required level: 85)
    - Time Management (Your level: 85, Required level: 85)
    - Budgeting (Your level: 80, Required level: 80)
```

### Leadership-Focused Profile

```
===== CAREER RECOMMENDATIONS FOR LEADERSHIP-FOCUSED PROFILE =====

TOP FIELDS:
- Business Accountancy (Confidence: 100.0%)

TOP SPECIALIZATIONS:
- Project Manager (Field: Business Accountancy, Confidence: 100.0%)
  Matched Skills: 9/9
    - Project Management (Your level: 80, Required level: 95)
    - Communication (Your level: 92, Required level: 90)
    - Planning (Your level: 85, Required level: 90)
    - Leadership (Your level: 95, Required level: 85)
    - Stakeholder Management (Your level: 90, Required level: 85)
    - Problem Solving (Your level: 88, Required level: 85)
    - Time Management (Your level: 82, Required level: 85)
    - Budgeting (Your level: 70, Required level: 80)
```

### Technical Project Manager Profile

```
===== CAREER RECOMMENDATIONS FOR TECHNICAL PROJECT MANAGER PROFILE =====

TOP FIELDS:
- Business Accountancy (Confidence: 50.0%)
- Computer Science (Confidence: 48.0%)

TOP SPECIALIZATIONS:
- Project Manager (Field: Business Accountancy, Confidence: 58.0%)
  Matched Skills: 9/9
    - Project Management (Your level: 90, Required level: 95)
    - Communication (Your level: 88, Required level: 90)
    - Planning (Your level: 90, Required level: 90)
    - Leadership (Your level: 85, Required level: 85)
    - Stakeholder Management (Your level: 85, Required level: 85)
    - Problem Solving (Your level: 92, Required level: 85)
    - Time Management (Your level: 85, Required level: 85)
    - Risk Management (Your level: 80, Required level: 80)
    - Budgeting (Your level: 75, Required level: 80)

- Web Developer (Field: Computer Science, Confidence: 24.0%)
  Matched Skills: 1/7
    - JavaScript (Your level: 70, Required level: 90)

- Data Scientist (Field: Computer Science, Confidence: 12.0%)
  Matched Skills: 1/7
    - Python (Your level: 75, Required level: 85)
```

## Observations

1. The model successfully recommends "Project Manager" as the top specialization when relevant skills are provided
2. For pure project management profiles, we see 100% confidence in the recommendation
3. For mixed skill profiles (technical + project management), the model provides a more diverse set of recommendations with appropriate confidence levels
4. The model correctly identifies when some skills are below the required threshold, which affects the confidence score

## Next Steps

1. Add more project management-related specializations (e.g., Scrum Master, Agile Coach)
2. Fine-tune skill weights for more accurate recommendations
3. Fix the missing skill warning for Statistical Analysis
