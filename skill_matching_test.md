# Skill Matching Test Results

## Date

April 17, 2025

## Overview

This document summarizes tests of the career recommender's ability to handle non-existent skills through fuzzy matching and skill variants.

## Test Input

We tested with the following non-existent or variant skills:

```
{
    "Projectt Managementt": 90,     # Misspelling of "Project Management"
    "Team Leadership": 85,          # Similar to "Leadership"
    "Verbal Communications": 88,     # Variation of "Communication"
    "Risk Assessment": 82,          # Similar to "Risk Management"
    "Client Relations": 85,         # New skill, not in the system
    "Project Planning": 90,         # Similar to "Planning"
    "Python Coding": 75,            # Variation of "Python"
    "Javascript Programming": 70,    # Variation of "JavaScript"
    "Agile Method": 85,             # Similar to "Agile Methodologies"
    "Scrum Master": 80,             # Similar to "Scrum"
    "Automated Testing": 60         # Similar to "Testing"
}
```

## Test Results

### Initial Test

With default fuzzy matching (threshold 0.8), only 3 out of 11 skills were matched correctly. Lowering the threshold gradually improved matches:

| Threshold | Skills Matched | Match Rate |
| --------- | -------------- | ---------- |
| 0.8       | 3              | 27%        |
| 0.7       | 4              | 36%        |
| 0.6       | 7              | 64%        |
| 0.5       | 10             | 91%        |

### After Adding Common Skill Variants

We added several common skill variants:

-   "Agile" and "Agile Development" (variants of "Agile Methodologies")
-   "Scrum Master Certification" (variant of "Scrum")

With these additions, match rates improved significantly:

| Threshold | Skills Matched | Match Rate |
| --------- | -------------- | ---------- |
| 0.8       | 3              | 27%        |
| 0.7       | 5              | 45%        |
| 0.6       | 9              | 82%        |
| 0.5       | 11             | 100%       |

## Findings

1. **Fuzzy matching effectiveness varies by threshold**

    - Higher thresholds (0.8) only match very similar skills
    - Lower thresholds match more skills but may introduce incorrect matches
    - 0.6-0.7 appears to be the optimal range

2. **Adding common skill variants is highly effective**

    - With minimal additions, match rates improved dramatically
    - Strategic selection of common variants yields high ROI

3. **Some skills require exact matching**

    - Certain skills like "Client Relations" don't have good matches
    - A comprehensive skill taxonomy would be beneficial

4. **Recommendation quality is maintained**
    - Despite using skill variants, the system still recommends appropriate career paths
    - Project Manager remained a top recommendation throughout tests

## Recommendations

1. **Implement a two-tier matching approach**:

    - First attempt match with high threshold (0.8)
    - For unmatched skills, try again with lower threshold (0.6)
    - Flag questionable matches for user review

2. **Expand skill variants database**:

    - Add common variants and misspellings of popular skills
    - Consider automatic generation of common variants

3. **Consider user feedback loop**:

    - When users enter unmatched skills, suggest possible matches
    - Allow users to confirm matches or suggest alternatives

4. **Documentation for users**:
    - Provide a list of recognized skills to guide users
    - Implement auto-complete for skills input

## Next Steps

1. Implement the two-tier matching approach
2. Expand the skill variants database with more common variants
3. Add feedback mechanism for unmatched skills
4. Create user documentation for skill input
