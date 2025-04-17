#!/usr/bin/env python3
"""
Generate a comprehensive specialization_skills.json file with specializations
across multiple professional domains.
"""

import os
import json
import random

# Define domains and their specializations with relevant skills
DOMAINS = {
    "Computer Science": {
        "Software Engineer": [
            "Data Structures", "Algorithms", "Code Optimization", "Software Architecture", 
            "Design Patterns", "Version Control", "Git", "Code Review", 
            "Test-Driven Development", "CI/CD", "Agile Methodology", "Debugging", 
            "Refactoring", "System Design", "OOP", "Functional Programming", 
            "Microservices", "RESTful APIs", "GraphQL", "Scalability", 
            "Performance Tuning", "Documentation", "Containerization", 
            "Cloud Services", "Multi-threading", "Unit Testing", "Integration Testing",
            "Database Design", "SQL", "NoSQL", "Web Services", "Python", "Java",
            "JavaScript", "C#", "C++", "Go", "Ruby", "PHP", "TypeScript", "Shell Scripting"
        ],
        "Data Scientist": [
            "Statistical Analysis", "Hypothesis Testing", "Regression Analysis", 
            "Classification", "Clustering", "Dimensionality Reduction", 
            "Natural Language Processing", "Computer Vision", "Feature Engineering", 
            "Data Cleaning", "Data Visualization", "Exploratory Data Analysis", 
            "A/B Testing", "Experiment Design", "Pandas", "NumPy", "SciPy", 
            "Scikit-learn", "TensorFlow", "PyTorch", "R", "Tableau", "Power BI", 
            "SQL", "Big Data", "Spark", "Hadoop", "Data Pipelines", "Python",
            "Statistical Modeling", "Bayesian Methods", "Time Series Analysis",
            "Machine Learning", "Deep Learning", "Data Mining", "Data Warehousing",
            "ETL Processes", "Model Validation", "Model Deployment"
        ],
        "Machine Learning Engineer": [
            "Deep Learning", "Neural Networks", "Reinforcement Learning", 
            "Model Deployment", "MLOps", "Model Monitoring", "Feature Stores", 
            "Model Versioning", "Hyperparameter Tuning", "Cross-Validation", 
            "Model Evaluation", "Model Optimization", "Docker", "Kubernetes", 
            "Data Pipelines", "Cloud ML Services", "GPU Programming", "CUDA", 
            "Distributed Training", "ML Frameworks", "Model Serving", 
            "Model Compression", "TensorRT", "ONNX", "ML Interpretability",
            "Python", "TensorFlow", "PyTorch", "Scikit-learn", "Feature Engineering",
            "Algorithm Selection", "Model Debugging", "Ensemble Methods", "Transfer Learning",
            "NLP", "Computer Vision", "Time Series Forecasting", "Anomaly Detection"
        ],
        "Cybersecurity Analyst": [
            "Network Security", "Vulnerability Assessment", "Penetration Testing", 
            "SIEM", "Security Protocols", "Encryption", "Firewalls", "IDS/IPS", 
            "Malware Analysis", "Security Compliance", "Risk Management", 
            "Incident Response", "Ethical Hacking", "Forensic Analysis", "OSINT", 
            "Security Frameworks", "OWASP", "Threat Intelligence", 
            "Security Auditing", "Security Tools", "Wireshark", "Kali Linux", 
            "CISSP", "NIST Frameworks", "ISO 27001", "Security Architecture",
            "Access Control", "Authentication Systems", "Authorization Systems",
            "Secure Coding Practices", "Cloud Security", "Network Protocols",
            "Social Engineering", "Secure Configuration", "Log Analysis"
        ],
        "Cloud Architect": [
            "AWS", "Azure", "Google Cloud Platform", "Cloud Migration", 
            "Infrastructure as Code", "Terraform", "CloudFormation", "Docker", 
            "Kubernetes", "Microservices", "Serverless Architecture", "DevOps", 
            "CI/CD", "Networking", "Security", "Load Balancing", "Auto Scaling", 
            "High Availability", "Disaster Recovery", "Multi-Cloud Strategy", 
            "Cost Optimization", "Cloud Security", "AWS Solutions Architect", 
            "Azure Solutions Architect", "Container Orchestration", "Service Mesh", 
            "Cloud Automation", "Architecture Design", "System Integration",
            "Performance Optimization", "Monitoring", "Logging", "API Gateway",
            "Identity and Access Management", "Database Services", "Storage Solutions"
        ],
        "UI/UX Designer": [
            "Figma", "Adobe XD", "Sketch", "InVision", "Wireframing", "Prototyping", 
            "User Research", "Usability Testing", "Information Architecture", 
            "User-Centered Design", "Design Systems", "Typography", "Color Theory", 
            "Adobe Illustrator", "Adobe Photoshop", "UI Components", 
            "Interaction Design", "Responsive Design", "Accessibility", "User Flows", 
            "A/B Testing", "Material Design", "Design Thinking", "User Personas",
            "Journey Mapping", "UX Writing", "Mobile Design Patterns", "UX Research",
            "Visual Design", "Animation", "UI Patterns", "Design Critique",
            "Card Sorting", "Heuristic Evaluation", "Gestalt Principles"
        ],
        "DevOps Engineer": [
            "Continuous Integration", "Continuous Deployment", "Infrastructure as Code", 
            "Configuration Management", "Container Orchestration", "Monitoring", 
            "Logging", "Alerting", "Performance Optimization", "Site Reliability", 
            "Jenkins", "GitHub Actions", "CircleCI", "Ansible", "Puppet", "Chef", 
            "Terraform", "CloudFormation", "ELK Stack", "Prometheus", "Grafana", 
            "Service Mesh", "Network Automation", "Security Automation",
            "Kubernetes", "Docker", "Linux Administration", "Shell Scripting",
            "Python", "Go", "AWS", "Azure", "GCP", "Load Testing", "Chaos Engineering",
            "Incident Management", "Version Control", "Git", "CI/CD Pipelines"
        ],
        "Full-Stack Developer": [
            "HTML", "CSS", "JavaScript", "TypeScript", "Front-end Frameworks", 
            "React", "Angular", "Vue.js", "Node.js", "Express.js", "Backend Frameworks", 
            "Django", "Flask", "Spring", "Ruby on Rails", "Laravel", "Database Design", 
            "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "RESTful APIs", 
            "GraphQL", "Authentication", "JWT", "OAuth", "Authorization", "AWS", 
            "Heroku", "Netlify", "Vercel", "Responsive Design", "Mobile-First Design", 
            "Testing", "Jest", "Mocha", "Cypress", "DevOps", "Git", "CI/CD"
        ],
        "Mobile App Developer": [
            "iOS Development", "Swift", "SwiftUI", "UIKit", "Android Development", 
            "Kotlin", "Java", "Flutter", "React Native", "Xamarin", "Mobile UI Design", 
            "Mobile UX", "Mobile Architecture", "Offline Storage", "Push Notifications", 
            "Mobile Authentication", "App Performance", "Mobile Testing", "Mobile Security", 
            "App Store Deployment", "Google Play Deployment", "App Analytics", 
            "Responsive Design", "Cross-platform Development", "Native APIs", 
            "Mobile Frameworks", "Mobile Backend as a Service", "Firebase", 
            "App Lifecycle Management", "Mobile Navigation Patterns", "Mobile Gestures", 
            "Animation", "Mobile Accessibility", "Mobile Payments", "Local Storage"
        ],
        "AI Research Scientist": [
            "Machine Learning", "Deep Learning", "Neural Networks", "Reinforcement Learning", 
            "Computer Vision", "Natural Language Processing", "Research Methodology", 
            "Academic Writing", "Paper Publication", "Literature Review", "Mathematics", 
            "Statistics", "Linear Algebra", "Calculus", "Probability Theory", 
            "TensorFlow", "PyTorch", "JAX", "Algorithm Design", "Experimentation", 
            "Transformers", "GANs", "RL Algorithms", "Data Analysis", "Ethics in AI", 
            "Model Interpretability", "Conference Presentations", "Peer Review", 
            "Hypothesis Testing", "Research Proposal Writing", "Collaborative Research", 
            "Graph Neural Networks", "Few-shot Learning", "Representation Learning"
        ]
    },
    "Marketing": {
        "Digital Marketing Specialist": [
            "SEO", "SEM", "Google Ads", "Meta Ads", "Social Media Marketing", 
            "Content Marketing", "Email Marketing", "Marketing Automation", 
            "Conversion Rate Optimization", "Google Analytics", "A/B Testing", 
            "Landing Page Optimization", "Copywriting", "Campaign Management", 
            "Keyword Research", "Competitor Analysis", "Marketing Metrics", 
            "User Acquisition", "Retention Strategy", "Marketing Funnels", 
            "Customer Journey Mapping", "Marketing Analytics", "Paid Media", 
            "Organic Traffic Growth", "Marketing Platforms", "HubSpot", "Mailchimp", 
            "Marketo", "Google Tag Manager", "UTM Parameters", "Attribution Modeling", 
            "Digital Marketing Strategy", "ROI Analysis", "Facebook Pixel"
        ],
        "Brand Manager": [
            "Brand Strategy", "Brand Development", "Brand Positioning", "Brand Identity", 
            "Brand Guidelines", "Brand Voice", "Brand Messaging", "Brand Architecture", 
            "Market Research", "Competitive Analysis", "Consumer Insights", 
            "Product Marketing", "Marketing Campaigns", "Campaign Management", 
            "Budget Management", "Team Leadership", "Creative Direction", 
            "Advertising", "Media Planning", "Content Strategy", "Brand Partnerships", 
            "Brand Awareness", "Brand Equity", "Brand Valuation", "Brand Health Metrics", 
            "Stakeholder Management", "Cross-functional Collaboration", 
            "Presentation Skills", "Brand Storytelling", "Customer Loyalty Programs"
        ],
        "Market Research Analyst": [
            "Quantitative Research", "Qualitative Research", "Survey Design", 
            "Focus Groups", "In-depth Interviews", "Statistical Analysis", "Data Collection", 
            "Data Analysis", "Data Visualization", "Report Writing", "Presentation Skills", 
            "Market Segmentation", "Market Sizing", "Competitive Intelligence", 
            "Consumer Behavior Analysis", "Trend Analysis", "SPSS", "SAS", "R", 
            "STATA", "SQL", "Tableau", "Power BI", "Excel", "Research Methodology", 
            "Sampling Methods", "Research Ethics", "Market Forecasting", 
            "Hypothesis Testing", "Correlation Analysis", "Regression Analysis", 
            "Factor Analysis", "Conjoint Analysis", "Sentiment Analysis", "Python", 
            "Web Scraping", "Text Mining", "Social Media Analytics"
        ],
        "Content Strategist": [
            "Content Planning", "Content Calendar", "Content Creation", "Content Editing", 
            "Editorial Guidelines", "Style Guides", "SEO Content", "Content Distribution", 
            "Content Performance Analysis", "Content Auditing", "Audience Research", 
            "Persona Development", "Storytelling", "Copywriting", "Content Marketing", 
            "Brand Voice Development", "Content Management Systems", "WordPress", 
            "Drupal", "Content Governance", "Content Templates", "Information Architecture", 
            "Content Taxonomy", "Content Workflow", "Content Collaboration Tools", 
            "Content Analytics", "Content Optimization", "Content ROI", "Content Strategy Documentation", 
            "Content Gap Analysis", "UX Writing", "Social Media Content", "Video Content Strategy", 
            "Podcast Strategy", "Email Content Strategy"
        ],
        "Social Media Manager": [
            "Social Media Strategy", "Content Creation", "Community Management", 
            "Social Media Analytics", "Platform Knowledge (Facebook, Instagram, Twitter, LinkedIn, TikTok)", 
            "Social Media Advertising", "Campaign Management", "Influencer Marketing", 
            "Crisis Management", "Social Listening", "Audience Engagement", "Copywriting", 
            "Visual Content Creation", "Video Editing", "Graphic Design Basics", 
            "Social Media Tools (Hootsuite, Buffer, Sprout Social)", "Content Calendar Management", 
            "Trend Spotting", "Brand Voice", "Social Media Policies", "Community Guidelines", 
            "Contest Management", "Social Commerce", "User-generated Content", 
            "Social Media Audits", "Reporting", "PR Basics", "Cross-platform Strategy", 
            "Social Media Compliance", "Hashtag Strategy", "Engagement Metrics"
        ]
    },
    "Finance": {
        "Investment Banker": [
            "Financial Modeling", "Valuation Methods", "Mergers & Acquisitions", 
            "IPO Process", "Capital Markets", "Financial Analysis", "Due Diligence", 
            "Pitch Book Creation", "Excel", "PowerPoint", "Financial Statement Analysis", 
            "Industry Analysis", "Market Analysis", "Deal Structuring", "Transaction Execution", 
            "Client Relationship Management", "Negotiation", "Corporate Finance", 
            "Leveraged Buyouts", "Debt Financing", "Equity Financing", "Discounted Cash Flow Analysis", 
            "Comparable Company Analysis", "Precedent Transaction Analysis", "Deal Sourcing", 
            "Financial Projections", "Sensitivity Analysis", "Scenario Analysis", 
            "Investment Memorandum", "Bloomberg Terminal", "Capital IQ", "Factset", "Dealogic", "Dataroom Management"
        ],
        "Portfolio Manager": [
            "Asset Allocation", "Investment Strategy", "Risk Management", "Performance Analysis", 
            "Market Research", "Economic Analysis", "Financial Analysis", "Portfolio Construction", 
            "Fund Management", "Client Relationship Management", "Investment Policy Statements", 
            "Quantitative Analysis", "Investment Research", "Equities", "Fixed Income", 
            "Alternative Investments", "Derivatives", "Bloomberg Terminal", "Excel", 
            "Python", "R", "MATLAB", "CFA Designation", "Portfolio Optimization", 
            "Attribution Analysis", "Risk Metrics", "Sharpe Ratio", "Alpha Generation", 
            "Beta Analysis", "Stress Testing", "VaR", "Asset-Liability Management", 
            "Regulatory Compliance", "Investment Committee Presentations", "Financial Modeling"
        ],
        "Risk Manager": [
            "Risk Assessment", "Risk Mitigation", "Risk Control", "Risk Reporting", 
            "Market Risk", "Credit Risk", "Operational Risk", "Liquidity Risk", 
            "Regulatory Risk", "Compliance", "Basel Frameworks", "Stress Testing", 
            "Scenario Analysis", "Value at Risk (VaR)", "Excel", "SAS", "R", "Python", 
            "SQL", "Risk Management Systems", "Quantitative Analysis", "Statistical Modeling", 
            "Financial Analysis", "Regulatory Reporting", "Risk Documentation", 
            "Risk Governance", "Enterprise Risk Management", "Risk Appetite Frameworks", 
            "Key Risk Indicators", "Risk Control Self-Assessment", "Audit Management", 
            "Business Continuity Planning", "Disaster Recovery", "Risk Training", 
            "Incident Management", "Fraud Detection", "AML Compliance"
        ],
        "Financial Analyst": [
            "Financial Modeling", "Financial Statement Analysis", "Budgeting", 
            "Forecasting", "Variance Analysis", "Excel", "PowerPoint", "Tableau", 
            "Power BI", "SQL", "Data Analysis", "Financial Reporting", "Industry Research", 
            "Company Valuation", "DCF Analysis", "Ratio Analysis", "Trend Analysis", 
            "Cost Analysis", "Capital Expenditure Analysis", "Working Capital Analysis", 
            "Profitability Analysis", "Cash Flow Analysis", "Sensitivity Analysis", 
            "Scenario Analysis", "Break-even Analysis", "ROI Analysis", "Financial Planning", 
            "Investment Analysis", "Financial Presentations", "Financial Decision Support", 
            "ERP Systems", "Financial Databases", "Accounting Principles", "Corporate Finance", 
            "Treasury Operations", "Financial Controls"
        ],
        "Actuary": [
            "Statistical Analysis", "Risk Assessment", "Insurance Pricing", 
            "Reserve Calculations", "Mortality Tables", "Life Contingencies", 
            "Pension Valuations", "Financial Mathematics", "Probability Theory", 
            "Actuarial Modeling", "Insurance Product Development", "Actuarial Valuation", 
            "Excel", "R", "Python", "SAS", "SQL", "VBA", "Actuarial Software", 
            "Regulatory Reporting", "Financial Reporting", "Solvency II", "IFRS 17", 
            "Asset-Liability Management", "Experience Studies", "Credibility Theory", 
            "Stochastic Modeling", "Investment Strategy", "Reinsurance Analysis", 
            "Underwriting", "Claims Analysis", "Pricing Strategy", "Actuarial Exams", 
            "Predictive Modeling", "Capital Modeling", "Profitability Analysis"
        ]
    },
    "Human Resources": {
        "HR Manager": [
            "Employee Relations", "Performance Management", "Talent Management", 
            "Workforce Planning", "HR Strategy", "HR Policies", "HR Compliance", 
            "HRIS Systems", "Benefits Administration", "Compensation Management", 
            "Recruitment Strategy", "Employee Engagement", "Training & Development", 
            "Succession Planning", "Labor Relations", "Change Management", 
            "Organizational Development", "HR Analytics", "Employee Lifecycle Management", 
            "HR Budget Management", "Employee Communication", "Leadership Development", 
            "Diversity & Inclusion", "Employee Wellness Programs", "HR Best Practices", 
            "Conflict Resolution", "Workplace Culture", "Employee Surveys", 
            "Exit Interviews", "HR Project Management", "HR Vendor Management", 
            "HR Technology Implementation", "Employment Law", "Talent Acquisition Strategy"
        ],
        "Talent Acquisition Specialist": [
            "Recruitment Strategy", "Sourcing Techniques", "Job Description Development", 
            "Candidate Screening", "Interviewing", "Applicant Tracking Systems", 
            "Recruitment Marketing", "Employer Branding", "Social Media Recruiting", 
            "LinkedIn Recruiting", "Job Boards", "Candidate Relationship Management", 
            "Talent Pipeline Development", "Recruitment Analytics", "Diversity Recruiting", 
            "Campus Recruiting", "Technical Recruiting", "Executive Recruiting", 
            "Assessment Tools", "Pre-employment Testing", "Background Checks", 
            "Offer Negotiation", "Onboarding Coordination", "Recruitment Events", 
            "Recruitment Compliance", "Candidate Experience", "Boolean Search", 
            "Recruitment Metrics", "Cost-per-hire Analysis", "Time-to-fill Analysis", 
            "Recruitment Budget Management", "Recruitment Communication", "Resume Screening", 
            "Job Fair Organization", "Hiring Manager Relationships"
        ],
        "Learning and Development Specialist": [
            "Training Needs Analysis", "Instructional Design", "Curriculum Development", 
            "Training Delivery", "Workshop Facilitation", "E-learning Development", 
            "LMS Administration", "Training Evaluation", "Adult Learning Theory", 
            "Microlearning", "Blended Learning", "Virtual Training", "Coaching", 
            "Mentoring Programs", "Leadership Development", "Onboarding Programs", 
            "Skills Assessment", "Learning Analytics", "Training ROI", "Learning Paths", 
            "Competency Frameworks", "Knowledge Management", "Content Creation", 
            "Training Materials", "Learning Experience Design", "Performance Support", 
            "Training Budget Management", "Vendor Management", "Learning Technology", 
            "Certification Programs", "Compliance Training", "Soft Skills Training", 
            "Technical Training", "Professional Development Programs", "Train-the-Trainer"
        ],
        "Compensation and Benefits Analyst": [
            "Compensation Structure", "Salary Benchmarking", "Market Pricing", 
            "Job Evaluation", "Salary Surveys", "Benefits Administration", 
            "Incentive Programs", "Bonus Plans", "Equity Compensation", "Compensation Strategy", 
            "Total Rewards Strategy", "HRIS Systems", "Payroll Systems", "Benefits Analysis", 
            "Healthcare Benefits", "Retirement Plans", "Paid Time Off Policies", 
            "Leave Administration", "Compensation Analytics", "Benefits Compliance", 
            "FLSA Compliance", "ACA Compliance", "ERISA Compliance", "Executive Compensation", 
            "Sales Compensation", "Merit Planning", "Compensation Communications", 
            "Benefits Enrollment", "Benefits Cost Analysis", "Compensation Metrics", 
            "Pay Equity Analysis", "Global Compensation", "Benefits Vendor Management", 
            "Compensation Surveys", "Financial Modeling"
        ],
        "Employee Relations Specialist": [
            "Employee Counseling", "Conflict Resolution", "Policy Interpretation", 
            "Workplace Investigations", "Performance Management", "Disciplinary Actions", 
            "Grievance Handling", "Employee Communications", "Employee Engagement", 
            "Culture Development", "Exit Interviews", "Employee Surveys", "Legal Compliance", 
            "Employment Law", "Documentation Management", "Mediation", "Employee Advocacy", 
            "HR Best Practices", "Diversity & Inclusion", "Change Management", 
            "Organizational Development", "Leadership Coaching", "Team Building", 
            "Employee Recognition Programs", "Workplace Behavior Analysis", 
            "HR Policy Development", "Harassment Prevention", "Case Management", 
            "Labor Relations", "Union Relations", "Collective Bargaining", 
            "Employee Handbook Development", "Conflict Management", "Whistleblower Programs", 
            "Workplace Ethics"
        ]
    }
}

def generate_specialization_skills():
    """
    Generate a comprehensive specialization_skills.json file with
    skills for various specializations across multiple domains.
    """
    # Create output directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create the output dictionary
    specialization_skills = {}
    
    # Populate specialization skills from defined domains
    for domain, specializations in DOMAINS.items():
        for specialization, skills in specializations.items():
            specialization_skills[specialization] = skills
    
    # Save to file
    output_path = os.path.join(data_dir, "specialization_skills.json")
    with open(output_path, 'w') as f:
        json.dump(specialization_skills, f, indent=4)
    
    print(f"Generated specialization skills for {len(specialization_skills)} specializations in {len(DOMAINS)} domains")
    print(f"Saved to {output_path}")
    
    return specialization_skills

if __name__ == "__main__":
    generate_specialization_skills() 