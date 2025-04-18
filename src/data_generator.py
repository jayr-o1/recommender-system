import json
import os
import random
from typing import Dict, List, Any

class DataGenerator:
    """
    Generates sample data for the career recommender system:
    - Fields (broad career areas)
    - Specializations (specific roles within fields)
    - Core skills for each specialization with weights
    """
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data generator
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define skill categories for reuse
        self.programming_languages = [
            "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", 
            "TypeScript", "PHP", "Swift", "Kotlin", "Ruby"
        ]
        
        self.data_skills = [
            "SQL", "Data Analysis", "Data Visualization", "Data Modeling",
            "ETL", "Data Warehousing", "Big Data", "Data Mining",
            "Tableau", "Power BI", "Excel", "Statistics"
        ]
        
        self.ml_skills = [
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "scikit-learn", "Neural Networks",
            "Reinforcement Learning", "Feature Engineering", "Model Deployment"
        ]
        
        self.web_skills = [
            "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js",
            "REST APIs", "GraphQL", "Responsive Design", "UI/UX", "Web Security"
        ]
        
        self.soft_skills = [
            "Communication", "Problem Solving", "Project Management", 
            "Teamwork", "Leadership", "Time Management", "Critical Thinking",
            "Adaptability", "Creativity", "Attention to Detail"
        ]
        
        self.cloud_skills = [
            "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
            "Cloud Architecture", "Serverless", "DevOps", "CI/CD",
            "Infrastructure as Code"
        ]
        
        # Add new skill categories for enhanced specializations
        self.finance_skills = [
            "Financial Analysis", "Investment Management", "Risk Assessment",
            "Portfolio Management", "Financial Modeling", "Accounting",
            "Budgeting", "Financial Reporting", "Tax Planning", "Valuation"
        ]
        
        self.healthcare_skills = [
            "Patient Care", "Clinical Assessment", "Medical Terminology", 
            "Healthcare Compliance", "Electronic Health Records", "Medical Coding",
            "Patient Education", "Medical Research", "Vital Signs Monitoring",
            "Treatment Planning", "Infection Control", "Diagnostic Procedures"
        ]
        
        self.art_design_skills = [
            "Graphic Design", "UI Design", "UX Design", "Photography", 
            "3D Modeling", "Animation", "Video Editing", "Illustration", 
            "Typography", "Color Theory", "Branding", "Creative Direction",
            "Adobe Creative Suite", "Figma", "Sketch"
        ]
        
        self.marketing_skills = [
            "Digital Marketing", "Content Marketing", "Social Media Marketing",
            "SEO", "SEM", "Email Marketing", "Marketing Analytics", "Brand Management",
            "Market Research", "Campaign Management", "CRM", "Customer Acquisition",
            "Copywriting", "Public Relations", "Growth Hacking"
        ]
        
        self.science_skills = [
            "Laboratory Techniques", "Experimental Design", "Scientific Writing",
            "Research Methods", "Data Collection", "Hypothesis Testing",
            "Statistical Analysis", "Microscopy", "Spectroscopy", "Chemical Analysis",
            "Genetic Analysis", "Field Research", "Literature Review"
        ]
        
        self.manufacturing_skills = [
            "Quality Control", "Lean Manufacturing", "Six Sigma", "Supply Chain Management",
            "Inventory Management", "Process Improvement", "Production Planning",
            "Equipment Maintenance", "Safety Protocols", "Regulatory Compliance",
            "CAD/CAM", "Industrial Design", "Automation"
        ]
        
    def generate_fields(self) -> Dict[str, Any]:
        """
        Generate career fields with core skills
        
        Returns:
            Dictionary of fields with their core skills
        """
        fields = {
            "Computer Science": {
                "description": "Field covering the theory and practice of computer technology and software development",
                "core_skills": {
                    "Programming": 90,
                    "Data Structures": 80,
                    "Algorithms": 80,
                    "Software Design": 70,
                    "Problem Solving": 85
                }
            },
            "Business Accountancy": {
                "description": "Field focusing on financial reporting, auditing and business accounting practices",
                "core_skills": {
                    "Financial Analysis": 90,
                    "Auditing": 85,
                    "Bookkeeping": 80,
                    "Tax Knowledge": 85,
                    "Business Ethics": 75
                }
            },
            "Engineering": {
                "description": "Field applying scientific and mathematical principles to design and build systems and structures",
                "core_skills": {
                    "Mathematics": 90,
                    "Problem Solving": 85,
                    "Technical Design": 85,
                    "Critical Thinking": 80,
                    "Analysis": 80
                }
            },
            "Law": {
                "description": "Field focused on legal systems, regulations, and advocacy",
                "core_skills": {
                    "Legal Research": 90,
                    "Critical Thinking": 85,
                    "Communication": 85,
                    "Analysis": 85,
                    "Ethics": 80
                }
            },
            "Criminology": {
                "description": "Study of crime, criminal behavior, and law enforcement systems",
                "core_skills": {
                    "Criminal Justice": 85,
                    "Research": 80,
                    "Analysis": 85,
                    "Psychology": 75,
                    "Ethics": 80
                }
            },
            "Nursing": {
                "description": "Healthcare field focused on patient care and medical support",
                "core_skills": {
                    "Patient Care": 90,
                    "Medical Knowledge": 85,
                    "Communication": 85,
                    "Critical Thinking": 80,
                    "Empathy": 85
                }
            },
            "Medicine": {
                "description": "Field focused on diagnosing, treating, and preventing diseases and injuries",
                "core_skills": {
                    "Medical Knowledge": 95,
                    "Diagnosis": 90,
                    "Patient Care": 85,
                    "Critical Thinking": 85,
                    "Ethics": 80
                }
            },
            "Hospitality Management": {
                "description": "Field focused on management of hotels, restaurants, and other service-oriented businesses",
                "core_skills": {
                    "Customer Service": 90,
                    "Management": 85,
                    "Communication": 85,
                    "Organization": 80,
                    "Problem Solving": 75
                }
            },
            "Tourism": {
                "description": "Field focused on travel, tourism services, and destination management",
                "core_skills": {
                    "Customer Service": 85,
                    "Geography": 80,
                    "Cultural Awareness": 85,
                    "Communication": 85,
                    "Organization": 75
                }
            },
            "Psychology": {
                "description": "Study of the human mind and behavior",
                "core_skills": {
                    "Research": 85,
                    "Analysis": 85,
                    "Communication": 80,
                    "Empathy": 85,
                    "Critical Thinking": 80
                }
            },
            "Medical Technology": {
                "description": "Field focused on laboratory testing and diagnostics in healthcare",
                "core_skills": {
                    "Laboratory Skills": 90,
                    "Analytical Thinking": 85,
                    "Attention to Detail": 90,
                    "Medical Knowledge": 85,
                    "Technical Skills": 80
                }
            },
            "Research": {
                "description": "Field focused on systematic investigation to establish facts and develop new knowledge",
                "core_skills": {
                    "Research Methods": 90,
                    "Data Analysis": 85,
                    "Critical Thinking": 85,
                    "Technical Writing": 80,
                    "Problem Solving": 80
                }
            },
            "Education": {
                "description": "Field focused on teaching, learning, and curriculum development",
                "core_skills": {
                    "Communication": 90,
                    "Subject Knowledge": 85,
                    "Lesson Planning": 85,
                    "Assessment": 80,
                    "Empathy": 80
                }
            },
            "Data Science": {
                "description": "Field focused on extracting knowledge and insights from structured and unstructured data",
                "core_skills": {
                    "Machine Learning": 90,
                    "Statistics": 85,
                    "Data Analysis": 90,
                    "Programming": 80,
                    "Data Visualization": 85
                }
            },
            "Cybersecurity": {
                "description": "Field focused on protecting systems, networks, and programs from digital attacks",
                "core_skills": {
                    "Network Security": 90,
                    "Threat Analysis": 85,
                    "Security Protocols": 85,
                    "Penetration Testing": 80,
                    "Cryptography": 75
                }
            },
            "Digital Marketing": {
                "description": "Field using digital channels to promote products and services",
                "core_skills": {
                    "SEO": 85,
                    "Social Media Marketing": 90,
                    "Content Creation": 85,
                    "Analytics": 80,
                    "Marketing Strategy": 85
                }
            },
            "Finance": {
                "description": "Field managing money, investments, and other financial assets",
                "core_skills": {
                    "Financial Analysis": 90,
                    "Investment Management": 85,
                    "Risk Assessment": 85,
                    "Financial Planning": 80,
                    "Market Analysis": 85
                }
            },
            "Environmental Science": {
                "description": "Field studying the environment and solutions to environmental problems",
                "core_skills": {
                    "Environmental Analysis": 90,
                    "Research Methods": 85,
                    "Data Collection": 85,
                    "Environmental Policy": 80,
                    "Sustainability": 85
                }
            },
            "Graphic Design": {
                "description": "Field creating visual content to communicate messages",
                "core_skills": {
                    "Visual Design": 90,
                    "Typography": 85,
                    "Color Theory": 85,
                    "Adobe Creative Suite": 85,
                    "Layout Design": 80
                }
            },
            "UX/UI Design": {
                "description": "Field focused on creating meaningful and relevant experiences for users",
                "core_skills": {
                    "User Research": 90,
                    "Wireframing": 85,
                    "Prototyping": 85,
                    "Interaction Design": 85,
                    "Usability Testing": 80
                }
            },
            "Human Resources": {
                "description": "Field managing the human capital of an organization",
                "core_skills": {
                    "Recruitment": 85,
                    "Employee Relations": 85,
                    "Performance Management": 80,
                    "HR Policies": 80,
                    "Training and Development": 85
                }
            },
            "Supply Chain Management": {
                "description": "Field overseeing the flow of goods, services, and information from raw materials to consumers",
                "core_skills": {
                    "Logistics Management": 90,
                    "Inventory Control": 85,
                    "Procurement": 85,
                    "Supply Chain Optimization": 85,
                    "Demand Planning": 80
                }
            },
            "Biotechnology": {
                "description": "Field using biological systems and organisms to develop products",
                "core_skills": {
                    "Laboratory Techniques": 90,
                    "Molecular Biology": 85,
                    "Research Methods": 85,
                    "Data Analysis": 80,
                    "Experimental Design": 85
                }
            },
            "Architecture": {
                "description": "Field designing buildings and other physical structures",
                "core_skills": {
                    "Architectural Design": 90,
                    "3D Modeling": 85,
                    "Building Codes": 80,
                    "Technical Drawing": 85,
                    "Project Management": 75
                }
            },
            "Journalism": {
                "description": "Field gathering, assessing, creating, and presenting news and information",
                "core_skills": {
                    "Reporting": 90,
                    "Writing": 90,
                    "Research": 85,
                    "Interviewing": 85,
                    "Media Ethics": 80
                }
            }
        }
        
        # Save to file
        with open(os.path.join(self.output_dir, "fields.json"), "w") as f:
            json.dump(fields, f, indent=4)
            
        return fields
    
    def generate_specializations(self) -> Dict[str, Any]:
        """
        Generate specializations with their fields and core skills
        
        Returns:
            Dictionary of specializations with their core skills and weights
        """
        specializations = {
            "Software Engineer": {
                "field": "Computer Science",
                "description": "Designs, develops, and maintains software systems",
                "core_skills": {
                    "Python": 80,
                    "Java": 75,
                    "JavaScript": 70,
                    "Data Structures": 85,
                    "Algorithms": 85,
                    "Software Design": 80,
                    "Testing": 75,
                    "Version Control": 70
                }
            },
            "Web Developer": {
                "field": "Computer Science",
                "description": "Creates and maintains websites",
                "core_skills": {
                    "HTML": 95,
                    "CSS": 95,
                    "JavaScript": 90,
                    "React": 80,
                    "Node.js": 75,
                    "Responsive Design": 85,
                    "Web Security": 70
                }
            },
            "Mobile App Developer": {
                "field": "Computer Science",
                "description": "Develops applications for mobile devices",
                "core_skills": {
                    "Swift": 85,
                    "Kotlin": 85,
                    "React Native": 80,
                    "Flutter": 80,
                    "Mobile UI Design": 75,
                    "API Integration": 70,
                    "App Store Deployment": 65
                }
            },
            "DevOps Engineer": {
                "field": "Computer Science",
                "description": "Implements and manages continuous delivery systems and methodologies",
                "core_skills": {
                    "CI/CD": 90,
                    "Docker": 85,
                    "Kubernetes": 85,
                    "Cloud Services": 80,
                    "Shell Scripting": 75,
                    "Infrastructure as Code": 80,
                    "Monitoring Tools": 70
                }
            },
            "Cloud Architect": {
                "field": "Computer Science",
                "description": "Designs and oversees cloud computing infrastructure",
                "core_skills": {
                    "AWS": 90,
                    "Azure": 85,
                    "Google Cloud": 85,
                    "Cloud Security": 80,
                    "Networking": 75,
                    "Distributed Systems": 80,
                    "Serverless Architecture": 75
                }
            },
            "Data Scientist": {
                "field": "Data Science",
                "description": "Uses advanced analytics and statistical methods to interpret data",
                "core_skills": {
                    "Python": 85,
                    "R": 70,
                    "Machine Learning": 90,
                    "Statistics": 85,
                    "Data Analysis": 90,
                    "Big Data": 75,
                    "Data Visualization": 80
                }
            },
            "Data Engineer": {
                "field": "Data Science",
                "description": "Builds and maintains data pipelines and infrastructure",
                "core_skills": {
                    "SQL": 90,
                    "ETL Processes": 85,
                    "Data Warehousing": 85,
                    "Big Data Technologies": 80,
                    "Python": 80,
                    "Data Modeling": 85,
                    "Cloud Platforms": 75
                }
            },
            "Machine Learning Engineer": {
                "field": "Data Science",
                "description": "Designs and implements machine learning models and systems",
                "core_skills": {
                    "Machine Learning Algorithms": 90,
                    "Deep Learning": 85,
                    "Python": 85,
                    "TensorFlow/PyTorch": 80,
                    "Feature Engineering": 80,
                    "Model Deployment": 75,
                    "Data Processing": 80
                }
            },
            "Business Intelligence Analyst": {
                "field": "Data Science",
                "description": "Analyzes data to improve business decision-making",
                "core_skills": {
                    "SQL": 85,
                    "Data Visualization": 90,
                    "Tableau/Power BI": 85,
                    "Business Analysis": 80,
                    "Statistical Analysis": 75,
                    "Reporting": 80,
                    "Data Modeling": 70
                }
            },
            "Security Analyst": {
                "field": "Cybersecurity",
                "description": "Monitors and protects systems from cybersecurity threats",
                "core_skills": {
                    "Threat Detection": 90,
                    "Security Tools": 85,
                    "Network Security": 85,
                    "Incident Response": 80,
                    "Vulnerability Assessment": 85,
                    "Security Monitoring": 80,
                    "Risk Analysis": 75
                }
            },
            "Penetration Tester": {
                "field": "Cybersecurity",
                "description": "Tests security by attempting to breach systems and networks",
                "core_skills": {
                    "Ethical Hacking": 90,
                    "Network Exploitation": 85,
                    "Social Engineering": 80,
                    "Security Tools": 85,
                    "Vulnerability Research": 80,
                    "Reporting": 75,
                    "Programming": 70
                }
            },
            "Information Security Manager": {
                "field": "Cybersecurity",
                "description": "Oversees organization's information security strategy",
                "core_skills": {
                    "Security Policy": 90,
                    "Risk Management": 85,
                    "Compliance": 85,
                    "Security Architecture": 80,
                    "Team Leadership": 85,
                    "Incident Management": 80,
                    "Security Awareness": 75
                }
            },
            "Digital Marketing Specialist": {
                "field": "Digital Marketing",
                "description": "Implements digital marketing strategies across platforms",
                "core_skills": {
                    "SEO": 85,
                    "PPC Advertising": 80,
                    "Social Media Marketing": 90,
                    "Content Marketing": 85,
                    "Analytics": 80,
                    "Email Marketing": 75,
                    "Marketing Automation": 70
                }
            },
            "SEO Specialist": {
                "field": "Digital Marketing",
                "description": "Optimizes websites for better search engine visibility",
                "core_skills": {
                    "Keyword Research": 90,
                    "On-Page SEO": 85,
                    "Off-Page SEO": 85,
                    "SEO Tools": 80,
                    "Content Optimization": 85,
                    "Analytics": 80,
                    "Link Building": 75
                }
            },
            "Social Media Manager": {
                "field": "Digital Marketing",
                "description": "Manages brand presence and engagement on social media platforms",
                "core_skills": {
                    "Content Creation": 90,
                    "Community Management": 85,
                    "Social Media Platforms": 90,
                    "Analytics": 80,
                    "Campaign Management": 85,
                    "Brand Voice": 80,
                    "Audience Growth": 75
                }
            },
            "Accountant": {
                "field": "Business Accountancy",
                "description": "Prepares and examines financial records",
                "core_skills": {
                    "Financial Analysis": 90,
                    "Bookkeeping": 85,
                    "Tax Knowledge": 80,
                    "Attention to Detail": 90,
                    "Excel": 80,
                    "Business Ethics": 75
                }
            },
            "Auditor": {
                "field": "Business Accountancy",
                "description": "Examines financial statements and records for accuracy and compliance",
                "core_skills": {
                    "Auditing": 95,
                    "Financial Analysis": 85,
                    "Risk Assessment": 80,
                    "Attention to Detail": 90,
                    "Business Ethics": 85,
                    "Communication": 75
                }
            },
            "Financial Analyst": {
                "field": "Finance",
                "description": "Analyzes financial data to guide business decisions",
                "core_skills": {
                    "Financial Modeling": 90,
                    "Data Analysis": 85,
                    "Excel": 90,
                    "Financial Reporting": 85,
                    "Forecasting": 80,
                    "Valuation": 85,
                    "Business Acumen": 80
                }
            },
            "Investment Banker": {
                "field": "Finance",
                "description": "Assists clients with raising capital and financial transactions",
                "core_skills": {
                    "Financial Analysis": 90,
                    "Valuation": 90,
                    "Deal Structuring": 85,
                    "Negotiation": 85,
                    "Financial Modeling": 85,
                    "Client Management": 80,
                    "Market Knowledge": 85
                }
            },
            "Portfolio Manager": {
                "field": "Finance",
                "description": "Manages investment portfolios to meet client goals",
                "core_skills": {
                    "Asset Allocation": 90,
                    "Investment Analysis": 90,
                    "Risk Management": 85,
                    "Market Research": 85,
                    "Performance Reporting": 80,
                    "Client Communication": 80,
                    "Financial Planning": 75
                }
            },
            "Civil Engineer": {
                "field": "Engineering",
                "description": "Designs and supervises construction of infrastructure projects",
                "core_skills": {
                    "Mathematics": 85,
                    "Technical Design": 90,
                    "Problem Solving": 85,
                    "Project Management": 80,
                    "Structural Analysis": 85,
                    "CAD Software": 80
                }
            },
            "Mechanical Engineer": {
                "field": "Engineering",
                "description": "Designs and develops mechanical systems and products",
                "core_skills": {
                    "Mathematics": 85,
                    "Technical Design": 90,
                    "Problem Solving": 85,
                    "Thermodynamics": 80,
                    "Material Science": 80,
                    "CAD Software": 85
                }
            },
            "Electrical Engineer": {
                "field": "Engineering",
                "description": "Designs and develops electrical systems and equipment",
                "core_skills": {
                    "Circuit Design": 90,
                    "Electronics": 85,
                    "Power Systems": 80,
                    "Technical Design": 85,
                    "Problem Solving": 85,
                    "Testing and Validation": 80
                }
            },
            "Lawyer": {
                "field": "Law",
                "description": "Advises and represents clients in legal matters",
                "core_skills": {
                    "Legal Research": 90,
                    "Critical Thinking": 85,
                    "Communication": 90,
                    "Analysis": 85,
                    "Ethics": 85,
                    "Negotiation": 80
                }
            },
            "Legal Consultant": {
                "field": "Law",
                "description": "Provides expert legal advice to organizations",
                "core_skills": {
                    "Legal Research": 85,
                    "Critical Thinking": 80,
                    "Communication": 85,
                    "Analysis": 85,
                    "Ethics": 80,
                    "Industry Knowledge": 85
                }
            },
            "Patent Attorney": {
                "field": "Law",
                "description": "Specializes in intellectual property and patent law",
                "core_skills": {
                    "Patent Law": 90,
                    "Technical Knowledge": 85,
                    "Legal Writing": 85,
                    "Analysis": 85,
                    "Client Counseling": 80,
                    "Negotiation": 75
                }
            },
            "Criminologist": {
                "field": "Criminology",
                "description": "Studies criminal behavior and societal responses to crime",
                "core_skills": {
                    "Research": 90,
                    "Analysis": 85,
                    "Criminal Justice": 90,
                    "Psychology": 80,
                    "Statistics": 75,
                    "Ethics": 80
                }
            },
            "Criminal Investigator": {
                "field": "Criminology",
                "description": "Investigates crimes and gathers evidence",
                "core_skills": {
                    "Criminal Justice": 90,
                    "Analysis": 85,
                    "Attention to Detail": 90,
                    "Critical Thinking": 85,
                    "Communication": 80,
                    "Ethics": 85
                }
            },
            "Registered Nurse": {
                "field": "Nursing",
                "description": "Provides direct patient care and health education",
                "core_skills": {
                    "Patient Care": 95,
                    "Medical Knowledge": 85,
                    "Communication": 90,
                    "Critical Thinking": 85,
                    "Empathy": 90,
                    "Assessment": 85
                }
            },
            "Nurse Practitioner": {
                "field": "Nursing",
                "description": "Advanced practice nurse who can diagnose and prescribe treatments",
                "core_skills": {
                    "Patient Care": 90,
                    "Medical Knowledge": 90,
                    "Diagnosis": 85,
                    "Critical Thinking": 90,
                    "Empathy": 85,
                    "Communication": 85
                }
            },
            "Clinical Nurse Specialist": {
                "field": "Nursing",
                "description": "Expert clinician in a specialized area of nursing practice",
                "core_skills": {
                    "Specialized Care": 90,
                    "Clinical Expertise": 90,
                    "Education": 85,
                    "Research": 80,
                    "Leadership": 85,
                    "Quality Improvement": 80
                }
            },
            "Physician": {
                "field": "Medicine",
                "description": "Diagnoses and treats illnesses and injuries",
                "core_skills": {
                    "Medical Knowledge": 95,
                    "Diagnosis": 95,
                    "Patient Care": 90,
                    "Critical Thinking": 90,
                    "Ethics": 85,
                    "Communication": 85
                }
            },
            "Surgeon": {
                "field": "Medicine",
                "description": "Performs surgical procedures to treat injuries and diseases",
                "core_skills": {
                    "Medical Knowledge": 95,
                    "Surgical Skills": 95,
                    "Attention to Detail": 95,
                    "Critical Thinking": 90,
                    "Decision Making": 90,
                    "Ethics": 85
                }
            },
            "Psychiatrist": {
                "field": "Medicine",
                "description": "Diagnoses and treats mental health disorders",
                "core_skills": {
                    "Psychiatric Assessment": 90,
                    "Medical Knowledge": 90,
                    "Psychopharmacology": 85,
                    "Communication": 90,
                    "Empathy": 90,
                    "Analysis": 85
                }
            },
            "Hotel Manager": {
                "field": "Hospitality Management",
                "description": "Oversees operations and staff of hotels and accommodations",
                "core_skills": {
                    "Management": 90,
                    "Customer Service": 90,
                    "Communication": 85,
                    "Organization": 85,
                    "Problem Solving": 85,
                    "Financial Management": 80
                }
            },
            "Restaurant Manager": {
                "field": "Hospitality Management",
                "description": "Oversees food service operations and staff",
                "core_skills": {
                    "Management": 90,
                    "Customer Service": 90,
                    "Food Safety": 85,
                    "Organization": 85,
                    "Problem Solving": 80,
                    "Staff Training": 80
                }
            },
            "Tour Guide": {
                "field": "Tourism",
                "description": "Leads groups on tours of attractions and places of interest",
                "core_skills": {
                    "Communication": 95,
                    "Customer Service": 90,
                    "Geography": 85,
                    "Cultural Awareness": 90,
                    "Organization": 80,
                    "History Knowledge": 85
                }
            },
            "Travel Consultant": {
                "field": "Tourism",
                "description": "Helps clients plan travel arrangements and accommodations",
                "core_skills": {
                    "Customer Service": 90,
                    "Geography": 85,
                    "Organization": 85,
                    "Communication": 90,
                    "Problem Solving": 80,
                    "Destination Knowledge": 85
                }
            },
            "Destination Manager": {
                "field": "Tourism",
                "description": "Oversees tourism development and marketing for specific locations",
                "core_skills": {
                    "Marketing": 85,
                    "Event Planning": 80,
                    "Cultural Knowledge": 85,
                    "Business Development": 80,
                    "Communication": 85,
                    "Project Management": 80
                }
            },
            "Clinical Psychologist": {
                "field": "Psychology",
                "description": "Assesses and treats mental, emotional, and behavioral disorders",
                "core_skills": {
                    "Assessment": 90,
                    "Therapy Techniques": 90,
                    "Communication": 90,
                    "Empathy": 95,
                    "Critical Thinking": 85,
                    "Ethics": 90
                }
            },
            "Research Psychologist": {
                "field": "Psychology",
                "description": "Conducts research on psychological processes and behavior",
                "core_skills": {
                    "Research Methods": 90,
                    "Statistics": 85,
                    "Analysis": 90,
                    "Critical Thinking": 85,
                    "Technical Writing": 80,
                    "Ethics": 85
                }
            },
            "Organizational Psychologist": {
                "field": "Psychology",
                "description": "Applies psychological principles to workplace issues",
                "core_skills": {
                    "Organizational Behavior": 90,
                    "Assessment": 85,
                    "Consulting": 85,
                    "Data Analysis": 80,
                    "Communication": 85,
                    "Human Resources": 80
                }
            },
            "Medical Laboratory Technologist": {
                "field": "Medical Technology",
                "description": "Performs medical tests and analyses in a laboratory setting",
                "core_skills": {
                    "Laboratory Skills": 95,
                    "Analytical Thinking": 90,
                    "Attention to Detail": 95,
                    "Medical Knowledge": 85,
                    "Technical Skills": 85,
                    "Quality Control": 85
                }
            },
            "Radiologic Technologist": {
                "field": "Medical Technology",
                "description": "Performs medical imaging exams for diagnosis and treatment",
                "core_skills": {
                    "Technical Skills": 90,
                    "Patient Care": 85,
                    "Attention to Detail": 90,
                    "Medical Knowledge": 85,
                    "Equipment Operation": 90,
                    "Safety Protocols": 85
                }
            },
            "Research Scientist": {
                "field": "Research",
                "description": "Conducts original research in scientific fields",
                "core_skills": {
                    "Research Methods": 95,
                    "Data Analysis": 90,
                    "Critical Thinking": 90,
                    "Technical Writing": 85,
                    "Problem Solving": 85,
                    "Laboratory Skills": 85
                }
            },
            "Market Research Analyst": {
                "field": "Research",
                "description": "Analyzes market conditions to examine potential sales of products or services",
                "core_skills": {
                    "Research Methods": 90,
                    "Data Analysis": 90,
                    "Statistics": 85,
                    "Critical Thinking": 85,
                    "Communication": 80,
                    "Industry Knowledge": 80
                }
            },
            "Elementary School Teacher": {
                "field": "Education",
                "description": "Teaches young students basic academic and social skills",
                "core_skills": {
                    "Communication": 90,
                    "Subject Knowledge": 85,
                    "Lesson Planning": 90,
                    "Assessment": 85,
                    "Empathy": 90,
                    "Classroom Management": 85
                }
            },
            "University Professor": {
                "field": "Education",
                "description": "Teaches and conducts research at higher education institutions",
                "core_skills": {
                    "Subject Knowledge": 95,
                    "Research": 90,
                    "Communication": 85,
                    "Critical Thinking": 90,
                    "Assessment": 80,
                    "Technical Writing": 85
                }
            },
            "Special Education Teacher": {
                "field": "Education",
                "description": "Works with students who have learning, mental, emotional, or physical disabilities",
                "core_skills": {
                    "Specialized Teaching Methods": 90,
                    "Patience": 95,
                    "Assessment": 85,
                    "IEP Development": 85,
                    "Adaptability": 90,
                    "Communication": 90
                }
            },
            "Environmental Scientist": {
                "field": "Environmental Science",
                "description": "Studies, develops, and implements solutions to environmental problems",
                "core_skills": {
                    "Research Methods": 90,
                    "Data Analysis": 85,
                    "Field Research": 85,
                    "Environmental Sampling": 80,
                    "Technical Writing": 80,
                    "Sustainability": 85
                }
            },
            "Conservation Scientist": {
                "field": "Environmental Science",
                "description": "Manages the overall land quality of forests, parks, and other natural resources",
                "core_skills": {
                    "Ecology": 90,
                    "Land Management": 85,
                    "Environmental Regulations": 85,
                    "GIS": 80,
                    "Field Work": 85,
                    "Conservation Planning": 80
                }
            },
            "Graphic Designer": {
                "field": "Graphic Design",
                "description": "Creates visual concepts to communicate ideas",
                "core_skills": {
                    "Adobe Creative Suite": 90,
                    "Typography": 85,
                    "Color Theory": 85,
                    "Layout Design": 90,
                    "Visual Communication": 90,
                    "Brand Development": 80
                }
            },
            "Illustrator": {
                "field": "Graphic Design",
                "description": "Creates original artwork using digital and traditional methods",
                "core_skills": {
                    "Drawing": 95,
                    "Digital Illustration": 90,
                    "Creative Concept Development": 85,
                    "Color Theory": 85,
                    "Adobe Illustrator": 90,
                    "Visual Storytelling": 80
                }
            },
            "UX Designer": {
                "field": "UX/UI Design",
                "description": "Focuses on optimizing user experience with digital products",
                "core_skills": {
                    "User Research": 90,
                    "Wireframing": 85,
                    "Prototyping": 90,
                    "Usability Testing": 85,
                    "User Flows": 85,
                    "Information Architecture": 80
                }
            },
            "UI Designer": {
                "field": "UX/UI Design",
                "description": "Creates visually appealing interfaces for digital products",
                "core_skills": {
                    "Visual Design": 90,
                    "UI Patterns": 85,
                    "Prototyping": 85,
                    "Interactive Design": 90,
                    "Typography": 80,
                    "Color Theory": 85
                }
            },
            "HR Manager": {
                "field": "Human Resources",
                "description": "Oversees hiring, administration, and training of employees",
                "core_skills": {
                    "Recruitment": 85,
                    "Employee Relations": 90,
                    "HR Policies": 85,
                    "Conflict Resolution": 85,
                    "Performance Management": 80,
                    "Employment Law": 80
                }
            },
            "Talent Acquisition Specialist": {
                "field": "Human Resources",
                "description": "Focuses on recruiting and hiring qualified candidates",
                "core_skills": {
                    "Recruiting": 95,
                    "Interviewing": 90,
                    "Candidate Assessment": 85,
                    "Employer Branding": 80,
                    "Job Market Knowledge": 85,
                    "Networking": 80
                }
            },
            "Supply Chain Manager": {
                "field": "Supply Chain Management",
                "description": "Oversees and coordinates all supply chain activities",
                "core_skills": {
                    "Supply Chain Planning": 90,
                    "Logistics Management": 85,
                    "Procurement": 85,
                    "Inventory Management": 85,
                    "Vendor Management": 80,
                    "Cost Analysis": 80
                }
            },
            "Logistics Coordinator": {
                "field": "Supply Chain Management",
                "description": "Manages the transportation and distribution of goods",
                "core_skills": {
                    "Transportation Management": 90,
                    "Routing Optimization": 85,
                    "Shipment Tracking": 85,
                    "Customs Knowledge": 80,
                    "Problem Solving": 85,
                    "Communication": 80
                }
            },
            "Biotechnologist": {
                "field": "Biotechnology",
                "description": "Applies biological processes to develop products and tools",
                "core_skills": {
                    "Laboratory Techniques": 90,
                    "Molecular Biology": 85,
                    "Cell Culture": 85,
                    "Data Analysis": 80,
                    "Research Design": 85,
                    "Documentation": 80
                }
            },
            "Biomedical Engineer": {
                "field": "Biotechnology",
                "description": "Designs and develops biological systems and products for medical use",
                "core_skills": {
                    "Engineering Design": 90,
                    "Medical Device Knowledge": 85,
                    "Biomaterials": 85,
                    "Research Methods": 80,
                    "Problem Solving": 85,
                    "Regulatory Affairs": 75
                }
            },
            "Architect": {
                "field": "Architecture",
                "description": "Designs buildings and structures, supervising their construction",
                "core_skills": {
                    "Architectural Design": 95,
                    "Building Codes": 85,
                    "CAD Software": 90,
                    "Project Management": 80,
                    "3D Modeling": 85,
                    "Technical Drawing": 85
                }
            },
            "Landscape Architect": {
                "field": "Architecture",
                "description": "Plans and designs land areas for parks, recreational facilities, and other outdoor spaces",
                "core_skills": {
                    "Landscape Design": 90,
                    "Site Planning": 85,
                    "Plant Knowledge": 85,
                    "CAD Software": 80,
                    "Environmental Analysis": 80,
                    "Visualization": 85
                }
            },
            "Journalist": {
                "field": "Journalism",
                "description": "Researches, writes, and reports news stories",
                "core_skills": {
                    "Reporting": 90,
                    "Writing": 95,
                    "Research": 85,
                    "Interviewing": 90,
                    "Fact-Checking": 85,
                    "Editing": 80
                }
            },
            "Photojournalist": {
                "field": "Journalism",
                "description": "Tells news stories through photographs",
                "core_skills": {
                    "Photography": 95,
                    "Visual Storytelling": 90,
                    "News Judgment": 85,
                    "Photo Editing": 85,
                    "Ethics": 80,
                    "Technical Camera Skills": 90
                }
            }
        }
        
        # Save to file
        with open(os.path.join(self.output_dir, "specializations.json"), "w") as f:
            json.dump(specializations, f, indent=4)
            
        return specializations
    
    def generate_skill_weights(self) -> Dict[str, Any]:
        """
        Generate skill weights for recommendations
        
        Returns:
            Dictionary of skills with their global and field-specific weights
        """
        all_skills = {}
        
        # Add base programming languages
        for lang in self.programming_languages:
            all_skills[lang] = {"global_weight": random.randint(70, 90)}
            
        # Add data skills
        for skill in self.data_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add ML skills
        for skill in self.ml_skills:
            all_skills[skill] = {"global_weight": random.randint(75, 90)}
            
        # Add web skills
        for skill in self.web_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add soft skills
        for skill in self.soft_skills:
            all_skills[skill] = {"global_weight": random.randint(80, 95)}
            
        # Add cloud skills
        for skill in self.cloud_skills:
            all_skills[skill] = {"global_weight": random.randint(75, 90)}
            
        # Add finance skills
        for skill in self.finance_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add healthcare skills
        for skill in self.healthcare_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add art and design skills
        for skill in self.art_design_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add marketing skills
        for skill in self.marketing_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add science skills
        for skill in self.science_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add manufacturing skills
        for skill in self.manufacturing_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Add additional skills that don't fit in the above categories
        additional_skills = {
            "Mathematics": {"global_weight": random.randint(70, 85)},
            "Statistics": {"global_weight": random.randint(70, 85)},
            "Data Structures": {"global_weight": random.randint(75, 90)},
            "Algorithms": {"global_weight": random.randint(75, 90)},
            "Software Design": {"global_weight": random.randint(75, 90)},
            "Testing": {"global_weight": random.randint(70, 85)},
            "Version Control": {"global_weight": random.randint(70, 85)},
            "Financial Analysis": {"global_weight": random.randint(70, 85)},
            "Bookkeeping": {"global_weight": random.randint(70, 85)},
            "Tax Knowledge": {"global_weight": random.randint(70, 85)},
            "Auditing": {"global_weight": random.randint(70, 85)},
            "Risk Assessment": {"global_weight": random.randint(70, 85)},
            "Technical Design": {"global_weight": random.randint(70, 85)},
            "CAD Software": {"global_weight": random.randint(70, 85)},
            "Structural Analysis": {"global_weight": random.randint(70, 85)},
            "Thermodynamics": {"global_weight": random.randint(70, 85)},
            "Material Science": {"global_weight": random.randint(70, 85)},
            "Legal Research": {"global_weight": random.randint(70, 85)},
            "Ethics": {"global_weight": random.randint(75, 90)},
            "Negotiation": {"global_weight": random.randint(70, 85)},
            "Criminal Justice": {"global_weight": random.randint(70, 85)},
            "Patient Care": {"global_weight": random.randint(75, 90)},
            "Medical Knowledge": {"global_weight": random.randint(75, 90)},
            "Empathy": {"global_weight": random.randint(75, 90)},
            "Diagnosis": {"global_weight": random.randint(75, 90)},
            "Surgical Skills": {"global_weight": random.randint(75, 90)},
            "Customer Service": {"global_weight": random.randint(75, 90)},
            "Geography": {"global_weight": random.randint(70, 85)},
            "Cultural Awareness": {"global_weight": random.randint(70, 85)},
            "Organization": {"global_weight": random.randint(75, 90)},
            "Laboratory Skills": {"global_weight": random.randint(70, 85)},
            "Analytical Thinking": {"global_weight": random.randint(75, 90)},
            "Technical Skills": {"global_weight": random.randint(70, 85)},
            "Research Methods": {"global_weight": random.randint(70, 85)},
            "Technical Writing": {"global_weight": random.randint(70, 85)},
            "Subject Knowledge": {"global_weight": random.randint(75, 90)},
            "Lesson Planning": {"global_weight": random.randint(70, 85)},
            "Assessment": {"global_weight": random.randint(70, 85)},
            "Network Security": {"global_weight": random.randint(70, 85)},
            "Threat Analysis": {"global_weight": random.randint(70, 85)},
            "Penetration Testing": {"global_weight": random.randint(70, 85)},
            "Cryptography": {"global_weight": random.randint(70, 85)},
            "SEO": {"global_weight": random.randint(70, 85)},
            "Content Creation": {"global_weight": random.randint(70, 85)},
            "Marketing Strategy": {"global_weight": random.randint(70, 85)},
            "Investment Management": {"global_weight": random.randint(70, 85)},
            "Financial Planning": {"global_weight": random.randint(70, 85)},
            "Market Analysis": {"global_weight": random.randint(70, 85)},
            "Environmental Analysis": {"global_weight": random.randint(70, 85)},
            "Environmental Policy": {"global_weight": random.randint(70, 85)},
            "Sustainability": {"global_weight": random.randint(70, 85)},
            "Visual Design": {"global_weight": random.randint(70, 85)},
            "Typography": {"global_weight": random.randint(70, 85)},
            "Layout Design": {"global_weight": random.randint(70, 85)},
            "User Research": {"global_weight": random.randint(70, 85)},
            "Wireframing": {"global_weight": random.randint(70, 85)},
            "Prototyping": {"global_weight": random.randint(70, 85)},
            "Interaction Design": {"global_weight": random.randint(70, 85)},
            "Usability Testing": {"global_weight": random.randint(70, 85)},
            "Recruitment": {"global_weight": random.randint(70, 85)},
            "Employee Relations": {"global_weight": random.randint(70, 85)},
            "HR Policies": {"global_weight": random.randint(70, 85)},
            "Training and Development": {"global_weight": random.randint(70, 85)},
            "Logistics Management": {"global_weight": random.randint(70, 85)},
            "Inventory Control": {"global_weight": random.randint(70, 85)},
            "Procurement": {"global_weight": random.randint(70, 85)},
            "Supply Chain Optimization": {"global_weight": random.randint(70, 85)},
            "Demand Planning": {"global_weight": random.randint(70, 85)},
            "Molecular Biology": {"global_weight": random.randint(70, 85)},
            "Experimental Design": {"global_weight": random.randint(70, 85)},
            "Architectural Design": {"global_weight": random.randint(70, 85)},
            "Building Codes": {"global_weight": random.randint(70, 85)},
            "Technical Drawing": {"global_weight": random.randint(70, 85)},
            "Reporting": {"global_weight": random.randint(70, 85)},
            "Writing": {"global_weight": random.randint(70, 85)},
            "Interviewing": {"global_weight": random.randint(70, 85)},
            "Media Ethics": {"global_weight": random.randint(70, 85)}
        }
        
        all_skills.update(additional_skills)
            
        # Add field-specific importance
        for skill in all_skills:
            field_weights = {}
            
            # Computer Science field weights
            if skill in self.programming_languages or skill == "Problem Solving" or skill == "Algorithms" or skill == "Data Structures":
                field_weights["Computer Science"] = random.randint(80, 95)
            
            # Business Accountancy field weights
            if skill in ["Financial Analysis", "Auditing", "Bookkeeping", "Tax Knowledge", "Business Ethics"]:
                field_weights["Business Accountancy"] = random.randint(80, 95)
            
            # Engineering field weights
            if skill in ["Mathematics", "Problem Solving", "Technical Design", "Critical Thinking", "Analysis"]:
                field_weights["Engineering"] = random.randint(80, 95)
            
            # Law field weights
            if skill in ["Legal Research", "Critical Thinking", "Communication", "Analysis", "Ethics"]:
                field_weights["Law"] = random.randint(80, 95)
            
            # Criminology field weights
            if skill in ["Criminal Justice", "Research", "Analysis", "Psychology", "Ethics"]:
                field_weights["Criminology"] = random.randint(80, 95)
            
            # Nursing field weights
            if skill in ["Patient Care", "Medical Knowledge", "Communication", "Critical Thinking", "Empathy"]:
                field_weights["Nursing"] = random.randint(80, 95)
            
            # Medicine field weights
            if skill in ["Medical Knowledge", "Diagnosis", "Patient Care", "Critical Thinking", "Ethics"]:
                field_weights["Medicine"] = random.randint(80, 95)
            
            # Hospitality Management field weights
            if skill in ["Customer Service", "Management", "Communication", "Organization", "Problem Solving"]:
                field_weights["Hospitality Management"] = random.randint(80, 95)
            
            # Tourism field weights
            if skill in ["Customer Service", "Geography", "Cultural Awareness", "Communication", "Organization"]:
                field_weights["Tourism"] = random.randint(80, 95)
            
            # Psychology field weights
            if skill in ["Research", "Analysis", "Communication", "Empathy", "Critical Thinking"]:
                field_weights["Psychology"] = random.randint(80, 95)
            
            # Medical Technology field weights
            if skill in ["Laboratory Skills", "Analytical Thinking", "Attention to Detail", "Medical Knowledge", "Technical Skills"]:
                field_weights["Medical Technology"] = random.randint(80, 95)
            
            # Research field weights
            if skill in ["Research Methods", "Data Analysis", "Critical Thinking", "Technical Writing", "Problem Solving"]:
                field_weights["Research"] = random.randint(80, 95)
            
            # Education field weights
            if skill in ["Communication", "Subject Knowledge", "Lesson Planning", "Assessment", "Empathy"]:
                field_weights["Education"] = random.randint(80, 95)
            
            # Data Science field weights
            if skill in ["Machine Learning", "Statistics", "Data Analysis", "Python", "Data Visualization"] or skill in self.ml_skills or skill in self.data_skills:
                field_weights["Data Science"] = random.randint(80, 95)
            
            # Cybersecurity field weights
            if skill in ["Network Security", "Threat Analysis", "Security Protocols", "Penetration Testing", "Cryptography"]:
                field_weights["Cybersecurity"] = random.randint(80, 95)
            
            # Digital Marketing field weights
            if skill in ["SEO", "Social Media Marketing", "Content Creation", "Analytics", "Marketing Strategy"] or skill in self.marketing_skills:
                field_weights["Digital Marketing"] = random.randint(80, 95)
            
            # Finance field weights
            if skill in ["Financial Analysis", "Investment Management", "Risk Assessment", "Financial Planning", "Market Analysis"] or skill in self.finance_skills:
                field_weights["Finance"] = random.randint(80, 95)
            
            # Environmental Science field weights
            if skill in ["Environmental Analysis", "Research Methods", "Data Collection", "Environmental Policy", "Sustainability"] or skill in self.science_skills:
                field_weights["Environmental Science"] = random.randint(80, 95)
            
            # Graphic Design field weights
            if skill in ["Visual Design", "Typography", "Color Theory", "Adobe Creative Suite", "Layout Design"] or skill in self.art_design_skills:
                field_weights["Graphic Design"] = random.randint(80, 95)
            
            # UX/UI Design field weights
            if skill in ["User Research", "Wireframing", "Prototyping", "Interaction Design", "Usability Testing"] or skill in self.art_design_skills:
                field_weights["UX/UI Design"] = random.randint(80, 95)
            
            # Human Resources field weights
            if skill in ["Recruitment", "Employee Relations", "Performance Management", "HR Policies", "Training and Development"]:
                field_weights["Human Resources"] = random.randint(80, 95)
            
            # Supply Chain Management field weights
            if skill in ["Logistics Management", "Inventory Control", "Procurement", "Supply Chain Optimization", "Demand Planning"] or skill in self.manufacturing_skills:
                field_weights["Supply Chain Management"] = random.randint(80, 95)
            
            # Biotechnology field weights
            if skill in ["Laboratory Techniques", "Molecular Biology", "Research Methods", "Data Analysis", "Experimental Design"] or skill in self.science_skills:
                field_weights["Biotechnology"] = random.randint(80, 95)
            
            # Architecture field weights
            if skill in ["Architectural Design", "3D Modeling", "Building Codes", "Technical Drawing", "Project Management"]:
                field_weights["Architecture"] = random.randint(80, 95)
            
            # Journalism field weights
            if skill in ["Reporting", "Writing", "Research", "Interviewing", "Media Ethics"]:
                field_weights["Journalism"] = random.randint(80, 95)
            
            # Web development field weights - add this section to properly associate web skills
            if skill in self.web_skills:
                field_weights["Computer Science"] = random.randint(80, 95)
                field_weights["Digital Marketing"] = random.randint(70, 85)
                field_weights["UX/UI Design"] = random.randint(75, 90)
                field_weights["Graphic Design"] = random.randint(65, 80)
            
            # Common skills that apply across multiple fields
            if skill in self.soft_skills:
                for field in ["Computer Science", "Business Accountancy", "Engineering", "Law", "Criminology", 
                              "Nursing", "Medicine", "Hospitality Management", "Tourism", "Psychology", 
                              "Medical Technology", "Research", "Education", "Data Science", "Cybersecurity",
                              "Digital Marketing", "Finance", "Environmental Science", "Graphic Design", 
                              "UX/UI Design", "Human Resources", "Supply Chain Management", "Biotechnology",
                              "Architecture", "Journalism"]:
                    if field not in field_weights:
                        field_weights[field] = random.randint(60, 80)
            
            all_skills[skill]["field_weights"] = field_weights
            
        # Save to file
        with open(os.path.join(self.output_dir, "skill_weights.json"), "w") as f:
            json.dump(all_skills, f, indent=4)
            
        return all_skills
    
    def generate_all(self) -> None:
        """Generate all data for the career recommender system"""
        self.generate_fields()
        self.generate_specializations()
        self.generate_skill_weights()
        print(f"Data generated and saved to {self.output_dir}")


if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_all() 