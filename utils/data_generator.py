"""
Synthetic data generator for the career recommendation system.
This module provides functionality to generate synthetic employee and career path data
for training and testing the career recommendation models.
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime

class SyntheticDataGenerator:
    """
    Generator for synthetic employee and career path data.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the data generator with a random seed for reproducibility.
        
        Args:
            seed (int): Random seed for reproducible data generation
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define career fields and their specializations
        self.career_fields = {
            "Computer Science": {
                "specializations": [
                    "Software Engineer", "Data Scientist", "UI/UX Designer", 
                    "Machine Learning Engineer", "DevOps Engineer", "Full-Stack Developer",
                    "Frontend Developer", "Backend Developer", "Data Engineer",
                    "Cybersecurity Analyst", "Cloud Architect", "Game Developer",
                    "Mobile App Developer", "Software Architect", "AI Research Scientist",
                    "Embedded Systems Developer", "Network Engineer", "System Administrator"
                ],
                "skills": [
                    "Python", "Java", "JavaScript", "C++", "C#", "SQL", "NoSQL",
                    "HTML", "CSS", "React.js", "Angular", "Vue.js", "Node.js",
                    "Express.js", "Django", "Flask", "Spring", "Docker", "Kubernetes",
                    "AWS", "Azure", "Git", "TensorFlow", "PyTorch", "Scikit-learn",
                    "Machine Learning", "Deep Learning", "Natural Language Processing",
                    "Computer Vision", "Data Analysis", "Data Visualization", "Big Data",
                    "Hadoop", "Spark", "Cloud Computing", "DevOps", "CI/CD", "Linux",
                    "Windows Server", "Networking", "Security", "Agile", "Scrum",
                    "Test-Driven Development", "RESTful APIs", "GraphQL", "Microservices",
                    "System Design", "Object-Oriented Programming", "Functional Programming",
                    "Game Development", "Unity", "Unreal Engine", "Mobile Development",
                    "Android", "iOS", "Swift", "Kotlin", "Web Development", "Responsive Design",
                    "UX Design", "UI Design", "Figma", "Adobe XD", "Sketch", "Blockchain",
                    "AR/VR", "Cybersecurity", "Ethical Hacking", "Penetration Testing"
                ]
            },
            "Finance": {
                "specializations": [
                    "Investment Banker", "Portfolio Manager", "Risk Manager", 
                    "Financial Advisor", "Hedge Fund Analyst", "Quantitative Analyst",
                    "Financial Analyst", "Trader", "Compliance Officer", "Auditor"
                ],
                "skills": [
                    "Financial Analysis", "Valuation", "Financial Modeling", 
                    "Investment Management", "Risk Assessment", "Portfolio Management",
                    "Financial Statements", "Excel", "VBA", "Bloomberg Terminal",
                    "Capital Markets", "Equity Research", "Fixed Income", "Derivatives",
                    "Hedge Funds", "Private Equity", "Mergers & Acquisitions", "IPOs",
                    "Due Diligence", "Forecasting", "Budgeting", "Accounting", "CFA",
                    "Series 7", "Series 63", "FINRA", "Regulatory Compliance", "GAAP",
                    "IFRS", "Tax Planning", "Estate Planning", "Wealth Management",
                    "Retirement Planning", "Insurance Planning", "Credit Analysis",
                    "Loan Underwriting", "Algorithmic Trading", "Quantitative Analysis",
                    "R Programming", "Python for Finance", "SAS", "STATA"
                ]
            },
            "Marketing": {
                "specializations": [
                    "Digital Marketing Specialist", "Brand Manager", "Market Research Analyst",
                    "Social Media Manager", "Content Strategist", "SEO Specialist",
                    "Marketing Analyst", "Product Marketing Manager", "Public Relations Manager",
                    "Marketing Director", "Campaign Manager", "Growth Hacker"
                ],
                "skills": [
                    "Digital Marketing", "Social Media Marketing", "Content Marketing",
                    "SEO", "SEM", "Google Ads", "Facebook Ads", "Email Marketing",
                    "Marketing Analytics", "Market Research", "Brand Management",
                    "Campaign Management", "Google Analytics", "Social Media Strategy",
                    "Content Strategy", "Content Creation", "Copywriting", "Blogging",
                    "Video Marketing", "Influencer Marketing", "Marketing Automation",
                    "CRM", "Lead Generation", "Conversion Rate Optimization", "A/B Testing",
                    "Customer Segmentation", "Marketing Strategy", "Competitive Analysis",
                    "Product Positioning", "Public Relations", "Press Releases", "Media Relations",
                    "Event Marketing", "Affiliate Marketing", "Mobile Marketing", "Growth Hacking",
                    "Adobe Creative Suite", "Photoshop", "Illustrator", "User Acquisition",
                    "Customer Retention", "Marketing ROI", "Marketing Metrics", "HubSpot",
                    "Marketo", "Mailchimp", "Hootsuite", "Buffer", "Canva", "WordPress"
                ]
            },
            "Healthcare": {
                "specializations": [
                    "Physician", "Registered Nurse", "Healthcare Administrator",
                    "Physical Therapist", "Medical Researcher", "Pharmacist",
                    "Dentist", "Veterinarian", "Nutritionist", "Mental Health Counselor"
                ],
                "skills": [
                    "Patient Care", "Medical Terminology", "Clinical Experience",
                    "Electronic Health Records", "Medical Diagnosis", "Treatment Planning",
                    "Medication Administration", "Healthcare Management", "Health Policy",
                    "Healthcare Regulations", "HIPAA", "Medical Billing", "Medical Coding",
                    "CPT Codes", "ICD-10", "Healthcare IT", "Medical Research",
                    "Clinical Trials", "Anatomy", "Physiology", "Pathology", "Pharmacology",
                    "Patient Assessment", "Vital Signs", "Medical Records", "Wound Care",
                    "Infection Control", "Sterilization", "Patient Education", "Care Planning",
                    "Emergency Medicine", "Surgical Procedures", "Mental Health", "Pediatrics",
                    "Geriatrics", "Oncology", "Cardiology", "Neurology", "Orthopedics",
                    "Radiology", "Laboratory Procedures", "Diagnostic Testing", "Therapy",
                    "Rehabilitation", "Nutrition", "Wellness", "Public Health", "Epidemiology"
                ]
            },
            "Law": {
                "specializations": [
                    "Corporate Lawyer", "Litigation Attorney", "Intellectual Property Lawyer",
                    "Criminal Defense Attorney", "Family Law Attorney", "Immigration Lawyer",
                    "Real Estate Attorney", "Tax Attorney", "Employment Law Attorney",
                    "Environmental Lawyer"
                ],
                "skills": [
                    "Legal Research", "Legal Writing", "Trial Advocacy", "Legal Analysis",
                    "Contract Drafting", "Contract Negotiation", "Due Diligence", "Litigation",
                    "Case Strategy", "Client Representation", "Regulatory Compliance",
                    "Legal Drafting", "Oral Advocacy", "Depositions", "Motion Practice",
                    "Discovery Process", "Witness Preparation", "Evidence Management",
                    "Negotiation", "Mediation", "Arbitration", "Client Communication",
                    "Legal Ethics", "Courtroom Procedure", "Legal Memos", "Brief Writing",
                    "Legal Documentation", "Corporate Governance", "Mergers & Acquisitions",
                    "Securities Law", "Intellectual Property", "Patent Law", "Trademark Law",
                    "Copyright Law", "Criminal Law", "Criminal Procedure", "Constitutional Law",
                    "Family Law", "Immigration Law", "Real Estate Law", "Tax Law", "Employment Law"
                ]
            }
        }
        
    def generate_employee_data(self, num_entries=200, output_file=None, append=False):
        """
        Generate synthetic employee data.
        
        Args:
            num_entries (int): Number of employee records to generate
            output_file (str): Path to save the generated data (if None, returns DataFrame)
            append (bool): Whether to append to an existing file
            
        Returns:
            pandas.DataFrame: Generated employee data if output_file is None
        """
        # Create lists to store the data
        employee_ids = [f'EMP{i:04d}' for i in range(1, num_entries + 1)]
        names = [f'Employee {i}' for i in range(1, num_entries + 1)]
        ages = np.random.randint(22, 65, size=num_entries)
        experience_years = np.random.randint(1, 30, size=num_entries)
        
        # Randomly select a field and specialization for each employee
        fields = []
        specializations = []
        skills_list = []
        
        for _ in range(num_entries):
            # Randomly select a field
            field = random.choice(list(self.career_fields.keys()))
            fields.append(field)
            
            # Randomly select a specialization from that field
            specialization = random.choice(self.career_fields[field]["specializations"])
            specializations.append(specialization)
            
            # Generate random skills for the employee
            field_skills = self.career_fields[field]["skills"]
            num_skills = random.randint(5, 15)
            skills = random.sample(field_skills, min(num_skills, len(field_skills)))
            skills_list.append(", ".join(skills))
        
        # Create a DataFrame
        data = {
            'Employee ID': employee_ids,
            'Name': names,
            'Age': ages,
            'Years Experience': experience_years,
            'Skills': skills_list,
            'Field': fields,
            'Specialization': specializations
        }
        
        df = pd.DataFrame(data)
        
        # Save to file if specified
        if output_file:
            if append and os.path.exists(output_file):
                # For append mode with JSON, we need to load existing data first,
                # combine with new data, then save
                try:
                    existing_df = pd.read_json(output_file, orient='records')
                    df = pd.concat([existing_df, df], ignore_index=True)
                except Exception as e:
                    print(f"Warning: Could not append to existing JSON file: {str(e)}")
                    print("Creating new file instead.")
            
            df.to_json(output_file, orient='records', indent=4)
            print(f"Generated {num_entries} employee records saved to {os.path.abspath(output_file)}")
            
        return df
    
    def generate_career_path_data(self, num_entries=150, output_file=None, append=False):
        """
        Generate synthetic career path data.
        
        Args:
            num_entries (int): Number of career path records to generate
            output_file (str): Path to save the generated data (if None, returns DataFrame)
            append (bool): Whether to append to an existing file
            
        Returns:
            pandas.DataFrame: Generated career path data if output_file is None
        """
        # Lists to store data
        entry_ids = [f'CP{i:04d}' for i in range(1, num_entries + 1)]
        career_paths = []
        fields = []
        specializations = []
        skills_list = []
        
        for _ in range(num_entries):
            # Randomly select a field
            field = random.choice(list(self.career_fields.keys()))
            fields.append(field)
            
            # Randomly select a specialization from that field
            specialization = random.choice(self.career_fields[field]["specializations"])
            specializations.append(specialization)
            
            # Generate a career path name
            career_paths.append(f"{field} - {specialization}")
            
            # Generate random skills for the career path
            field_skills = self.career_fields[field]["skills"]
            num_skills = random.randint(10, 20)
            skills = random.sample(field_skills, min(num_skills, len(field_skills)))
            skills_list.append(", ".join(skills))
        
        # Create a DataFrame
        data = {
            'ID': entry_ids,
            'Career Path': career_paths,
            'Field': fields,
            'Specialization': specializations,
            'Required Skills': skills_list
        }
        
        df = pd.DataFrame(data)
        
        # Save to file if specified
        if output_file:
            if append and os.path.exists(output_file):
                # For append mode with JSON, we need to load existing data first,
                # combine with new data, then save
                try:
                    existing_df = pd.read_json(output_file, orient='records')
                    df = pd.concat([existing_df, df], ignore_index=True)
                except Exception as e:
                    print(f"Warning: Could not append to existing JSON file: {str(e)}")
                    print("Creating new file instead.")
            
            df.to_json(output_file, orient='records', indent=4)
            print(f"Generated {num_entries} career path records saved to {os.path.abspath(output_file)}")
            
        return df
    
    def generate_datasets(self, employee_count=200, career_path_count=150, 
                         employee_file=None, career_file=None, append=False):
        """
        Generate both employee and career path datasets.
        
        Args:
            employee_count (int): Number of employee records to generate
            career_path_count (int): Number of career path records to generate
            employee_file (str): Path to save employee data
            career_file (str): Path to save career path data
            append (bool): Whether to append to existing files
            
        Returns:
            tuple: (employee_df, career_path_df) if output files are None
        """
        employee_df = self.generate_employee_data(
            num_entries=employee_count,
            output_file=employee_file,
            append=append
        )
        
        career_path_df = self.generate_career_path_data(
            num_entries=career_path_count,
            output_file=career_file,
            append=append
        )
        
        return employee_df, career_path_df 