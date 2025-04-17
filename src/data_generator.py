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
            "Data Scientist": {
                "field": "Computer Science",
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
            }
        }
        
        # Save to file
        with open(os.path.join(self.output_dir, "specializations.json"), "w") as f:
            json.dump(specializations, f, indent=4)
            
        return specializations
    
    def generate_skill_weights(self) -> Dict[str, Any]:
        """
        Generate skill weights for importance in career fields
        
        Returns:
            Dictionary of skills with their weights
        """
        # Combine all skills and assign global importance weights
        all_skills = {}
        
        # Programming languages
        for skill in self.programming_languages:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Data skills
        for skill in self.data_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # ML skills
        for skill in self.ml_skills:
            all_skills[skill] = {"global_weight": random.randint(75, 95)}
            
        # Web skills
        for skill in self.web_skills:
            all_skills[skill] = {"global_weight": random.randint(70, 90)}
            
        # Soft skills
        for skill in self.soft_skills:
            all_skills[skill] = {"global_weight": random.randint(60, 85)}
            
        # Cloud skills
        for skill in self.cloud_skills:
            all_skills[skill] = {"global_weight": random.randint(75, 90)}
            
        # Additional skills for new fields
        additional_skills = {
            "Financial Analysis": {"global_weight": random.randint(75, 90)},
            "Auditing": {"global_weight": random.randint(70, 85)},
            "Bookkeeping": {"global_weight": random.randint(70, 85)},
            "Tax Knowledge": {"global_weight": random.randint(75, 90)},
            "Business Ethics": {"global_weight": random.randint(70, 85)},
            "Technical Design": {"global_weight": random.randint(75, 90)},
            "Legal Research": {"global_weight": random.randint(80, 95)},
            "Ethics": {"global_weight": random.randint(70, 85)},
            "Criminal Justice": {"global_weight": random.randint(75, 90)},
            "Patient Care": {"global_weight": random.randint(80, 95)},
            "Medical Knowledge": {"global_weight": random.randint(80, 95)},
            "Empathy": {"global_weight": random.randint(70, 85)},
            "Diagnosis": {"global_weight": random.randint(80, 95)},
            "Customer Service": {"global_weight": random.randint(75, 90)},
            "Management": {"global_weight": random.randint(75, 90)},
            "Organization": {"global_weight": random.randint(70, 85)},
            "Geography": {"global_weight": random.randint(70, 85)},
            "Cultural Awareness": {"global_weight": random.randint(70, 85)},
            "Laboratory Skills": {"global_weight": random.randint(80, 95)},
            "Analytical Thinking": {"global_weight": random.randint(75, 90)},
            "Attention to Detail": {"global_weight": random.randint(75, 90)},
            "Technical Skills": {"global_weight": random.randint(75, 90)},
            "Research Methods": {"global_weight": random.randint(80, 95)},
            "Technical Writing": {"global_weight": random.randint(70, 85)},
            "Subject Knowledge": {"global_weight": random.randint(80, 95)},
            "Lesson Planning": {"global_weight": random.randint(75, 90)},
            "Assessment": {"global_weight": random.randint(70, 85)}
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
            
            # Common skills that apply across multiple fields
            if skill in self.soft_skills:
                for field in ["Computer Science", "Business Accountancy", "Engineering", "Law", "Criminology", 
                              "Nursing", "Medicine", "Hospitality Management", "Tourism", "Psychology", 
                              "Medical Technology", "Research", "Education"]:
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