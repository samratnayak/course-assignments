"""
ATS (Applicant Tracking System) Scoring Module

This module calculates ATS compatibility scores for resumes based on:
- Semantic similarity with job description
- Skill match percentage
- Keyword presence
"""

from typing import Dict, List
from embedding_model import EmbeddingModel
from skill_matcher import SkillMatcher
import warnings

warnings.filterwarnings("ignore")


class ATSScorer:
    """
    Class for calculating ATS compatibility scores.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, skill_matcher: SkillMatcher):
        """
        Initialize ATSScorer.
        
        Args:
            embedding_model: EmbeddingModel instance
            skill_matcher: SkillMatcher instance
        """
        self.embedding_model = embedding_model
        self.skill_matcher = skill_matcher
        
        # ATS Score weights
        self.weight_semantic_similarity = 0.5
        self.weight_skill_match = 0.3
        self.weight_keyword_presence = 0.2
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text (simple approach - can be enhanced).
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Common technical keywords
        common_keywords = [
            "python", "java", "javascript", "sql", "machine learning", "ai", "deep learning",
            "data science", "cloud", "aws", "azure", "docker", "kubernetes", "git",
            "agile", "scrum", "api", "rest", "microservices", "react", "node.js",
            "tensorflow", "pytorch", "nlp", "computer vision", "big data", "hadoop",
            "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "spark",
            "ci/cd", "jenkins", "terraform", "ansible", "linux", "bash"
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in common_keywords if kw in text_lower]
        
        return found_keywords
    
    def calculate_keyword_presence(self, resume_text: str, jd_keywords: List[str]) -> float:
        """
        Calculate keyword presence score.
        
        Args:
            resume_text: Resume text
            jd_keywords: Keywords from job description
            
        Returns:
            Keyword presence score between 0 and 1
        """
        if not jd_keywords:
            return 1.0
        
        resume_keywords = self.extract_keywords(resume_text)
        resume_keywords_lower = [kw.lower() for kw in resume_keywords]
        jd_keywords_lower = [kw.lower() for kw in jd_keywords]
        
        # Count matching keywords
        matching_keywords = set(resume_keywords_lower) & set(jd_keywords_lower)
        presence_score = len(matching_keywords) / len(jd_keywords) if jd_keywords else 0.0
        
        return min(presence_score, 1.0)
    
    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Calculate semantic similarity between resume and job description.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            
        Returns:
            Semantic similarity score between 0 and 1
        """
        # Use first 512 characters for efficiency (can be adjusted)
        resume_snippet = resume_text[:512] if len(resume_text) > 512 else resume_text
        jd_snippet = jd_text[:512] if len(jd_text) > 512 else jd_text
        
        similarity = self.embedding_model.similarity(resume_snippet, jd_snippet)
        
        # Normalize to 0-1 range (cosine similarity is already -1 to 1, but typically 0 to 1)
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def calculate_ats_score(self, resume_text: str, resume_skills: List[str], 
                           jd_text: str, jd_skills: List[str], jd_keywords: List[str]) -> Dict:
        """
        Calculate overall ATS compatibility score.
        
        Args:
            resume_text: Full resume text
            resume_skills: List of skills from resume
            jd_text: Job description text
            jd_skills: List of skills from job description
            jd_keywords: Keywords from job description
            
        Returns:
            Dictionary with score breakdown and overall score
        """
        # Calculate component scores
        semantic_score = self.calculate_semantic_similarity(resume_text, jd_text)
        skill_match_score = self.skill_matcher.calculate_skill_match_score(resume_skills, jd_skills)
        keyword_score = self.calculate_keyword_presence(resume_text, jd_keywords)
        
        # Calculate weighted overall score
        overall_score = (
            self.weight_semantic_similarity * semantic_score +
            self.weight_skill_match * skill_match_score +
            self.weight_keyword_presence * keyword_score
        )
        
        return {
            "overall_score": overall_score,
            "semantic_similarity": semantic_score,
            "skill_match": skill_match_score,
            "keyword_presence": keyword_score,
            "breakdown": {
                "semantic_weight": self.weight_semantic_similarity,
                "skill_weight": self.weight_skill_match,
                "keyword_weight": self.weight_keyword_presence
            }
        }
    
    def get_score_interpretation(self, score: float) -> str:
        """
        Get human-readable interpretation of ATS score.
        
        Args:
            score: ATS score (0-1)
            
        Returns:
            Interpretation string
        """
        if score >= 0.8:
            return "Excellent - High ATS compatibility"
        elif score >= 0.6:
            return "Good - Moderate ATS compatibility"
        elif score >= 0.4:
            return "Fair - Some improvements needed"
        else:
            return "Poor - Significant improvements required"
