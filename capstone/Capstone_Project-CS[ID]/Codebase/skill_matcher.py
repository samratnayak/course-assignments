"""
Skill Gap Detection Module

This module detects missing skills by comparing job description skills
with resume skills using semantic similarity.
"""

from typing import List, Dict, Tuple
from embedding_model import EmbeddingModel
import warnings

warnings.filterwarnings("ignore")


class SkillMatcher:
    """
    Class for skill gap detection using semantic similarity.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize SkillMatcher with embedding model.
        
        Args:
            embedding_model: EmbeddingModel instance for semantic similarity
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.55  # Threshold for skill matching (lowered for more lenient matching)
    
    def find_missing_skills(self, resume_skills: List[str], jd_skills: List[str]) -> List[str]:
        """
        Find missing skills by comparing job description skills with resume skills.
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of skills from job description
            
        Returns:
            List of missing skills (skills from JD that don't match resume)
        """
        missing_skills = []
        
        print(f"Comparing {len(jd_skills)} job skills with {len(resume_skills)} resume skills...")
        
        for jd_skill in jd_skills:
            best_similarity = 0.0
            
            # Find best matching resume skill
            for resume_skill in resume_skills:
                similarity = self.embedding_model.similarity(
                    jd_skill.lower(), 
                    resume_skill.lower()
                )
                best_similarity = max(best_similarity, similarity)
            
            # If best similarity is below threshold, skill is missing
            if best_similarity < self.similarity_threshold:
                missing_skills.append(jd_skill)
                print(f"  ⚠ Missing: '{jd_skill}' (best match: {best_similarity:.2f})")
            else:
                print(f"  ✓ Found: '{jd_skill}' (similarity: {best_similarity:.2f})")
        
        return missing_skills
    
    def calculate_skill_match_score(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """
        Calculate overall skill match score between resume and job description.
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of skills from job description
            
        Returns:
            Skill match score between 0 and 1
        """
        if not jd_skills:
            return 1.0  # No JD skills means perfect match
        
        missing_skills = self.find_missing_skills(resume_skills, jd_skills)
        matched_count = len(jd_skills) - len(missing_skills)
        match_score = matched_count / len(jd_skills)
        
        return match_score
    
    def get_skill_mapping(self, resume_skills: List[str], jd_skills: List[str]) -> Dict[str, Tuple[str, float]]:
        """
        Get mapping of JD skills to best matching resume skills.
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of skills from job description
            
        Returns:
            Dictionary mapping JD skill to (best_match_resume_skill, similarity_score)
        """
        mapping = {}
        
        for jd_skill in jd_skills:
            best_match, best_score = self.embedding_model.find_best_match(
                jd_skill.lower(),
                [s.lower() for s in resume_skills]
            )
            # Find original case version
            original_match = next((s for s in resume_skills if s.lower() == best_match), best_match)
            mapping[jd_skill] = (original_match, best_score)
        
        return mapping
