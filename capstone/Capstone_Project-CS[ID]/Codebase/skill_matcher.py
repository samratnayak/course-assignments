"""
Skill Gap Detection Module

This module detects missing skills by comparing job description skills
with resume skills using both exact/partial matching and semantic similarity.
"""

from typing import List, Dict, Tuple
from embedding_model import EmbeddingModel
import re
import warnings

warnings.filterwarnings("ignore")


class SkillMatcher:
    """
    Class for skill gap detection using semantic similarity and exact/partial matching.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize SkillMatcher with embedding model.
        
        Args:
            embedding_model: EmbeddingModel instance for semantic similarity
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.50  # Threshold for semantic similarity matching
        self.exact_match_threshold = 0.8  # Threshold for exact/partial text matching
    
    def _normalize_skill(self, skill: str) -> str:
        """
        Normalize skill string for better matching.
        
        Args:
            skill: Skill string to normalize
            
        Returns:
            Normalized skill string
        """
        # Remove extra whitespace, convert to lowercase
        skill = re.sub(r'\s+', ' ', skill.strip().lower())
        # Remove common punctuation that doesn't affect meaning
        skill = re.sub(r'[()]', '', skill)
        return skill
    
    def _extract_individual_skills(self, skill_text: str) -> List[str]:
        """
        Extract individual skills from a skill text that may contain multiple skills.
        
        Args:
            skill_text: Skill text that may contain multiple skills (e.g., "Spark, Cassandra, Kafka, Redis")
            
        Returns:
            List of individual skills
        """
        # Split by common delimiters
        skills = re.split(r'[,;|/]', skill_text)
        # Clean and filter
        individual_skills = []
        for skill in skills:
            skill = skill.strip()
            if skill:
                # Remove parenthetical information but keep the main skill
                skill = re.sub(r'\s*\([^)]*\)', '', skill)
                skill = skill.strip()
                if skill and len(skill) > 1:
                    individual_skills.append(skill)
        return individual_skills if individual_skills else [skill_text]
    
    def _exact_or_partial_match(self, jd_skill: str, resume_skill: str) -> float:
        """
        Check for exact or partial match between JD skill and resume skill.
        
        Args:
            jd_skill: Job description skill
            resume_skill: Resume skill
            
        Returns:
            Match score between 0 and 1 (1.0 for exact match, 0.8+ for partial match)
        """
        jd_normalized = self._normalize_skill(jd_skill)
        resume_normalized = self._normalize_skill(resume_skill)
        
        # Exact match
        if jd_normalized == resume_normalized:
            return 1.0
        
        # Check if JD skill is contained in resume skill (e.g., "Java" in "Java Programming")
        if jd_normalized in resume_normalized:
            # Calculate containment score
            containment = len(jd_normalized) / len(resume_normalized)
            if containment >= 0.5:  # At least 50% of resume skill matches
                return 0.9
        
        # Check if resume skill is contained in JD skill
        if resume_normalized in jd_normalized:
            containment = len(resume_normalized) / len(jd_normalized)
            if containment >= 0.5:
                return 0.9
        
        # Check for word-level matches (e.g., "Big Data" matches "Big Data Technologies")
        jd_words = set(jd_normalized.split())
        resume_words = set(resume_normalized.split())
        
        if jd_words and resume_words:
            # Calculate word overlap
            common_words = jd_words.intersection(resume_words)
            if common_words:
                # If significant words match, consider it a partial match
                word_overlap = len(common_words) / max(len(jd_words), len(resume_words))
                if word_overlap >= 0.6:  # 60% word overlap
                    return 0.85
        
        return 0.0
    
    def _match_skill_with_technology_groups(self, jd_skill: str, resume_skills: List[str]) -> Tuple[float, str]:
        """
        Match JD skill that may be a technology group (e.g., "Big Data Technologies (Spark, Cassandra)")
        against resume skills.
        
        Args:
            jd_skill: Job description skill (may contain technology groups)
            resume_skills: List of resume skills
            
        Returns:
            Tuple of (best_match_score, best_matching_resume_skill)
        """
        best_score = 0.0
        best_match = ""
        
        # Extract individual technologies from JD skill if it's a group
        jd_individual_skills = self._extract_individual_skills(jd_skill)
        
        # Check each resume skill
        for resume_skill in resume_skills:
            # Extract individual technologies from resume skill
            resume_individual_skills = self._extract_individual_skills(resume_skill)
            
            # Check for matches between individual skills
            matches_found = 0
            for jd_ind in jd_individual_skills:
                for resume_ind in resume_individual_skills:
                    # Check exact/partial match
                    exact_score = self._exact_or_partial_match(jd_ind, resume_ind)
                    if exact_score > 0.8:
                        matches_found += 1
                        break
            
            # Calculate match score based on how many technologies matched
            if matches_found > 0:
                match_score = matches_found / max(len(jd_individual_skills), len(resume_individual_skills))
                # Boost score if multiple technologies match
                if match_score > best_score:
                    best_score = match_score
                    best_match = resume_skill
        
        return best_score, best_match
    
    def find_missing_skills(self, resume_skills: List[str], jd_skills: List[str]) -> List[str]:
        """
        Find missing skills by comparing job description skills with resume skills.
        Uses both exact/partial matching and semantic similarity.
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of skills from job description
            
        Returns:
            List of missing skills (skills from JD that don't match resume)
        """
        missing_skills = []
        
        # Expand resume skills to include individual technologies from groups
        expanded_resume_skills = []
        for skill in resume_skills:
            expanded_resume_skills.append(skill)  # Keep original
            # Also add individual skills from groups
            individual = self._extract_individual_skills(skill)
            expanded_resume_skills.extend(individual)
        
        # Remove duplicates while preserving order
        seen = set()
        expanded_resume_skills = [s for s in expanded_resume_skills if not (s.lower() in seen or seen.add(s.lower()))]
        
        print(f"Comparing {len(jd_skills)} job skills with {len(resume_skills)} resume skills "
              f"({len(expanded_resume_skills)} expanded)...")
        
        for jd_skill in jd_skills:
            best_score = 0.0
            best_match = ""
            match_type = ""
            
            # First, try exact/partial matching (faster and more reliable for exact matches)
            for resume_skill in expanded_resume_skills:
                exact_score = self._exact_or_partial_match(jd_skill, resume_skill)
                if exact_score > best_score:
                    best_score = exact_score
                    best_match = resume_skill
                    match_type = "exact/partial"
            
            # If exact match is good enough, use it
            if best_score >= self.exact_match_threshold:
                print(f"  ✓ Found: '{jd_skill}' (exact/partial match: {best_score:.2f} with '{best_match}')")
                continue
            
            # Try technology group matching
            group_score, group_match = self._match_skill_with_technology_groups(jd_skill, resume_skills)
            if group_score > best_score:
                best_score = group_score
                best_match = group_match
                match_type = "technology group"
            
            # If still no good match, try semantic similarity
            if best_score < self.exact_match_threshold:
                for resume_skill in resume_skills:
                    try:
                        similarity = self.embedding_model.similarity(
                            jd_skill.lower(), 
                            resume_skill.lower()
                        )
                        if similarity > best_score:
                            best_score = similarity
                            best_match = resume_skill
                            match_type = "semantic"
                    except Exception as e:
                        # If semantic matching fails, continue with other methods
                        continue
            
            # Determine if skill is missing
            if best_score < self.similarity_threshold:
                missing_skills.append(jd_skill)
                print(f"  ⚠ Missing: '{jd_skill}' (best match: {best_score:.2f} with '{best_match}' via {match_type})")
            else:
                print(f"  ✓ Found: '{jd_skill}' (match: {best_score:.2f} with '{best_match}' via {match_type})")
        
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
