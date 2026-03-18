"""
Resume Optimizer Module with Self-Improvement Loop

This module implements automatic resume optimization with ATS scoring
and self-improvement loop until target score is achieved.
"""

from typing import Dict, List, Optional
from cv_generator import CVGenerator
from ats_scorer import ATSScorer
from skill_matcher import SkillMatcher
import warnings

warnings.filterwarnings("ignore")


class ResumeOptimizer:
    """
    Class for resume optimization with self-improvement loop.
    """
    
    def __init__(self, cv_generator: CVGenerator, ats_scorer: ATSScorer, 
                 skill_matcher: SkillMatcher, target_score: float = 0.8, max_iterations: int = 3):
        """
        Initialize ResumeOptimizer.
        
        Args:
            cv_generator: CVGenerator instance (LLM 2 - Stronger Model)
            ats_scorer: ATSScorer instance
            skill_matcher: SkillMatcher instance
            target_score: Target ATS score (default: 0.8)
            max_iterations: Maximum optimization iterations (default: 3)
        """
        self.cv_generator = cv_generator
        self.ats_scorer = ats_scorer
        self.skill_matcher = skill_matcher
        self.target_score = target_score
        self.max_iterations = max_iterations
    
    def optimize_with_self_improvement(self, resume_text: str, user_data: Dict,
                                       jd_text: str, jd_skills: List[str], 
                                       jd_keywords: List[str],
                                       job_requirements: Optional[Dict] = None,
                                       cv_sections: Optional[Dict[str, str]] = None) -> Dict:
        """
        Optimize resume with self-improvement loop until target score is reached.
        
        Args:
            resume_text: Initial resume text
            user_data: User information dictionary
            jd_text: Job description text
            jd_skills: Skills from job description
            jd_keywords: Keywords from job description
            job_requirements: Optional job requirements dictionary
            
        Returns:
            Dictionary with optimized resume, scores, and iteration history
        """
        # Combine CV sections into text if provided, otherwise use resume_text
        if cv_sections:
            current_resume = "\n\n".join([f"{k}\n{v}" for k, v in cv_sections.items()])
        else:
            current_resume = resume_text
        
        resume_skills = user_data.get("all_skills", []) + user_data.get("skills", [])
        iteration = 0
        history = []
        
        print("\n" + "=" * 80)
        print("SELF-IMPROVEMENT LOOP - Automatic Resume Optimization")
        print("=" * 80)
        print(f"Target ATS Score: {self.target_score}")
        print(f"Maximum Iterations: {self.max_iterations}")
        print("=" * 80)
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n[Iteration {iteration}]")
            
            # Calculate current ATS score
            score_result = self.ats_scorer.calculate_ats_score(
                current_resume, resume_skills, jd_text, jd_skills, jd_keywords
            )
            current_score = score_result["overall_score"]
            
            print(f"Current ATS Score: {current_score:.3f}")
            print(f"  - Semantic Similarity: {score_result['semantic_similarity']:.3f}")
            print(f"  - Skill Match: {score_result['skill_match']:.3f}")
            print(f"  - Keyword Presence: {score_result['keyword_presence']:.3f}")
            print(f"  - Interpretation: {self.ats_scorer.get_score_interpretation(current_score)}")
            
            history.append({
                "iteration": iteration,
                "score": current_score,
                "breakdown": score_result
            })
            
            # Check if target score reached
            if current_score >= self.target_score:
                print(f"\n✓ Target score ({self.target_score}) achieved!")
                break
            
            # Find missing skills
            missing_skills = self.skill_matcher.find_missing_skills(resume_skills, jd_skills)
            
            # Regenerate CV sections with aggressive optimization (more effective than text optimization)
            if cv_sections:
                if not missing_skills:
                    print("✓ No missing skills detected. Optimizing for better keyword presence and ATS matching...")
                else:
                    print(f"⚠ Found {len(missing_skills)} missing skills: {', '.join(missing_skills[:5])}")
                
                print("✓ Regenerating CV sections with aggressive optimization...")
                # Regenerate key sections with missing skills and job keywords
                optimized_sections = self.cv_generator.generate_tailored_cv(
                    user_data, job_requirements, missing_skills
                )
                
                # Update cv_sections with optimized versions (prioritize key sections)
                for section_name in ["Professional Summary", "Work Experience", "Skills"]:
                    if section_name in optimized_sections:
                        cv_sections[section_name] = optimized_sections[section_name]
                
                # Rebuild resume text from updated sections
                current_resume = "\n\n".join([f"{k}\n{v}" for k, v in cv_sections.items()])
            else:
                # Fallback to text-based optimization
                if not missing_skills:
                    print("✓ No missing skills detected. Optimizing for better keyword presence...")
                    optimized = self.cv_generator.optimize_resume_with_missing_skills(
                        current_resume, user_data, [], jd_skills, job_requirements
                    )
                else:
                    print(f"⚠ Found {len(missing_skills)} missing skills: {', '.join(missing_skills[:5])}")
                    optimized = self.cv_generator.optimize_resume_with_missing_skills(
                        current_resume, user_data, missing_skills, jd_skills, job_requirements
                    )
                current_resume = optimized
            
            print("✓ Resume optimized")
        
        if iteration >= self.max_iterations:
            print(f"\n⚠ Maximum iterations ({self.max_iterations}) reached")
        
        # Final score calculation
        final_score_result = self.ats_scorer.calculate_ats_score(
            current_resume, resume_skills, jd_text, jd_skills, jd_keywords
        )
        
        # Return optimized CV sections if available, otherwise return text
        result = {
            "optimized_resume": current_resume,
            "final_score": final_score_result["overall_score"],
            "final_breakdown": final_score_result,
            "iterations": iteration,
            "history": history,
            "target_achieved": final_score_result["overall_score"] >= self.target_score
        }
        
        # Include optimized CV sections if we have them
        if cv_sections:
            result["optimized_cv_sections"] = cv_sections
        
        return result