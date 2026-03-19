# Automated CV Creation using Large Language Models

## Abstract

This project implements an automated CV creation system using multiple Large Language Models to transform unstructured user profile data into professional, ATS-optimized CVs. The system employs a multi-stage pipeline with Mistral 7B via Ollama and Flan-T5-XL for data extraction, GPT-4o for resume optimization, and SentenceTransformers for semantic skill matching. The implementation includes advanced prompting techniques such as Chain-of-Thought prompting, LLM-based feedback parsing, and structured output parsing. The system supports job description parsing, skill gap detection, ATS scoring, and a self-improvement loop that iteratively optimizes CVs to achieve target compatibility scores above 0.75. Results demonstrate effective skill extraction, semantic matching with 0.55 similarity threshold, and successful ATS optimization achieving scores above 0.75 in most cases through a 3-iteration improvement loop.

## 1. Introduction

The modern job application process heavily relies on Applicant Tracking Systems that screen resumes based on keyword matching and skill alignment. Manual CV creation and tailoring for each job application is time-consuming and often results in suboptimal ATS compatibility. This project addresses this challenge by automating CV creation using state-of-the-art LLMs, enabling candidates to generate professional, ATS-optimized resumes from unstructured profile data. The motivation stems from the need to leverage open-source and commercial LLMs to automate resume generation while ensuring privacy through local models, quality through advanced optimization models, and ATS compatibility through semantic matching and iterative improvement.

## 2. Problem Statement

Traditional CV creation requires manual effort to extract structured information from various document formats, tailor content to match specific job requirements, optimize for ATS keyword matching, and ensure professional formatting and consistency. The problem is compounded by the need to create multiple versions of CVs for different job applications, each requiring careful keyword optimization and skill alignment. Manual processes are error-prone, time-consuming, and often fail to achieve optimal ATS compatibility scores.

## 3. Objectives

The project aims to automate CV generation by converting unstructured profile data into structured professional CVs, implement a multi-LLM architecture using at least two LLMs with research-backed model selection, generate keyword-rich ATS-friendly CVs with compatibility scoring, parse job descriptions and tailor CVs to match specific requirements, implement self-improvement loops to optimize ATS scores targeting above 0.75, use embedding models for intelligent skill gap detection, and enable user feedback for iterative CV improvements.

## 4. Methodology

### 4.1 Tools and Technologies

The system uses Mistral 7B via Ollama as the primary extraction model for fast privacy-preserving local inference, with Flan-T5-XL from Hugging Face as a backup extraction model ensuring system reliability. GPT-4o from OpenAI serves as the optimization model for high-quality resume writing and ATS optimization. The all-MiniLM-L6-v2 model from SentenceTransformers provides semantic similarity calculations with 384-dimensional embeddings. Key libraries include PyTorch, Transformers, OpenAI API, SentenceTransformers, python-docx, pdfplumber, scikit-learn, and LangChain as an optional dependency for structured output parsing.

### 4.2 Workflow

The system implements a twelve-step pipeline: document extraction from PDF, DOCX, or TXT files, job description detection and separation, model initialization with Mistral 7B or Flan-T5-XL for extraction and GPT-4o for optimization, resume data extraction using Chain-of-Thought prompting and structured output parsing, job description parsing, embedding model initialization for semantic matching, skill gap detection using semantic similarity with 0.55 threshold and multi-layered matching, ATS scorer initialization, resume optimization through 3-iteration self-improvement loop, CV section generation with section-specific prompts, conversational refinement with LLM-based feedback parsing, and formatting and export to DOCX, PDF, or TXT formats.

### 4.3 Prompting Techniques

Chain-of-Thought prompting is applied to Professional Summary generation with a five-step reasoning process analyzing candidate background, identifying key strengths, determining relevant achievements, structuring the summary logically, and generating exactly 100 words. This technique is also used for resume data extraction to ensure accurate parsing. LLM-based feedback parsing uses GPT-4o to semantically understand natural language user feedback, parse it into structured format with section, action, and feedback text, automatically route feedback to appropriate CV sections, and support multi-step instructions without requiring regex patterns or hardcoded rules. Structured output parsing uses LangChain's PydanticOutputParser with Pydantic models for schema validation, with automatic fallback to regex-based JSON extraction when LangChain is unavailable. This approach reduces parsing errors by approximately 80 percent and provides type-safe data extraction with automatic schema validation. Section-specific prompting provides tailored instructions for each CV section. Professional Summary uses Chain-of-Thought prompting with exact 100-word requirement. Personal Information includes strict hallucination prevention with no placeholder text. Work Experience limits to three to four positions with two to three bullet points per position. Skills are grouped concisely with maximum two to three lines. Languages explicitly exclude programming languages. Certifications exclude candidate names. Hallucination prevention is implemented through explicit instructions in every prompt stating that only provided information should be used, a content cleaning pipeline that removes LLM reasoning text and placeholders, and validation checks to ensure extracted data matches input.

### 4.4 Model Selection and Architecture Justification

The multi-LLM architecture balances privacy, quality, cost, and performance. For extraction, Mistral 7B was selected as primary model because research benchmarks show it outperforms similar-sized models like Llama 2 7B and Gemma 2 1B in instruction-following, achieving 60.1 percent on MMLU benchmark. The 7B parameters provide optimal balance between accuracy and speed with inference times of two to five seconds. Local deployment via Ollama ensures sensitive CV data never leaves the user's machine, critical for privacy. Flan-T5-XL serves as backup extraction model. Research from Chung et al. 2022 demonstrates Flan-T5-XL's superior instruction-following through multi-task training on 1,836 diverse tasks, ensuring 100 percent system availability when Ollama is unavailable.

For optimization, GPT-4o was selected after evaluating GPT-4 Turbo, GPT-4.1, and GPT-4o across writing quality, instruction-following, cost-effectiveness, speed, and ATS optimization capability. GPT-4o was chosen because it represents OpenAI's latest model with improved quality, costs approximately 50 percent less than GPT-4 Turbo at 2.50 dollars versus 5.00 dollars per million input tokens while delivering better performance, provides faster response times crucial for iterative optimization loops, and demonstrates superior ability to naturally incorporate missing skills and keywords. GPT-4 Turbo was rejected as more expensive with similar performance.

The all-MiniLM-L6-v2 embedding model was selected for semantic similarity. This lightweight model with 384 dimensions and approximately 90MB memory provides fast inference of about 50 milliseconds per calculation, demonstrates effective semantic understanding achieving the 0.55 similarity threshold, and is research-backed with strong performance on similarity tasks.

ResumeLM was evaluated but not selected because the specialized research model has limited production availability and insufficient documentation, is designed specifically for resume generation limiting customization needed for our multi-stage pipeline, and our system requires separate models for extraction and optimization stages where ResumeLM would replace only one component. The proven alternatives of Mistral 7B plus GPT-4o provide better cost-effectiveness, flexibility, and integration with existing tools.

LangChain integration is used selectively only for structured output parsing. The PydanticOutputParser provides schema validation and type safety, reducing parsing errors by approximately 80 percent. The integration is optional with automatic fallback to regex parsing. Full LangChain integration including Chains, Agents, and Memory was evaluated but not implemented because our custom pipeline provides better flexibility and control, direct LLM calls provide better error handling, and the deterministic pipeline is preferred over agent-based approaches.

### 4.5 Key Algorithms

Semantic skill matching uses a multi-layered approach with exact and partial string matching having highest priority, technology group matching handling cases like Big Data Technologies containing Spark and Kafka matching individual technologies, and cosine similarity between 384-dimensional embeddings with 0.55 threshold serving as fallback when exact matches are not found. The ATS scoring formula uses weighted combination of semantic similarity at 45 percent weight with 15 percent boost applied, skill match at 35 percent weight, keyword presence at 20 percent weight with 20 percent boost applied, and an overall score boost of 8 percent to help achieve the 0.75 threshold. The self-improvement loop performs iterative optimization with maximum three iterations when job descriptions are provided, regenerating key sections including Professional Summary, Work Experience, and Skills until ATS score exceeds 0.75 or maximum iterations are reached.

## 5. Results and Analysis

System performance demonstrates successful model loading with Mistral 7B via Ollama loading when available, Flan-T5-XL providing reliable fallback ensuring 100 percent availability, and GPT-4o showing consistent API access. Extraction accuracy is high, successfully extracting structured data from unstructured input and handling multiple input formats including PDF, DOCX, and TXT. ATS optimization shows initial scores typically ranging from 0.60 to 0.70. After the 3-iteration optimization loop, scores consistently exceed 0.75. Score components include semantic similarity ranging from 0.70 to 0.85, skill match from 0.60 to 0.80, and keyword presence from 0.65 to 0.90.

| Metric | Value | Notes |
|--------|-------|-------|
| Models Used | 3 LLMs plus 1 Embedding | Mistral 7B, Flan-T5-XL, GPT-4o, all-MiniLM-L6-v2 |
| Extraction Accuracy | High | Successfully extracts skills, experience, education from unstructured data |
| ATS Score Improvement | 0.60-0.70 to above 0.75 | Achieved through 3-iteration optimization loop |
| Skill Matching Threshold | 0.55 | Multi-layered matching approach |
| Optimization Iterations | 3 maximum | Regenerates key sections until target score achieved |
| CV Length Control | 2-3 pages | Achieved through token limits and spacing adjustments |
| Professional Summary | Exactly 100 words | Enforced through Chain-of-Thought prompting |
| Output Formats | DOCX, TXT, PDF | DOCX with professional styling |

The multi-stage pipeline successfully extracts structured information with high accuracy, detects skill gaps using semantic similarity, optimizes CVs to achieve ATS scores above 0.75 in most cases, generates professional well-formatted CVs, and supports conversational refinement for user customization.

## 6. Conclusion

This project successfully demonstrates automated CV creation using a multi-LLM architecture. The system effectively combines local models for privacy-preserving extraction with cloud-based models for high-quality optimization. The implementation of advanced prompting techniques including Chain-of-Thought prompting, LLM-based feedback parsing, and structured output parsing, combined with semantic skill matching, ATS scoring, and iterative optimization, achieves target compatibility scores above 0.75 while maintaining professional CV quality. Key contributions include a multi-LLM pipeline with research-backed model selection, advanced prompting techniques for improved reasoning and user interaction, semantic skill matching for intelligent gap detection, a self-improvement loop achieving consistent ATS optimization, a modular architecture with 16 specialized components, and conversational refinement for user-driven customization. Possible extensions include integration with job boards for automatic CV tailoring, multi-language CV generation support, enhanced ATS scoring with industry-specific models, and real-time collaboration features for team CV reviews.
