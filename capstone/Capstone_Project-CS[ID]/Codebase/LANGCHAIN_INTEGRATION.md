# LangChain Integration Guide

## Overview

LangChain has been integrated into the CV generation system for **structured output parsing**. This provides more reliable JSON extraction and validation compared to regex-based parsing.

## Integration Status

✅ **Integrated**: Structured Output Parsing
- Location: `langchain_parsers.py`
- Used in: `cv_utils.py`, `resume_extractor.py`, `job_parser.py`
- Status: **Optional with automatic fallback**

## How It Works

### 1. Automatic Detection
The system automatically detects if LangChain is available:
- If available: Uses LangChain's `PydanticOutputParser` for reliable structured output
- If not available: Falls back to existing regex-based parsing
- **No breaking changes** - system works in both modes

### 2. Integration Points

#### A. Feedback Parsing (`cv_utils.py`)
- **Function**: `_parse_feedback_with_llm()`
- **Benefit**: More reliable parsing of user feedback into structured format
- **Fallback**: Regex-based JSON extraction

#### B. Resume Data Extraction (`resume_extractor.py`)
- **Function**: `_parse_json_extraction()`
- **Benefit**: Validated schema for resume data extraction
- **Fallback**: Regex-based JSON extraction

#### C. Job Description Parsing (`job_parser.py`)
- **Function**: `_llm_enhanced_parsing()`
- **Benefit**: Structured extraction of job requirements
- **Fallback**: Regex-based JSON extraction

#### D. Multi-Step Instructions (`cv_utils.py`)
- **Function**: `_parse_multi_step_with_llm()`
- **Benefit**: Better understanding of complex instructions
- **Fallback**: Regex-based JSON extraction

## Installation

LangChain is already in `requirements.txt`. To ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Or install LangChain components separately:
```bash
pip install langchain langchain-openai pydantic
```

## Usage

The integration is **automatic** and **transparent**:

1. **With LangChain installed:**
   - System uses structured output parsing
   - Better reliability and validation
   - Automatic schema validation

2. **Without LangChain:**
   - System falls back to regex parsing
   - All functionality preserved
   - No errors or warnings

## Verification

When you run the system, you'll see a message indicating which parser is being used:

```
✓ LLM 1 (Extraction Model): ollama
✓ LLM 2 (Optimization Model): openai
✓ LangChain structured output parser (with Pydantic validation)
```

Or if LangChain is not available:
```
ℹ Regex-based JSON parser (fallback mode)
```

## Benefits

1. **Reliability**: Pydantic models ensure valid JSON structure
2. **Type Safety**: Automatic validation of data types
3. **Error Handling**: Better error messages when parsing fails
4. **Maintainability**: Centralized parsing logic
5. **Backward Compatible**: Works with or without LangChain

## Testing

To test the integration:

1. **With LangChain:**
   ```bash
   python main.py input/sample_profile.txt -o output/test.docx
   ```
   - Should use LangChain parser
   - Check console output for parser status

2. **Without LangChain (simulate):**
   - Temporarily rename `langchain_parsers.py`
   - System should fallback to regex parsing
   - All functionality should work

## Troubleshooting

### Issue: "LangChain not available" message
**Solution**: Install LangChain:
```bash
pip install langchain langchain-openai pydantic
```

### Issue: Pydantic version conflicts
**Solution**: LangChain handles both Pydantic v1 and v2 automatically

### Issue: Parsing still fails
**Solution**: System automatically falls back to regex parsing - no action needed

## Future Enhancements

Potential future integrations (not yet implemented):
- Conversational Memory for feedback loop
- LangChain Chains for complex workflows
- LangChain Agents for dynamic CV optimization

## Files Modified

1. `langchain_parsers.py` - New module for structured parsing
2. `cv_utils.py` - Updated to use LangChain parsers
3. `resume_extractor.py` - Updated to use LangChain parsers
4. `job_parser.py` - Updated to use LangChain parsers
5. `main.py` - Added parser status display
6. `requirements.txt` - Added LangChain dependencies

## Code Example

```python
# Automatic usage - no code changes needed
from langchain_parsers import parse_feedback_response

# This function automatically:
# 1. Uses LangChain if available
# 2. Falls back to regex if not available
parsed = parse_feedback_response(llm_response)
```

## Summary

✅ **Integration Complete**
- Zero breaking changes
- Automatic fallback
- Improved reliability
- Transparent to end users

The system now benefits from LangChain's structured output parsing while maintaining full backward compatibility.
