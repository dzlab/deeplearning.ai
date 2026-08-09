"""
Unified code reviewer that works with or without context.
The ONLY difference between diff-only and context-aware review is whether context is provided.
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


DEFAULT_REVIEW_PROMPT = """You are a code security reviewer. Your job is to review code changes and identify security vulnerabilities, bugs, and critical issues.

PR Title: {title}

{task_section}

Code Changes:
{diff}

{context_section}

{instructions}

For each issue found, respond in this exact format:

ISSUE: <brief description>
SEVERITY: <high|medium|low>

Focus on actual security vulnerabilities and bugs, not minor style issues."""


def review(pr: dict, context: str = None, task_context: str = None, custom_prompt: str = None) -> dict:
    """
    Review a PR with optional context and task information.

    The key insight: This is the SAME function, SAME model, SAME process.
    The ONLY difference is whether context and task information are provided.

    Args:
        pr: Dictionary with 'title', 'diff', and 'expected_issues'
        context: Optional context from the codebase. If provided, enables context-aware review.
                 If None or empty, performs diff-only review.
        task_context: Optional task/requirement context (Jira-like description)
        custom_prompt: Optional custom prompt template. Must include {title}, {diff},
                      {task_section}, {context_section}, and {instructions} placeholders.

    Returns:
        Dictionary with 'findings' list and 'approach' indicator
    """
    # Determine if we have context
    has_context = context and context.strip()
    has_task = task_context and task_context.strip()
    approach = 'context-aware' if has_context else 'diff-only'

    # Build task section
    if has_task:
        task_section = f"""TASK REQUIREMENTS:
{task_context}"""
    else:
        task_section = ""

    # Build context section
    if has_context:
        context_section = f"""EXISTING CODEBASE PATTERNS (How similar code is handled elsewhere):
{context}"""
    else:
        context_section = ""

    # Build instructions based on whether we have context and task info
    if has_context and has_task:
        instructions = """CRITICAL REVIEW PROCESS:
1. Read the TASK REQUIREMENTS to understand what the code SHOULD do
2. Study the EXISTING CODEBASE PATTERNS to understand HOW it should be done
3. Compare the NEW CODE changes against BOTH the requirements AND the patterns
4. Identify violations of:
   - Task security requirements (missing features from requirements)
   - Codebase patterns (not following established coding standards)
5. Focus especially on security requirements explicitly mentioned in the task

Look for:
- Missing security features listed in task requirements
- Deviations from established codebase patterns
- Requirements marked as CRITICAL that are not implemented
- Pattern violations (auth, SQL, secrets, crypto, validation, error handling)"""
    elif has_context:
        instructions = """CRITICAL TASK:
1. Carefully study the EXISTING CODEBASE PATTERNS above - these show the CORRECT way to do things
2. Compare the NEW CODE changes against these established patterns
3. Identify any places where the NEW CODE violates or ignores the established patterns
4. Focus especially on security patterns (authentication, authorization, SQL queries, secrets, cryptography, input validation)

Look for deviations from established patterns:
- If existing code uses Depends(get_current_user) for auth but new code doesn't
- If existing code uses parameterized queries but new code uses string formatting
- If existing code uses os.getenv() for secrets but new code hardcodes them
- If existing code uses secrets module but new code uses random module"""
    else:
        instructions = """Review ONLY what you can see in the code changes above. Identify security vulnerabilities, bugs, and code quality issues based solely on the changed lines."""

    # Format prompt
    prompt_template = custom_prompt if custom_prompt else DEFAULT_REVIEW_PROMPT
    prompt = prompt_template.format(
        title=pr['title'],
        diff=pr['diff'],
        task_section=task_section,
        context_section=context_section,
        instructions=instructions
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert code security reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Set to 0 for maximum consistency and reproducibility
            max_tokens=800
        )

        content = response.choices[0].message.content
        findings = parse_findings(content)

        return {
            'approach': approach,
            'findings': findings,
            'pr_id': pr.get('id', 'unknown'),
            'has_context': has_context
        }

    except Exception as e:
        print(f"Error in {approach} review: {e}")
        return {
            'approach': approach,
            'findings': [],
            'pr_id': pr.get('id', 'unknown'),
            'error': str(e),
            'has_context': has_context
        }


def parse_findings(content: str) -> list:
    """
    Parse the review output into structured findings.

    Handles multiple response formats from GPT-4o:
    - Plain format: "ISSUE: ..." and "SEVERITY: ..."
    - Markdown format: "### ISSUE: ..." and "**SEVERITY: High**"
    """
    findings = []
    lines = content.strip().split('\n')

    current_issue = None
    current_severity = None

    for line in lines:
        line = line.strip()

        # Remove markdown formatting (###, **, etc.)
        cleaned_line = line.replace('###', '').replace('**', '').strip()

        # Check for ISSUE line (case-insensitive)
        if cleaned_line.upper().startswith('ISSUE:'):
            current_issue = cleaned_line[6:].strip()  # Remove "ISSUE:" prefix

        # Check for SEVERITY line (case-insensitive)
        elif cleaned_line.upper().startswith('SEVERITY:'):
            severity_text = cleaned_line[9:].strip()  # Remove "SEVERITY:" prefix
            current_severity = severity_text.lower()

            # Save the finding if we have both issue and severity
            if current_issue and current_severity:
                findings.append({
                    'issue': current_issue,
                    'severity': current_severity
                })
                current_issue = None
                current_severity = None

    return findings
