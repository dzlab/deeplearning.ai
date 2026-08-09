"""
Context-aware code reviewer - reviews changes with relevant surrounding code context.
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


DEFAULT_CONTEXT_AWARE_PROMPT = """You are a code reviewer with deep knowledge of this codebase. Your job is to review new code changes and ensure they follow the established security patterns and best practices already used in the codebase.

PR Title: {title}

NEW CODE (Changes being proposed):
{diff}

EXISTING CODEBASE PATTERNS (How we currently handle similar situations):
{context}

CRITICAL TASK:
1. Carefully study the EXISTING CODEBASE PATTERNS above - these show the CORRECT way to do things
2. Compare the NEW CODE changes against these established patterns
3. Identify any places where the NEW CODE violates or ignores the established patterns
4. Focus especially on security patterns (authentication, authorization, SQL queries, secrets, cryptography, input validation)

For EACH issue where the new code fails to follow existing patterns, respond in this exact format:

ISSUE: <specific description of what pattern is violated>
SEVERITY: <high|medium|low>

Examples of issues to catch:
- If existing code uses Depends(get_current_user) for auth but new code doesn't
- If existing code uses parameterized queries but new code uses string formatting
- If existing code uses os.getenv() for secrets but new code hardcodes them
- If existing code uses secrets module but new code uses random module

Only report issues where the new code deviates from established patterns, not minor style issues."""


def review_with_context(pr: dict, custom_prompt: str = None) -> dict:
    """
    Review a PR using the diff AND relevant surrounding context.

    Args:
        pr: Dictionary with 'title', 'diff', 'context', and 'expected_issues'
        custom_prompt: Optional custom prompt template (must include {title}, {diff}, {context} placeholders)

    Returns:
        Dictionary with 'findings' list
    """
    context = pr.get('context', '')

    prompt_template = custom_prompt if custom_prompt else DEFAULT_CONTEXT_AWARE_PROMPT
    prompt = prompt_template.format(title=pr['title'], diff=pr['diff'], context=context)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert code security reviewer with knowledge of the codebase."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        content = response.choices[0].message.content
        findings = parse_findings(content)

        return {
            'approach': 'context-aware',
            'findings': findings,
            'pr_id': pr.get('id', 'unknown')
        }

    except Exception as e:
        print(f"Error in context-aware review: {e}")
        return {
            'approach': 'context-aware',
            'findings': [],
            'pr_id': pr.get('id', 'unknown'),
            'error': str(e)
        }


def parse_findings(content: str) -> list:
    """Parse the review output into structured findings."""
    findings = []
    lines = content.strip().split('\n')

    current_issue = None
    current_severity = None

    for line in lines:
        line = line.strip()
        if line.startswith('ISSUE:'):
            current_issue = line.replace('ISSUE:', '').strip()
        elif line.startswith('SEVERITY:'):
            current_severity = line.replace('SEVERITY:', '').strip().lower()

            if current_issue and current_severity:
                findings.append({
                    'issue': current_issue,
                    'severity': current_severity
                })
                current_issue = None
                current_severity = None

    return findings
