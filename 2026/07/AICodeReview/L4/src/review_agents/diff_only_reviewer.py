"""
Diff-only code reviewer - reviews only the changed lines without context.
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


DEFAULT_DIFF_ONLY_PROMPT = """You are a code security reviewer. Review ONLY the code changes shown below and identify security vulnerabilities, bugs, and critical issues.

PR Title: {title}

Code Changes:
{diff}

Review ONLY what you can see in the diff above. Identify security vulnerabilities, bugs, and code quality issues based solely on the changed lines.

For each issue found, respond in this exact format:

ISSUE: <brief description>
SEVERITY: <high|medium|low>

Focus on actual security vulnerabilities and bugs, not minor style issues."""


def review_diff_only(pr: dict, custom_prompt: str = None) -> dict:
    """
    Review a PR using only the diff (no surrounding context).

    Args:
        pr: Dictionary with 'title', 'diff', and 'expected_issues'
        custom_prompt: Optional custom prompt template (must include {title} and {diff} placeholders)

    Returns:
        Dictionary with 'findings' list
    """
    prompt_template = custom_prompt if custom_prompt else DEFAULT_DIFF_ONLY_PROMPT
    prompt = prompt_template.format(title=pr['title'], diff=pr['diff'])

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert code security reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        content = response.choices[0].message.content
        findings = parse_findings(content)

        return {
            'approach': 'diff-only',
            'findings': findings,
            'pr_id': pr.get('id', 'unknown')
        }

    except Exception as e:
        print(f"Error in diff-only review: {e}")
        return {
            'approach': 'diff-only',
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
