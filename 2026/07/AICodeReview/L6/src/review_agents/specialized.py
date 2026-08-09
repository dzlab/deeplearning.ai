"""
Specialized review agents for Lab 3.

Key insight: Focused agents with domain expertise outperform general-purpose agents.
- Security Agent: Expert in security vulnerabilities (OWASP, attack vectors)
- Pattern Agent: Expert in codebase patterns and consistency
- Ensemble: Combines both agents for maximum coverage
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# Security-focused prompt
SECURITY_AGENT_PROMPT = """You are a security expert specializing in application security vulnerabilities.

Your ONLY focus: Find security vulnerabilities in the code changes.

PR Title: {title}

{task_section}

Code Changes:
{diff}

{context_section}

CRITICAL SECURITY ANALYSIS:

1. If TASK REQUIREMENTS are provided:
   - Check if security requirements (marked as CRITICAL or REQUIRED) are implemented
   - Verify that security features mentioned in the task are present in the code

2. If EXISTING CODEBASE PATTERNS are provided:
   - Identify how security is handled elsewhere (auth, SQL, secrets, crypto, validation)
   - Check if new code follows or violates these security patterns

3. Core security vulnerabilities to check:
   - SQL Injection: Look for string formatting/concatenation in SQL queries (f-strings, %, +)
   - XSS: Look for unescaped user input in HTML/templates
   - Authentication bypass: Look for missing authentication checks
   - Hardcoded secrets: Look for API keys, passwords, tokens in code
   - Weak cryptography: Look for weak random generators, hash functions
   - Path traversal: Look for unsanitized file paths
   - Insecure deserialization: Look for pickle.loads, eval, exec with user input
   - CORS misconfiguration: Look for overly permissive CORS settings
   - Timing attacks: Look for == comparison of secrets/tokens instead of hmac.compare_digest
   - Open redirects: Look for redirect URLs that aren't validated against a whitelist
   - Information disclosure: Look for raw exception details, stack traces, or internal
     details (file paths, SQL, config) leaking into user-facing responses or logs
   - Mass assignment / privilege escalation: Look for client-controlled fields
     (e.g. is_admin, role) accepted without server-side restriction
   - Sensitive data in logs: Look for passwords, tokens, or secrets written to log statements

For each SECURITY VULNERABILITY found, respond in this exact format:

ISSUE: <brief security-focused description>
SEVERITY: <high|medium|low>

IMPORTANT:
- Only report SECURITY vulnerabilities, not code quality or style issues
- If new code violates a security pattern from the codebase, report it
- If task requirements specify security features that are missing, report it
- Be specific about the vulnerability type (SQL injection, XSS, etc.)
"""


# Pattern compliance focused prompt
PATTERN_AGENT_PROMPT = """You are a code consistency expert specializing in codebase pattern enforcement.

Your ONLY focus: Find violations of established codebase patterns and requirements.

PR Title: {title}

{task_section}

Code Changes:
{diff}

{context_section}

CRITICAL PATTERN ANALYSIS:

1. If TASK REQUIREMENTS are provided:
   - Check if ALL required features/patterns mentioned are implemented
   - Verify compliance with specified patterns (error handling, validation, etc.)

2. If EXISTING CODEBASE PATTERNS are provided:
   - Learn the established patterns for: auth, database, error handling, validation, logging
   - Identify any deviations from these established patterns in the new code

3. Core patterns to verify:
   - Authentication: Are auth decorators/dependencies used consistently?
   - Database queries: Is the parameterized query pattern followed?
   - Error handling: Is the try-except pattern applied where required?
   - Input validation: Is validation applied consistently?
   - Secrets management: Are environment variables used for config?
   - Locking/transactions: Are FOR UPDATE locks used for critical operations?
   - Rate limiting: Are rate limiters applied to sensitive endpoints?
   - Logging: Is the logging pattern followed (error logged, generic message shown)?

For each PATTERN VIOLATION found, respond in this exact format:

ISSUE: <brief description of pattern violation>
SEVERITY: <high|medium|low>

IMPORTANT:
- Only report violations of ESTABLISHED PATTERNS or TASK REQUIREMENTS
- If the codebase shows a consistent way of doing X, flag when new code does it differently
- If task explicitly requires pattern Y, flag when it's missing
- Focus on consistency and compliance, not personal preferences
"""


def parse_findings(content: str) -> list:
    """
    Parse the review output into structured findings.

    Handles multiple response formats from GPT.
    """
    findings = []
    lines = content.strip().split('\n')

    current_issue = None
    current_severity = None

    for line in lines:
        line = line.strip()

        # Remove markdown formatting
        cleaned_line = line.replace('###', '').replace('**', '').strip()

        # Check for ISSUE line
        if cleaned_line.upper().startswith('ISSUE:'):
            current_issue = cleaned_line[6:].strip()

        # Check for SEVERITY line
        elif cleaned_line.upper().startswith('SEVERITY:'):
            severity_text = cleaned_line[9:].strip()
            current_severity = severity_text.lower()

            # Save the finding
            if current_issue and current_severity:
                findings.append({
                    'issue': current_issue,
                    'severity': current_severity
                })
                current_issue = None
                current_severity = None

    return findings


def review_security(pr: dict, context: str = None, task_context: str = None) -> dict:
    """
    Security-focused review agent.

    Specializes in finding security vulnerabilities like SQL injection, XSS, auth bypass, etc.

    Args:
        pr: PR dictionary with title, diff, expected_issues
        context: Code context from retrieval engine
        task_context: Task requirements/description

    Returns:
        Dict with findings list and metadata
    """
    has_context = context and context.strip()
    has_task = task_context and task_context.strip()

    # Build task section
    task_section = f"TASK REQUIREMENTS:\n{task_context}" if has_task else ""

    # Build context section
    context_section = f"EXISTING CODEBASE PATTERNS (Security patterns used elsewhere):\n{context}" if has_context else ""

    # Format prompt
    prompt = SECURITY_AGENT_PROMPT.format(
        title=pr['title'],
        diff=pr['diff'],
        task_section=task_section,
        context_section=context_section
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert security vulnerability analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800
        )

        content = response.choices[0].message.content
        findings = parse_findings(content)

        return {
            'agent': 'security',
            'findings': findings,
            'pr_id': pr.get('id', 'unknown')
        }

    except Exception as e:
        print(f"Error in security review: {e}")
        return {
            'agent': 'security',
            'findings': [],
            'pr_id': pr.get('id', 'unknown'),
            'error': str(e)
        }


def review_pattern(pr: dict, context: str = None, task_context: str = None) -> dict:
    """
    Pattern compliance review agent.

    Specializes in finding violations of established codebase patterns and task requirements.

    Args:
        pr: PR dictionary with title, diff, expected_issues
        context: Code context from retrieval engine
        task_context: Task requirements/description

    Returns:
        Dict with findings list and metadata
    """
    has_context = context and context.strip()
    has_task = task_context and task_context.strip()

    # Build task section
    task_section = f"TASK REQUIREMENTS:\n{task_context}" if has_task else ""

    # Build context section
    context_section = f"EXISTING CODEBASE PATTERNS (How similar features are implemented):\n{context}" if has_context else ""

    # Format prompt
    prompt = PATTERN_AGENT_PROMPT.format(
        title=pr['title'],
        diff=pr['diff'],
        task_section=task_section,
        context_section=context_section
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in code consistency and pattern enforcement."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800
        )

        content = response.choices[0].message.content
        findings = parse_findings(content)

        return {
            'agent': 'pattern',
            'findings': findings,
            'pr_id': pr.get('id', 'unknown')
        }

    except Exception as e:
        print(f"Error in pattern review: {e}")
        return {
            'agent': 'pattern',
            'findings': [],
            'pr_id': pr.get('id', 'unknown'),
            'error': str(e)
        }


def deduplicate_findings(findings_list: list) -> list:
    """
    Merge and deduplicate findings from multiple agents using aggressive semantic matching.

    Keeps only high-quality, unique findings.

    Args:
        findings_list: List of finding dicts with 'issue' and 'severity' keys

    Returns:
        Deduplicated list of findings
    """
    if not findings_list:
        return []

    # Filter: Only keep high and medium severity findings
    filtered_findings = [f for f in findings_list if f.get('severity', 'low') in ['high', 'medium']]

    if not filtered_findings:
        return []

    # Aggressive deduplication using keyword overlap
    seen = []
    unique_findings = []

    for finding in filtered_findings:
        issue_text = finding['issue'].lower().strip()
        issue_words = set(word for word in issue_text.split() if len(word) > 3)  # Only meaningful words

        # Check if this is a duplicate of an existing finding
        is_duplicate = False
        for existing in seen:
            existing_words = set(word for word in existing.lower().split() if len(word) > 3)

            if not issue_words or not existing_words:
                continue

            # Calculate bidirectional overlap
            overlap_ratio = len(issue_words & existing_words) / max(len(issue_words), len(existing_words))

            # More aggressive threshold: 50% overlap = duplicate
            if overlap_ratio > 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            seen.append(issue_text)
            unique_findings.append(finding)

    return unique_findings


def _already_covered(finding: dict, existing_findings: list, threshold: float = 0.5) -> bool:
    """Check whether a finding overlaps enough with an existing finding to count as a duplicate."""
    finding_words = set(finding['issue'].lower().split())
    for existing in existing_findings:
        existing_words = set(existing['issue'].lower().split())
        overlap = len(existing_words & finding_words) / max(len(finding_words), 1)
        if overlap > threshold:
            return True
    return False


def review_ensemble(pr: dict, context: str = None, task_context: str = None) -> dict:
    """
    Ensemble review using multiple specialized agents with intelligent combination.

    Strategy:
    1. Run both specialized agents
    2. Prioritize findings found by BOTH agents (higher confidence)
    3. Add at most one unique high/medium-severity finding per specialist
       as a bounded recall safety net
    4. Deduplicate aggressively

    Note: each specialist's prompt is scoped to its domain, but both still
    tend to speculatively report tangential issues (e.g. the pattern agent
    flagging "missing auth" on unrelated PRs). Adding *all* unique findings
    from both agents unconditionally floods the result with this noise and
    tanks precision. Capping uniques to each specialist's single best
    remaining finding keeps agreement (the high-precision signal) as the
    core of the ensemble while still catching genuine one-sided finds.

    Args:
        pr: PR dictionary
        context: Code context from retrieval engine
        task_context: Task requirements

    Returns:
        Combined review results with high-confidence findings
    """
    # Run both specialized agents
    security_review = review_security(pr, context, task_context)
    pattern_review = review_pattern(pr, context, task_context)

    security_findings = security_review['findings']
    pattern_findings = pattern_review['findings']

    # Strategy 1: Find findings that both agents agree on (high confidence)
    agreed_findings = []
    for sec_finding in security_findings:
        for pat_finding in pattern_findings:
            # Check if they're talking about the same issue
            sec_words = set(sec_finding['issue'].lower().split())
            pat_words = set(pat_finding['issue'].lower().split())
            overlap = len(sec_words & pat_words) / max(len(sec_words), len(pat_words), 1)

            if overlap > 0.4:  # Both agents found similar issue
                # Keep the more detailed one
                if len(sec_finding['issue']) > len(pat_finding['issue']):
                    agreed_findings.append(sec_finding)
                else:
                    agreed_findings.append(pat_finding)
                break

    # Strategy 2: Add each specialist's single best unique finding (if any) as a
    # bounded recall safety net, instead of every speculative finding they produced
    for findings in (security_findings, pattern_findings):
        for finding in findings:
            if finding.get('severity') in ('high', 'medium') and not _already_covered(finding, agreed_findings):
                agreed_findings.append(finding)
                break

    # Final deduplication
    unique_findings = deduplicate_findings(agreed_findings)

    return {
        'agent': 'ensemble',
        'findings': unique_findings,
        'pr_id': pr.get('id', 'unknown'),
        'security_count': len(security_findings),
        'pattern_count': len(pattern_findings),
        'agreed_count': len(agreed_findings),
        'final_count': len(unique_findings)
    }
