"""
Unified issue matching logic used for both display and evaluation.
Uses hybrid approach: keyword matching + embedding similarity for better accuracy.
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_embedding(text: str) -> list:
    """
    Get OpenAI embedding for text.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Faster and cheaper than 3-large
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Warning: Embedding failed for '{text[:50]}...': {e}")
        return None


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score between 0 and 1
    """
    if not vec1 or not vec2:
        return 0.0

    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitudes
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def extract_keywords(text: str) -> set:
    """
    Extract meaningful keywords from text for comparison.

    Args:
        text: Text to extract keywords from

    Returns:
        Set of lowercase keywords (words > 3 characters)
    """
    keywords = set()
    # Split on common delimiters and filter short words
    for word in text.replace('-', ' ').replace('_', ' ').split():
        if len(word) > 3:  # Skip short words like "the", "and", "not"
            keywords.add(word.lower())
    return keywords


def issues_match(expected_issue: str, found_issue: str, threshold: float = 0.3, use_embeddings: bool = True) -> bool:
    """
    Determine if a found issue matches an expected issue.

    Uses hybrid approach:
    1. Keyword overlap matching (fast)
    2. Substring matching
    3. Embedding similarity (semantic, more accurate but slower)

    Args:
        expected_issue: The issue we expected to find
        found_issue: The issue that was found by the reviewer
        threshold: Minimum keyword overlap ratio to consider a match (default 0.3)
        use_embeddings: Whether to use embedding similarity as fallback (default True)

    Returns:
        True if the issues match, False otherwise

    Examples:
        >>> issues_match("missing authentication", "lacks authentication checks")
        True  # Keywords overlap: {authentication}

        >>> issues_match("sql injection", "SQL Injection Vulnerability")
        True  # Embeddings show semantic similarity

        >>> issues_match("missing error handling", "Lack of error handling and validation")
        True  # Keywords overlap: {error, handling}
    """
    expected_lower = expected_issue.lower()
    found_lower = found_issue.lower()

    # Method 1: Extract keywords from both
    expected_keywords = extract_keywords(expected_lower)
    found_keywords = extract_keywords(found_lower)

    # Calculate keyword overlap score
    if expected_keywords:
        overlap = expected_keywords & found_keywords
        keyword_score = len(overlap) / len(expected_keywords)
    else:
        keyword_score = 0

    # Method 2: Substring matching as fallback
    # This catches cases like "XSS" in "XSS vulnerability"
    if expected_lower in found_lower or found_lower in expected_lower:
        keyword_score = max(keyword_score, 0.7)

    # If keyword matching succeeded, return early
    if keyword_score >= threshold:
        return True

    # Method 3: Embedding similarity (semantic matching)
    # Only used if keyword matching failed and embeddings are enabled
    if use_embeddings and keyword_score < threshold:
        try:
            emb1 = get_embedding(expected_issue)
            emb2 = get_embedding(found_issue)

            if emb1 and emb2:
                semantic_similarity = cosine_similarity(emb1, emb2)
                # Use threshold of 0.45 for embeddings to catch semantic matches
                # (e.g., "SQL injection" matches "violates parameterized query pattern")
                if semantic_similarity >= 0.45:
                    return True
        except Exception as e:
            # If embeddings fail, fall back to keyword score
            pass

    # Final check: keyword score
    return keyword_score >= threshold


def find_best_match(expected_issue: str, found_issues: list, threshold: float = 0.3, use_embeddings: bool = True) -> tuple:
    """
    Find the best matching issue from a list of found issues.

    Uses hybrid approach: keyword matching + embedding similarity.

    Args:
        expected_issue: The issue we're looking for
        found_issues: List of found issue strings
        threshold: Minimum score to consider a match
        use_embeddings: Whether to use embedding similarity (default True)

    Returns:
        Tuple of (best_match_index, best_score) or (None, 0) if no match
    """
    best_idx = None
    best_score = 0

    expected_lower = expected_issue.lower()
    expected_keywords = extract_keywords(expected_lower)

    # Get embedding for expected issue once (reuse for all comparisons)
    expected_emb = None
    if use_embeddings:
        try:
            expected_emb = get_embedding(expected_issue)
        except:
            pass

    for idx, found in enumerate(found_issues):
        found_lower = found.lower()
        found_keywords = extract_keywords(found_lower)

        # Method 1: Keyword overlap
        if expected_keywords:
            overlap = expected_keywords & found_keywords
            keyword_score = len(overlap) / len(expected_keywords)
        else:
            keyword_score = 0

        # Method 2: Substring matching
        if expected_lower in found_lower or found_lower in expected_lower:
            keyword_score = max(keyword_score, 0.7)

        score = keyword_score

        # Method 3: Embedding similarity (if keyword score is low)
        if use_embeddings and keyword_score < 0.5 and expected_emb:
            try:
                found_emb = get_embedding(found)
                if found_emb:
                    semantic_similarity = cosine_similarity(expected_emb, found_emb)
                    # Use threshold of 0.45 for semantic matching
                    if semantic_similarity >= 0.45:
                        score = max(score, semantic_similarity)
            except:
                pass

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score >= threshold:
        return best_idx, best_score
    return None, 0


def match_findings(findings: list, expected_issues: list, threshold: float = 0.3, use_embeddings: bool = True) -> list:
    """
    Match findings against expected issues and return matched indices.

    Args:
        findings: List of finding dicts with 'issue' key
        expected_issues: List of expected issue strings
        threshold: Minimum score to consider a match
        use_embeddings: Whether to use embedding similarity

    Returns:
        List of indices into expected_issues that were matched
    """
    matched_indices = []
    found_issue_strings = [f['issue'] for f in findings]

    for i, expected in enumerate(expected_issues):
        match_idx, score = find_best_match(expected, found_issue_strings, threshold, use_embeddings)
        if match_idx is not None:
            matched_indices.append(i)

    return matched_indices
