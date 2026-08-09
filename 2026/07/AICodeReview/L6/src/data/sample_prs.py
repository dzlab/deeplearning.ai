"""
Sample PRs dataset for code review education.
Contains 15 synthetic PRs with various security and quality issues.
"""

SAMPLE_PRS = {
    'pr_001_auth_bypass': {
        'title': 'Add user profile update endpoint',
        'diff': '''
@@ -15,6 +15,12 @@ def get_user(user_id: int):
     return db.get_user(user_id)


+@app.put("/api/users/{user_id}")
+def update_user(user_id: int, profile: UserProfile):
+    db.update_user(user_id, profile)
+    return {"status": "success"}
+
+
 @app.delete("/api/users/{user_id}")
 def delete_user(user_id: int, current_user: User = Depends(get_current_user)):
     if current_user.id != user_id and not current_user.is_admin:
''',
        'task_context': '''
TASK-001: Add User Profile Update Endpoint

Description:
Implement PUT /api/users/{user_id} endpoint to allow users to update their profile information (name, bio, avatar, etc.)

Requirements:
- Accept user_id in URL and UserProfile object in request body
- Update user profile in database
- Return success status

Security Requirements (CRITICAL):
- User must be authenticated to update profile
- User can only update their own profile (unless admin)
- Follow existing authentication patterns used in delete_user and change_password endpoints

Additional Requirement:
- Wrap the database update in error handling, matching the try-except + logging pattern used elsewhere for write operations

Acceptance Criteria:
✓ Endpoint accepts PUT requests
✓ Profile data is validated
✓ User authentication is enforced
✓ Authorization check prevents updating other users' profiles
''',
        'context': '''
# CRITICAL SECURITY PATTERN - All user modification endpoints MUST verify authentication and authorization
# This pattern is consistently used across the entire codebase for all user-related operations:

# Example 1: Delete user endpoint (shows the REQUIRED pattern)
@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, current_user: User = Depends(get_current_user)):
    # REQUIRED: Depends(get_current_user) authenticates the request
    # REQUIRED: Authorization check ensures user can only modify themselves or admin can modify anyone
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Unauthorized")
    db.delete_user(user_id)
    return {"status": "deleted"}

# Example 2: Update email endpoint (follows same pattern)
@app.post("/api/users/{user_id}/email")
def update_email(user_id: int, new_email: str, current_user: User = Depends(get_current_user)):
    # REQUIRED: Authentication via Depends(get_current_user)
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot modify other users")
    db.update_user_email(user_id, new_email)
    return {"status": "updated"}

# Example 3: Change password endpoint (follows same pattern)
@app.post("/api/users/{user_id}/password")
def change_password(user_id: int, new_password: str, current_user: User = Depends(get_current_user)):
    # REQUIRED: Must authenticate and verify user can only change their own password
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    db.update_password(user_id, hash_password(new_password))
    return {"status": "password_changed"}

# SECURITY NOTE: ANY endpoint that modifies user data MUST include:
# 1. current_user: User = Depends(get_current_user) parameter for authentication
# 2. Authorization check: if current_user.id != user_id (and not admin check if needed)

# SECONDARY PATTERN - Wrap database writes in try-except with logging
@app.post("/api/users/{user_id}/email")
def update_email_safe(user_id: int, new_email: str, current_user: User = Depends(get_current_user)):
    try:
        db.update_user_email(user_id, new_email)
        return {"status": "updated"}
    except DatabaseError as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
# NOTE: ALL database write operations should be wrapped in try-except with logging
''',
        'expected_issues': ['violates mandatory Depends(get_current_user) authentication pattern', 'missing try-except error handling around the database write, unlike other write endpoints']
    },

    'pr_002_sql_injection': {
        'title': 'Add user search functionality',
        'diff': '''
@@ -20,6 +20,14 @@ def list_users():
     return db.get_all_users()


+@app.get("/api/users/search")
+def search_users(query: str):
+    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
+    results = db.execute(sql)
+    return results
+
+
 @app.get("/api/users/{user_id}")
 def get_user(user_id: int):
     return db.get_user(user_id)
''',
        'task_context': '''
TASK-002: Add User Search Functionality

Description:
Implement GET /api/users/search endpoint to allow searching users by name with partial matching

Requirements:
- Accept 'query' parameter with search term
- Search users table for names matching the query
- Support partial matching (LIKE pattern)
- Return list of matching users

Security Requirements (CRITICAL):
- Prevent SQL injection vulnerabilities
- Follow existing database query patterns
- Use parameterized queries like other endpoints in the codebase

Additional Requirement:
- Apply rate limiting to the search endpoint, matching the @limiter.limit() pattern used on other public-facing endpoints

Acceptance Criteria:
✓ Endpoint accepts query parameter
✓ Returns matching users
✓ Partial name matching works
✓ SQL queries use safe parameterized approach
''',
        'context': '''
# CRITICAL SECURITY PATTERN - NEVER use string formatting for SQL queries
# The codebase consistently uses parameterized queries to prevent SQL injection:

# Example 1: Getting user by ID (safe parameterized query)
@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    # SAFE: Uses ORM/parameterized query, NOT string formatting
    return db.get_user(user_id)

# Example 2: Finding users by email (safe parameterized pattern)
@app.get("/api/users/by-email")
def find_user_by_email(email: str):
    # SAFE: db.query uses parameterized queries internally
    user = db.query("SELECT * FROM users WHERE email = ?", (email,))
    return user

# Example 3: Filtering posts by tag (safe pattern)
@app.get("/api/posts/by-tag")
def get_posts_by_tag(tag: str):
    # SAFE: ORM automatically parameterizes queries
    posts = db.query_posts(tag=tag)
    return posts

# Example 4: Complex search with multiple parameters (safe pattern)
@app.get("/api/products/search")
def search_products(name: str, category: str):
    # SAFE: Uses parameterized query with named parameters
    results = db.execute(
        "SELECT * FROM products WHERE name LIKE ? AND category = ?",
        (f"%{name}%", category)
    )
    return results

# SECURITY NOTE: ALL database queries in this codebase use:
# 1. ORM methods like db.get_user() which automatically parameterize
# 2. Parameterized queries with ? placeholders and tuple parameters
# 3. NEVER use f-strings or % formatting to build SQL queries with user input

# SECONDARY PATTERN - Rate limit public search/query endpoints
from app.auth import limiter
@app.get("/api/products/search")
@limiter.limit("20/minute")
def search_products_limited(name: str):
    return db.query_products(name)
# NOTE: search endpoints are prime targets for scraping/DoS and should be rate limited
''',
        'expected_issues': ['violates parameterized query pattern', 'missing rate limiting on the new search endpoint, unlike other public-facing endpoints']
    },

    'pr_003_hardcoded_secret': {
        'title': 'Add email notification service',
        'diff': '''
@@ -1,6 +1,7 @@
 from fastapi import FastAPI
 from typing import Optional
 import smtplib
+from email.mime.text import MIMEText

 app = FastAPI()

@@ -10,3 +11,15 @@ def send_notification(user_email: str, message: str):
+    smtp_server = "smtp.gmail.com"
+    smtp_port = 587
+    smtp_user = "notifications@company.com"
+    smtp_password = "MyP@ssw0rd123!"
+
+    server = smtplib.SMTP(smtp_server, smtp_port)
+    server.login(smtp_user, smtp_password)
+    server.sendmail(smtp_user, user_email, message)
+    server.quit()
+    return {"status": "sent"}
''',
        'task_context': '''
TASK-003: Add Email Notification Service

Description:
Implement email notification functionality to send alerts to users via SMTP

Requirements:
- Create send_notification endpoint that accepts user email and message
- Connect to Gmail SMTP server
- Send email using company notification account
- Return success status after sending

Security Requirements (CRITICAL):
- SMTP credentials MUST be loaded from environment variables
- Follow existing pattern for secrets management (database, API keys, etc.)
- NEVER commit credentials to source control

Configuration:
- SMTP server: smtp.gmail.com:587
- Use environment variables for credentials
- Credentials should be in .env file (not in code)

Acceptance Criteria:
✓ Email sending functionality works
✓ SMTP credentials loaded from environment
✓ No hardcoded passwords in code
✓ Follows os.getenv() pattern used throughout codebase

Additional Requirement:
- Wrap the SMTP connection/send in try-except with logging, matching the error handling pattern used for other external calls
''',
        'context': '''
# CRITICAL SECURITY PATTERN - NEVER hardcode credentials, API keys, or secrets in code
# The codebase consistently loads ALL sensitive configuration from environment variables:

# Example 1: Database connection (at application startup)
import os
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
db_engine = create_engine(DATABASE_URL, password=DATABASE_PASSWORD)

# Example 2: External API authentication
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
stripe.api_key = STRIPE_API_KEY

# Example 3: JWT secret key
JWT_SECRET = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"

def create_token(user_id: int):
    return jwt.encode({"user_id": user_id}, JWT_SECRET, algorithm=JWT_ALGORITHM)

# Example 4: AWS credentials
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Example 5: Email service configuration
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
sg = SendGridAPIClient(SENDGRID_API_KEY)

# SECURITY NOTE: The entire codebase follows this pattern:
# 1. ALL passwords, API keys, tokens stored in environment variables
# 2. Use os.getenv() to load at runtime
# 3. NEVER commit credentials to version control
# 4. See .env.example for required environment variables

# SECONDARY PATTERN - Wrap external service calls (SMTP, HTTP, etc.) in try-except with logging
def send_notification_safe(user_email: str, message: str):
    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, user_email, message)
        server.quit()
        return {"status": "sent"}
    except smtplib.SMTPException as e:
        logger.error(f"Failed to send notification to {user_email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")
# NOTE: External service calls (SMTP, HTTP, etc.) should always be wrapped in try-except
''',
        'expected_issues': ['violates environment variable pattern for secrets', 'missing try-except error handling around the SMTP call, unlike other external service calls']
    },

    'pr_004_missing_error_handling': {
        'title': 'Add file upload endpoint',
        'diff': '''
@@ -15,3 +15,11 @@ def get_document(doc_id: int):
     return db.get_document(doc_id)
+
+
+@app.post("/api/upload")
+def upload_file(file: UploadFile):
+    content = file.file.read()
+    save_path = f"./uploads/{file.filename}"
+    db.save_file(save_path, content)
+    return {"filename": file.filename, "size": len(content)}
''',
        'task_context': '''
TASK-004: Add File Upload Endpoint

Description:
Implement POST /api/upload endpoint to allow users to upload files to the server

Requirements:
- Accept file uploads via UploadFile parameter
- Save uploaded files to database
- Return filename and file size in response

Security & Quality Requirements (CRITICAL):
- Implement error handling for file I/O operations
- Validate file type (only allow specific extensions)
- Validate file size (prevent DOS attacks from large files)
- Validate filename (prevent path traversal attacks)
- Check for empty files
- Follow existing file upload patterns in the codebase

Acceptance Criteria:
✓ File upload functionality works
✓ Try-except blocks handle I/O errors
✓ File type validation implemented
✓ File size limits enforced
✓ Filename validation prevents security issues
✓ Matches error handling pattern from other upload endpoints

Additional Requirement:
- Validate and sanitize the filename before constructing a filesystem path, matching the path traversal protection pattern used by other file access endpoints
''',
        'context': '''
# CRITICAL PATTERN - ALL file operations MUST include comprehensive error handling and validation
# The codebase consistently wraps file operations in try-except blocks with proper validation:

# Example 1: Document retrieval with error handling
@app.get("/api/document/{doc_id}")
def get_document(doc_id: int):
    try:
        doc = db.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Example 2: Image upload with validation and error handling
@app.post("/api/images/upload")
def upload_image(file: UploadFile):
    try:
        # REQUIRED: Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only images allowed")

        # REQUIRED: Validate file size
        content = file.file.read()
        if len(content) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(status_code=400, detail="File too large")

        # REQUIRED: Validate filename
        if not file.filename or '..' in file.filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        db.save_image(file.filename, content)
        return {"filename": file.filename, "size": len(content)}
    except IOError as e:
        logger.error(f"File IO error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

# Example 3: CSV file upload with validation
@app.post("/api/data/upload")
def upload_csv(file: UploadFile):
    try:
        # REQUIRED: Validate file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")

        content = file.file.read()

        # REQUIRED: Validate content is not empty
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        db.save_csv(file.filename, content)
        return {"status": "uploaded"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

# Example 4: PDF upload with comprehensive checks
@app.post("/api/documents/upload")
def upload_pdf(file: UploadFile):
    try:
        # REQUIRED: Multiple validation checks
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        if len(file.filename) > 255:
            raise HTTPException(status_code=400, detail="Filename too long")

        content = file.file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File cannot be empty")

        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size exceeds limit")

        db.save_document(file.filename, content)
        return {"filename": file.filename}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

# SECURITY NOTE: ALL file upload endpoints MUST include:
# 1. try-except blocks to catch IO errors and database errors
# 2. File type validation (content_type or extension check)
# 3. File size limits to prevent DOS attacks
# 4. Filename validation to prevent path traversal
# 5. Empty file checks

# SECONDARY PATTERN - Validate file paths with os.path.abspath + startswith
UPLOAD_DIR = os.path.abspath("/var/app/uploads")

@app.post("/api/uploads/safe")
def upload_file_safe(file: UploadFile):
    safe_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
    if not safe_path.startswith(UPLOAD_DIR):
        raise HTTPException(status_code=403, detail="Invalid path")
    db.save_file(safe_path, file.file.read())
    return {"status": "uploaded"}
# NOTE: NEVER build a filesystem path directly from a client-supplied filename
''',
        'expected_issues': ['missing try-except pattern required for all file operations', 'builds the upload path directly from an unsanitized filename, missing the path traversal protection used by other file access endpoints']
    },

    'pr_005_race_condition': {
        'title': 'Optimize inventory decrement',
        'diff': '''
@@ -10,8 +10,8 @@ def purchase_item(item_id: int, quantity: int):
     inventory = db.get_inventory(item_id)
-    if inventory.quantity < quantity:
+    if inventory.quantity >= quantity:
+        inventory.quantity -= quantity
+        db.save_inventory(inventory)
-        raise HTTPException(400, "Insufficient stock")
-    inventory.quantity -= quantity
-    db.save_inventory(inventory)
     return {"status": "purchased"}
''',
        'task_context': '''
TASK-005: Optimize Inventory Decrement Logic

Description:
Refactor inventory checking logic in purchase_item endpoint to be more efficient

Requirements:
- Check if inventory quantity is sufficient before purchase
- Decrement inventory quantity after validation
- Save updated inventory to database
- Return purchase success status

Concurrency Requirements (CRITICAL):
- Handle concurrent purchase requests safely
- Prevent race conditions where multiple users buy the last item
- Follow existing inventory locking patterns from other purchase endpoints
- Ensure atomic read-modify-write operations

References:
- See how other inventory operations handle concurrency
- Review cart checkout and reservation endpoints

Acceptance Criteria:
✓ Inventory quantity checked before purchase
✓ Inventory decremented correctly
✓ Race conditions prevented
✓ Follows database locking pattern from other inventory operations

Additional Requirement:
- Require authentication on the purchase endpoint, matching the current_user pattern used by other write endpoints
''',
        'context': '''
# CRITICAL CONCURRENCY PATTERN - Always use database-level locking for inventory operations
# The codebase consistently uses FOR UPDATE locks for ALL inventory modifications:

# Example 1: Inventory purchase with locking
@app.post("/api/orders/purchase")
def purchase_product(product_id: int, quantity: int):
    # REQUIRED: Use SELECT FOR UPDATE to lock the row
    with db.transaction():
        inventory = db.get_inventory_locked(product_id)  # SELECT ... FOR UPDATE
        if inventory.quantity < quantity:
            raise HTTPException(400, "Insufficient stock")
        inventory.quantity -= quantity
        db.save_inventory(inventory)
        return {"status": "purchased"}

# Example 2: Inventory reservation with locking
@app.post("/api/cart/reserve")
def reserve_items(cart_items: list):
    with db.transaction():
        for item in cart_items:
            # REQUIRED: Lock each inventory row before modification
            inv = db.get_inventory_locked(item.product_id)
            inv.reserved += item.quantity
            db.save_inventory(inv)
    return {"status": "reserved"}

# Example 3: Inventory replenishment with locking
@app.post("/api/admin/restock")
def restock(product_id: int, amount: int):
    with db.transaction():
        # REQUIRED: Lock before any read-modify-write operation
        inventory = db.get_inventory_locked(product_id)
        inventory.quantity += amount
        db.save_inventory(inventory)
        return {"new_quantity": inventory.quantity}

# Example 4: Cart checkout with multiple inventory locks
@app.post("/api/checkout")
def checkout(cart_id: int):
    with db.transaction():
        cart_items = db.get_cart_items(cart_id)
        # REQUIRED: Lock ALL inventory rows in the transaction
        for item in cart_items:
            inv = db.get_inventory_locked(item.product_id)
            if inv.quantity < item.quantity:
                raise HTTPException(400, f"{item.name} out of stock")
            inv.quantity -= item.quantity
            db.save_inventory(inv)
    return {"status": "completed"}

# CONCURRENCY NOTE: ALL inventory operations MUST:
# 1. Use db.transaction() context manager
# 2. Use get_inventory_locked() which adds FOR UPDATE clause
# 3. This prevents race conditions in concurrent requests
# 4. NEVER do read-modify-write without locking

# SECONDARY PATTERN - Require authentication on endpoints that mutate state
@app.post("/api/orders/purchase-safe")
def purchase_product_safe(product_id: int, quantity: int, current_user: User = Depends(get_current_user)):
    with db.transaction():
        inventory = db.get_inventory_locked(product_id)
        inventory.quantity -= quantity
        db.save_inventory(inventory)
        return {"status": "purchased"}
# NOTE: ALL endpoints that write/mutate data MUST authenticate the caller
''',
        'expected_issues': ['violates mandatory FOR UPDATE locking pattern for inventory', 'missing authentication on the purchase endpoint, unlike other endpoints that mutate data']
    },

    'pr_006_xss_vulnerability': {
        'title': 'Add user comment feature',
        'diff': '''
@@ -20,3 +20,11 @@ def get_post(post_id: int):
     return db.get_post(post_id)
+
+
+@app.post("/api/posts/{post_id}/comments")
+def add_comment(post_id: int, comment: str):
+    db.add_comment(post_id, comment)
+    return {"comment": comment, "html": f"<div>{comment}</div>"}
''',
                'task_context': '''
TASK-006: Add User Comment Feature

Description:
Allow users to post comments on blog posts

Requirements:
- Accept post_id and comment text as parameters
- Save comment to database
- Return comment data with HTML-formatted version for display

Security Requirements (CRITICAL):
- Prevent XSS (Cross-Site Scripting) attacks
- Sanitize user-generated content before rendering
- Follow existing HTML rendering patterns in the codebase

References:
- See how user bios are displayed
- See how post content is rendered
- Follow the established sanitization pattern

Acceptance Criteria:
✓ Comments can be posted
✓ Comments stored in database
✓ HTML output is safe from XSS
✓ Follows html.escape() pattern from other user content endpoints

Additional Requirement:
- Reject empty comments and enforce a maximum length, following the input validation pattern used elsewhere
''',
'context': '''
# CRITICAL SECURITY PATTERN - ALWAYS escape user input before rendering in HTML
# The codebase consistently uses html.escape() for ALL user-generated content:

# Example 1: Displaying user profile bio
from html import escape

@app.get("/api/users/{user_id}/bio")
def get_user_bio(user_id: int):
    user = db.get_user(user_id)
    # REQUIRED: Escape HTML to prevent XSS
    safe_bio = escape(user.bio)
    return {"bio": safe_bio, "html": f"<div>{safe_bio}</div>"}

# Example 2: Showing post content
@app.get("/api/posts/{post_id}")
def get_post(post_id: int):
    post = db.get_post(post_id)
    # REQUIRED: Escape user content before rendering
    return {
        "title": escape(post.title),
        "content": escape(post.content),
        "html": f"<h1>{escape(post.title)}</h1><p>{escape(post.content)}</p>"
    }

# Example 3: User status updates
@app.get("/api/users/{user_id}/status")
def get_status(user_id: int):
    status = db.get_user_status(user_id)
    # REQUIRED: Always escape before inserting into HTML
    return {"status": escape(status.text)}

# Example 4: Forum posts with user names
@app.get("/api/forum/{thread_id}")
def get_thread(thread_id: int):
    posts = db.get_forum_posts(thread_id)
    # REQUIRED: Escape both username and content
    safe_posts = [
        {
            "author": escape(p.author_name),
            "content": escape(p.content)
        }
        for p in posts
    ]
    return {"posts": safe_posts}

# SECURITY NOTE: EVERY endpoint that returns user-generated content:
# 1. MUST use html.escape() on ALL user input before rendering
# 2. This prevents XSS (Cross-Site Scripting) attacks
# 3. NEVER directly interpolate user input into HTML strings

# SECONDARY PATTERN - Validate and bound user-supplied text input
class CommentInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)
# NOTE: Text fields should reject empty input and enforce a maximum length
''',
        'expected_issues': ['violates html.escape() pattern for user content', 'missing input validation on the comment content (no empty-check or length limit)']
    },

    'pr_007_weak_crypto': {
        'title': 'Add password reset tokens',
        'diff': '''
@@ -1,5 +1,6 @@
 from fastapi import FastAPI
 import hashlib
+import random

 @app.post("/api/auth/reset-password")
 def request_reset(email: str):
+    token = str(random.randint(100000, 999999))
+    db.save_reset_token(email, token)
+    send_email(email, f"Reset token: {token}")
     return {"status": "sent"}
''',
        'task_context': '''
TASK-007: Implement Password Reset Token Generation

Description:
Add functionality to generate and send password reset tokens when users forget their password

Requirements:
- Generate a unique reset token for the user
- Store token in database associated with user's email
- Send token to user's email
- Token should be easy to type (6-digit number format)

Security Requirements (CRITICAL):
- Token MUST be cryptographically secure and unpredictable
- Follow existing token generation patterns in the codebase
- Use same approach as session tokens, API keys, and verification tokens
- Token must be resistant to brute force attacks

References:
- See how login endpoint generates session tokens
- See how registration generates verification tokens
- Follow the established pattern for ALL authentication tokens

Acceptance Criteria:
✓ Token generation works
✓ Token stored in database
✓ Email sent with token
✓ Token generation uses cryptographically secure random source
✓ Matches security pattern from other token-generating endpoints

Additional Requirement:
- Apply rate limiting to the password reset request endpoint, matching the pattern used on other authentication endpoints
''',
        'context': '''
# CRITICAL SECURITY PATTERN - Always use secrets module for cryptographic operations
# The codebase consistently uses secrets.* for ALL security-sensitive random generation:

# Example 1: Session token generation
import secrets

@app.post("/api/auth/login")
def login(username: str, password: str):
    user = authenticate(username, password)
    # REQUIRED: Use secrets.token_urlsafe() for session tokens
    session_token = secrets.token_urlsafe(32)  # NOT random.randint()
    db.save_session(user.id, session_token)
    return {"token": session_token}

# Example 2: API key generation
@app.post("/api/admin/generate-api-key")
def generate_api_key(user_id: int):
    # REQUIRED: Use secrets.token_hex() for API keys
    api_key = secrets.token_hex(32)  # NOT random.random()
    db.save_api_key(user_id, api_key)
    return {"api_key": api_key}

# Example 3: Email verification tokens
@app.post("/api/auth/register")
def register(email: str, password: str):
    user = create_user(email, password)
    # REQUIRED: Use secrets for email verification
    verification_token = secrets.token_urlsafe(48)  # NOT random.choice()
    db.save_verification_token(user.id, verification_token)
    send_verification_email(email, verification_token)
    return {"user_id": user.id}

# Example 4: CSRF tokens
def generate_csrf_token():
    # REQUIRED: Use secrets for CSRF protection
    return secrets.token_urlsafe(32)

# SECURITY NOTE: NEVER use random module for security purposes:
# ❌ random.randint() - predictable, NOT cryptographically secure
# ❌ random.random() - predictable, NOT cryptographically secure
# ❌ random.choice() - predictable, NOT cryptographically secure
# ✅ secrets.token_urlsafe() - cryptographically secure
# ✅ secrets.token_hex() - cryptographically secure
# ✅ secrets.token_bytes() - cryptographically secure

# SECONDARY PATTERN - Rate limit password reset requests
@app.post("/api/auth/reset-password-safe")
@limiter.limit("3/hour")
def request_reset_safe(email: str):
    token = secrets.token_urlsafe(32)
    db.save_reset_token(email, token)
    return {"status": "sent"}
# NOTE: Password reset requests should be rate limited to prevent email bombing/abuse
''',
        'expected_issues': ['violates secrets module requirement for tokens', 'missing rate limiting on the password reset request endpoint, unlike other authentication endpoints']
    },

    'pr_008_missing_rate_limit': {
        'title': 'Add login endpoint',
        'diff': '''
@@ -10,3 +10,15 @@ def health_check():
     return {"status": "healthy"}
+
+
+import logging
+logger = logging.getLogger(__name__)
+
+
+@app.post("/api/auth/login")
+def login(username: str, password: str):
+    user = db.authenticate(username, password)
+    if user:
+        return {"token": create_token(user)}
+    logger.warning(f"Failed login attempt: {username}:{password}")
+    raise HTTPException(401, "Invalid credentials")
''',
                'task_context': '''
TASK-008: Add Login Endpoint

Description:
Implement POST /api/auth/login endpoint for user authentication

Requirements:
- Accept username and password
- Authenticate against database
- Generate and return session token on success
- Return 401 error on invalid credentials

Security Requirements (CRITICAL):
- Implement rate limiting to prevent brute force attacks
- Follow existing auth endpoint security patterns
- Protect against credential stuffing attacks

References:
- See how admin login endpoint implements rate limiting
- See how password reset endpoint prevents abuse
- Follow the @limiter.limit() pattern used on all auth endpoints

Acceptance Criteria:
✓ Login functionality works
✓ Rate limiting applied (5 attempts per minute)
✓ Brute force protection implemented
✓ Matches rate limiting pattern from other authentication endpoints

Additional Requirement:
- Never log raw credentials; log only the username (or nothing) on failed attempts
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always apply rate limiting to authentication endpoints
# The codebase consistently uses @limiter.limit() decorator for ALL auth endpoints:

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Example 1: Password login with rate limiting
@app.post("/api/auth/password-login")
@limiter.limit("5/minute")  # REQUIRED for all auth endpoints
def password_login(email: str, password: str):
    user = db.authenticate_password(email, password)
    if user:
        return {"token": create_token(user)}
    raise HTTPException(401, "Invalid credentials")

# Example 2: OTP verification with rate limiting
@app.post("/api/auth/verify-otp")
@limiter.limit("3/minute")  # REQUIRED: Even stricter for OTP
def verify_otp(phone: str, code: str):
    if db.verify_otp_code(phone, code):
        return {"token": create_token_for_phone(phone)}
    raise HTTPException(401, "Invalid OTP")

# Example 3: Password reset with rate limiting
@app.post("/api/auth/reset-password")
@limiter.limit("3/hour")  # REQUIRED: Prevent brute force
def reset_password(email: str):
    user = db.get_user_by_email(email)
    if user:
        send_reset_email(user)
    return {"status": "sent"}  # Same response even if user doesn't exist

# Example 4: API key validation with rate limiting
@app.post("/api/auth/validate-key")
@limiter.limit("10/minute")  # REQUIRED for credential validation
def validate_key(api_key: str):
    if db.is_valid_api_key(api_key):
        return {"valid": True}
    raise HTTPException(401, "Invalid API key")

# Example 5: Admin login with strict rate limiting
@app.post("/api/admin/login")
@limiter.limit("3/minute")  # REQUIRED: Stricter for admin endpoints
def admin_login(username: str, password: str):
    admin = db.authenticate_admin(username, password)
    if admin:
        return {"token": create_admin_token(admin)}
    raise HTTPException(401, "Invalid credentials")

# SECURITY NOTE: ALL authentication endpoints MUST have @limiter.limit():
# 1. Login endpoints: 5/minute to prevent brute force
# 2. OTP/2FA endpoints: 3/minute (stricter)
# 3. Password reset: 3/hour to prevent email bombing
# 4. Admin endpoints: 3/minute (stricter for elevated privileges)

# SECONDARY PATTERN - Never log raw credentials
def login_safe(username: str, password: str):
    user = db.authenticate(username, password)
    if user:
        return {"token": create_token(user)}
    logger.warning(f"Failed login attempt for username={username}")  # NEVER include password
    raise HTTPException(401, "Invalid credentials")
# NOTE: NEVER write passwords, tokens, or secrets to logs, even on failure paths
''',
        'expected_issues': ['violates mandatory @limiter.limit() pattern for auth endpoints', 'logs the raw password on failed login attempts, exposing credentials in logs']
    },

    'pr_009_insecure_deserialization': {
        'title': 'Add session management',
        'diff': '''
@@ -1,4 +1,5 @@
 from fastapi import FastAPI, Cookie
+import pickle

 @app.post("/api/session")
 def save_session(data: dict):
+    session_data = pickle.dumps(data)
+    return {"session": session_data.hex()}
+
+
+@app.get("/api/session")
+def load_session(session: str):
+    session_data = bytes.fromhex(session)
+    data = pickle.loads(session_data)
+    return data
''',
                'task_context': '''
TASK-009: Add Session Management

Description:
Implement session serialization and deserialization for storing user session data

Requirements:
- Serialize session data for storage
- Deserialize session data for retrieval
- Support saving and loading session state

Security Requirements (CRITICAL):
- Use safe serialization formats
- Prevent arbitrary code execution vulnerabilities
- Follow existing serialization patterns in the codebase

References:
- See how cache data is serialized
- See how user preferences are stored
- Follow the JSON serialization pattern used throughout

Acceptance Criteria:
✓ Session data can be serialized
✓ Session data can be deserialized
✓ Safe serialization method used (not pickle)
✓ Matches json.dumps/loads pattern from other data serialization

Additional Requirement:
- Wrap session serialization/deserialization in try-except with logging, matching the error handling pattern used elsewhere
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always use JSON for serialization, NEVER pickle
# The codebase consistently uses json.dumps/loads for ALL data serialization:

import json

# Example 1: Session data serialization
@app.post("/api/sessions/create")
def create_session(user_id: int, preferences: dict):
    session_data = {
        "user_id": user_id,
        "preferences": preferences,
        "created_at": datetime.now().isoformat()
    }
    # REQUIRED: Use json.dumps() for serialization
    serialized = json.dumps(session_data)  # NOT pickle.dumps()
    db.save_session(serialized)
    return {"session_id": generate_session_id()}

# Example 2: Deserializing session data
@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    serialized = db.get_session(session_id)
    # REQUIRED: Use json.loads() for deserialization
    session_data = json.loads(serialized)  # NOT pickle.loads()
    return session_data

# Example 3: Cache data serialization
@app.post("/api/cache/store")
def store_cache(key: str, data: dict):
    # REQUIRED: JSON for cache serialization
    cached_value = json.dumps(data)
    redis.set(key, cached_value)
    return {"status": "cached"}

@app.get("/api/cache/{key}")
def get_cache(key: str):
    cached_value = redis.get(key)
    # REQUIRED: JSON for cache deserialization
    return json.loads(cached_value)

# Example 4: User preferences storage
@app.put("/api/users/{user_id}/preferences")
def update_preferences(user_id: int, prefs: dict):
    # REQUIRED: JSON serialization for storage
    serialized_prefs = json.dumps(prefs)
    db.update_user_field(user_id, "preferences", serialized_prefs)
    return {"status": "updated"}

@app.get("/api/users/{user_id}/preferences")
def get_preferences(user_id: int):
    serialized_prefs = db.get_user_field(user_id, "preferences")
    # REQUIRED: JSON deserialization
    return json.loads(serialized_prefs)

# SECURITY NOTE: NEVER use pickle for untrusted data:
# ❌ pickle.dumps() / pickle.loads() - ALLOWS ARBITRARY CODE EXECUTION
# ❌ Can deserialize malicious objects that run code on load
# ✅ json.dumps() / json.loads() - SAFE, only handles data types
# ✅ JSON cannot execute code during deserialization

# SECONDARY PATTERN - Wrap serialization/deserialization in try-except
def load_session_safe(session: str):
    try:
        return json.loads(bytes.fromhex(session))
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load session: {e}")
        raise HTTPException(status_code=400, detail="Invalid session")
# NOTE: Serialization/deserialization calls should be wrapped in try-except
''',
        'expected_issues': ['violates mandatory json serialization pattern', 'missing try-except error handling around session serialization/deserialization']
    },

    'pr_010_missing_input_validation': {
        'title': 'Add user age field',
        'diff': '''
@@ -15,6 +15,8 @@ class UserProfile(BaseModel):
     name: str
     email: str
+    age: int
+    is_admin: bool = False

 @app.post("/api/users")
 def create_user(profile: UserProfile):
''',
                'task_context': '''
TASK-010: Add Age Field to User Profile

Description:
Extend UserProfile model to include age field for better demographics

Requirements:
- Add age field to UserProfile Pydantic model
- Accept age in user creation endpoint
- Store age in database

Data Quality Requirements (CRITICAL):
- Validate age is within reasonable range
- Prevent negative or unrealistic ages
- Follow existing field validation patterns

References:
- See how other numeric fields are validated (height, weight, rating)
- Follow Field() constraint pattern used in Product and Discount models

Acceptance Criteria:
✓ Age field added to model
✓ Age validation implemented
✓ Age constraints prevent invalid values (realistic age range)
✓ Matches Field(ge=, le=) pattern from other integer fields

Additional Requirement:
- Never trust client-supplied privilege fields (e.g. is_admin); such fields must be server-controlled
''',
'context': '''
# CRITICAL DATA VALIDATION PATTERN - All numeric fields MUST have Field() constraints
# The codebase consistently uses Field() with ge/le/gt/lt for ALL integer/float fields:

from pydantic import BaseModel, Field, EmailStr

# Example 1: User profile with validated fields
class UserProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    # REQUIRED: All integers must have Field() with range constraints
    height_cm: int = Field(..., ge=50, le=300)  # 50-300 cm
    weight_kg: float = Field(..., ge=20.0, le=500.0)  # 20-500 kg

# Example 2: Product model with validated price
class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    # REQUIRED: Price must have minimum value
    price: float = Field(..., ge=0.01)  # Must be positive
    # REQUIRED: Stock must be non-negative
    stock: int = Field(..., ge=0)
    # REQUIRED: Rating must be in valid range
    rating: float = Field(..., ge=0.0, le=5.0)

# Example 3: Discount configuration with constraints
class Discount(BaseModel):
    code: str = Field(..., min_length=3, max_length=20)
    # REQUIRED: Percentage must be between 0-100
    percentage: int = Field(..., ge=0, le=100)
    # REQUIRED: Max uses must be positive
    max_uses: int = Field(..., ge=1)

# Example 4: Survey response with age validation
class SurveyResponse(BaseModel):
    respondent_name: str = Field(..., min_length=1, max_length=100)
    # REQUIRED: Age must have realistic constraints
    respondent_age: int = Field(..., ge=13, le=120)  # 13+ years old
    satisfaction_score: int = Field(..., ge=1, le=10)

# Example 5: Pagination parameters with validation
class PaginationParams(BaseModel):
    # REQUIRED: Page number must be positive
    page: int = Field(default=1, ge=1)
    # REQUIRED: Page size must have min and max
    page_size: int = Field(default=20, ge=1, le=100)

# VALIDATION NOTE: ALL numeric Pydantic fields MUST use Field() with:
# 1. ge (greater than or equal) for minimum values
# 2. le (less than or equal) for maximum values
# 3. gt/lt for exclusive bounds if needed
# 4. This prevents invalid data from entering the system

# SECONDARY PATTERN - Never trust client-supplied privilege/role fields
class SafeUserProfile(BaseModel):
    name: str
    email: EmailStr
    # NOTE: is_admin is NEVER accepted from client input - it defaults False and is only
    # ever set server-side (e.g. by an existing admin through a dedicated admin endpoint)
    is_admin: bool = Field(default=False, exclude=True)
# NOTE: Accepting is_admin (or similar privilege fields) directly from client input enables privilege escalation
''',
        'expected_issues': ['violates mandatory Field() constraint pattern for numeric fields', 'accepts a client-controlled is_admin field, enabling privilege escalation via mass assignment']
    },

    'pr_011_info_disclosure': {
        'title': 'Improve error messages',
        'diff': '''
@@ -20,7 +20,8 @@ def get_user(user_id: int):
     try:
         return db.get_user(user_id)
     except Exception as e:
-        raise HTTPException(500, "Internal error")
+        raise HTTPException(500, f"Database error: {str(e)}")
''',
                'task_context': '''
TASK-011: Improve Error Messages for Debugging

Description:
Enhance error messages to help debug production issues more effectively

Requirements:
- Include more detailed error information in responses
- Help identify root cause of failures
- Make debugging easier for support team

Security Requirements (CRITICAL):
- Do NOT expose sensitive information to end users
- Log detailed errors internally only
- Return generic messages to clients
- Follow existing error handling patterns

References:
- See how database errors are handled in other endpoints
- See how API errors are logged vs returned
- Follow the logger.error() + generic HTTPException pattern

Acceptance Criteria:
✓ Detailed errors logged for internal debugging
✓ Generic error messages returned to users
✓ No sensitive data exposed in responses
✓ Matches logging pattern from other error handlers

Additional Requirement:
- Apply rate limiting to this endpoint to prevent user enumeration, matching the pattern used on other public endpoints
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always log detailed errors but return generic messages
# The codebase consistently uses logger.error() + generic HTTPException for ALL error handling:

import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Example 1: Database error handling
@app.get("/api/products/{product_id}")
def get_product(product_id: int):
    try:
        return db.get_product(product_id)
    except DatabaseError as e:
        # REQUIRED: Log the full error internally
        logger.error(f"Database error in get_product: {e}")
        # REQUIRED: Return generic message to client
        raise HTTPException(500, "Internal error")  # DO NOT include error details

# Example 2: External API error handling
@app.get("/api/external/data")
def fetch_external_data():
    try:
        response = external_api.fetch()
        return response.json()
    except RequestException as e:
        # REQUIRED: Log details for debugging
        logger.error(f"External API failed: {e}")
        # REQUIRED: Generic message to prevent information leakage
        raise HTTPException(503, "Service temporarily unavailable")

# Example 3: File operation error handling
@app.get("/api/reports/{report_id}")
def get_report(report_id: int):
    try:
        report_path = db.get_report_path(report_id)
        with open(report_path, 'r') as f:
            return {"content": f.read()}
    except FileNotFoundError as e:
        logger.error(f"Report file not found: {e}")
        raise HTTPException(404, "Report not found")
    except Exception as e:
        # REQUIRED: Log full exception stack trace
        logger.exception(f"Error reading report {report_id}")
        # REQUIRED: Never expose file paths or system details
        raise HTTPException(500, "Failed to retrieve report")

# Example 4: Authentication error handling
@app.post("/api/auth/verify")
def verify_credentials(username: str, password: str):
    try:
        user = db.authenticate(username, password)
        if not user:
            raise HTTPException(401, "Invalid credentials")
        return {"token": create_token(user)}
    except Exception as e:
        # REQUIRED: Log for security monitoring
        logger.error(f"Auth error for user {username}: {e}")
        # REQUIRED: Generic message (don't reveal if user exists)
        raise HTTPException(401, "Invalid credentials")

# Example 5: Payment processing error handling
@app.post("/api/payments/charge")
def process_payment(payment_info: PaymentInfo):
    try:
        result = payment_gateway.charge(payment_info)
        return {"transaction_id": result.id}
    except PaymentError as e:
        # REQUIRED: Log full payment error details
        logger.error(f"Payment failed: {e}, Gateway response: {e.gateway_response}")
        # REQUIRED: User-friendly message without internal details
        raise HTTPException(400, "Payment processing failed")

# SECURITY NOTE: ALL exception handlers MUST:
# 1. Use logger.error() or logger.exception() to log full details
# 2. Raise HTTPException with GENERIC messages only
# 3. NEVER include: stack traces, SQL queries, file paths, config values
# 4. This prevents information disclosure to attackers

# SECONDARY PATTERN - Rate limit endpoints that can be used for enumeration
@app.get("/api/users/{user_id}/safe")
@limiter.limit("30/minute")
def get_user_safe(user_id: int):
    return db.get_user(user_id)
# NOTE: Endpoints that reveal per-ID information should be rate limited to slow enumeration
''',
        'expected_issues': ['violates logger.error + generic message pattern', 'missing rate limiting on the endpoint, making user enumeration easier']
    },

    'pr_012_path_traversal': {
        'title': 'Add file download endpoint',
        'diff': '''
@@ -10,3 +10,9 @@ def upload_file(file: UploadFile):
     save_file(file)
     return {"status": "uploaded"}
+
+
+@app.get("/api/files/{filename}")
+def download_file(filename: str):
+    file_path = f"./uploads/{filename}"
+    return FileResponse(file_path)
''',
                'task_context': '''
TASK-012: Add File Download Endpoint

Description:
Implement GET /api/files/{filename} to allow downloading uploaded files

Requirements:
- Accept filename as URL parameter
- Retrieve file from uploads directory
- Return file for download

Security Requirements (CRITICAL):
- Prevent path traversal attacks
- Validate file paths to prevent access to system files
- Follow existing file access security patterns

References:
- See how user uploads are accessed
- See how reports are downloaded
- Follow the os.path.abspath + startswith validation pattern

Acceptance Criteria:
✓ File download functionality works
✓ Path validation prevents traversal attacks
✓ Only files in uploads directory can be accessed
✓ Matches path validation pattern from other file access endpoints

Additional Requirement:
- Apply rate limiting to the file download endpoint, matching the pattern used on other public-facing endpoints
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always validate file paths with os.path.abspath + startswith check
# The codebase consistently uses this pattern for ALL file access operations:

import os
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

# Base directory for file operations
UPLOAD_DIR = os.path.abspath("/var/app/uploads")
REPORTS_DIR = os.path.abspath("/var/app/reports")
DOCUMENTS_DIR = os.path.abspath("/var/app/documents")

# Example 1: Download user uploads with path validation
@app.get("/api/uploads/{filename}")
def download_upload(filename: str):
    # REQUIRED: Validate path to prevent traversal
    safe_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))
    if not safe_path.startswith(UPLOAD_DIR):
        raise HTTPException(403, "Invalid path")
    if not os.path.exists(safe_path):
        raise HTTPException(404, "File not found")
    return FileResponse(safe_path)

# Example 2: Download reports with path validation
@app.get("/api/reports/{report_name}")
def download_report(report_name: str):
    # REQUIRED: Always use abspath + startswith validation
    safe_path = os.path.abspath(os.path.join(REPORTS_DIR, report_name))
    if not safe_path.startswith(REPORTS_DIR):
        raise HTTPException(403, "Invalid path")
    if not os.path.isfile(safe_path):
        raise HTTPException(404, "Report not found")
    return FileResponse(safe_path)

# Example 3: Read document with path validation
@app.get("/api/documents/{doc_id}/content")
def get_document_content(doc_id: int):
    doc_filename = db.get_document_filename(doc_id)
    # REQUIRED: Validate even database-provided filenames
    safe_path = os.path.abspath(os.path.join(DOCUMENTS_DIR, doc_filename))
    if not safe_path.startswith(DOCUMENTS_DIR):
        logger.warning(f"Path traversal attempt: {doc_filename}")
        raise HTTPException(403, "Invalid document path")
    with open(safe_path, 'r') as f:
        return {"content": f.read()}

# Example 4: User avatar access with path validation
@app.get("/api/users/{user_id}/avatar")
def get_user_avatar(user_id: int):
    avatar_filename = db.get_user_avatar(user_id)
    AVATAR_DIR = os.path.abspath("/var/app/avatars")
    # REQUIRED: Three-step validation pattern
    safe_path = os.path.abspath(os.path.join(AVATAR_DIR, avatar_filename))
    if not safe_path.startswith(AVATAR_DIR):
        raise HTTPException(403, "Invalid path")
    if not os.path.exists(safe_path):
        raise HTTPException(404, "Avatar not found")
    return FileResponse(safe_path)

# Example 5: Export file with path validation
@app.get("/api/exports/{export_id}")
def download_export(export_id: str):
    EXPORT_DIR = os.path.abspath("/var/app/exports")
    export_file = f"export_{export_id}.csv"
    # REQUIRED: Validate even if filename is constructed internally
    safe_path = os.path.abspath(os.path.join(EXPORT_DIR, export_file))
    if not safe_path.startswith(EXPORT_DIR):
        raise HTTPException(403, "Invalid path")
    return FileResponse(safe_path)

# SECURITY NOTE: ALL file access endpoints MUST use this pattern:
# 1. Define absolute base directory: BASE_DIR = os.path.abspath(...)
# 2. Build path: safe_path = os.path.abspath(os.path.join(BASE_DIR, filename))
# 3. Validate: if not safe_path.startswith(BASE_DIR): raise 403
# 4. This prevents path traversal attacks like "../../../etc/passwd"

# SECONDARY PATTERN - Rate limit file download endpoints
@app.get("/api/uploads/safe/{filename}")
@limiter.limit("20/minute")
def download_upload_safe(filename: str):
    safe_path = os.path.abspath(os.path.join(UPLOAD_DIR, filename))
    return FileResponse(safe_path)
# NOTE: File download endpoints should be rate limited to prevent scraping/abuse
''',
        'expected_issues': ['violates mandatory os.path.abspath + startswith validation pattern', 'missing rate limiting on the file download endpoint']
    },

    'pr_013_cors_misconfiguration': {
        'title': 'Enable CORS for API',
        'diff': '''
@@ -1,8 +1,20 @@
 from fastapi import FastAPI
+from fastapi.middleware.cors import CORSMiddleware

 app = FastAPI()

+app.add_middleware(
+    CORSMiddleware,
+    allow_origins=["*"],
+    allow_credentials=True,
+    allow_methods=["*"],
+)

 @app.get("/api/data")
 def get_data():
+    return db.get_public_data()
+
+
+@app.get("/api/data/{data_id}")
+def get_data_by_id(data_id: int):
+    return db.get_data(data_id)
''',
                'task_context': '''
TASK-013: Enable CORS for Frontend Integration

Description:
Configure CORS middleware to allow frontend application to access the API

Requirements:
- Add CORS middleware to FastAPI application
- Allow credentials for authenticated requests
- Support all HTTP methods needed by frontend

Security Requirements (CRITICAL):
- Specify explicit list of allowed origins
- Do NOT use wildcard origins with credentials
- Follow existing CORS configuration patterns

References:
- See production API CORS configuration
- Review security policy on allowed origins
- Use environment-based origin whitelist

Acceptance Criteria:
✓ CORS middleware configured
✓ Explicit origin whitelist defined
✓ Credentials allowed only for whitelisted domains
✓ Matches explicit allow_origins pattern from production config

Additional Requirement:
- Wrap the new lookup endpoint's database call in try-except with logging, matching the error handling pattern used elsewhere
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always use explicit allow_origins list, NEVER ["*"]
# The codebase consistently uses explicit domain whitelists for ALL CORS configurations:

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Example 1: Production API with explicit origins
app.add_middleware(
    CORSMiddleware,
    # REQUIRED: Explicit list of allowed domains
    allow_origins=[
        "https://app.company.com",
        "https://dashboard.company.com",
        "https://mobile.company.com"
    ],
    allow_credentials=True,  # OK with explicit origins
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Example 2: Multi-environment CORS with explicit origins
import os

# REQUIRED: Load allowed origins from config, never use "*"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
# Default for local dev: ["http://localhost:3000", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Explicit list from environment
    allow_credentials=True,
    allow_methods=["GET", "POST"],
)

# Example 3: Admin API with strict CORS
admin_origins = [
    "https://admin.company.com",  # Only admin panel
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=admin_origins,  # REQUIRED: Minimal whitelist
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
)

# Example 4: Public API with controlled CORS
public_api_origins = [
    "https://partner1.com",
    "https://partner2.com",
    "https://company.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=public_api_origins,  # REQUIRED: Known partners only
    allow_credentials=False,  # No credentials for public API
    allow_methods=["GET"],  # Read-only for external access
)

# SECURITY NOTE: CORS configuration MUST follow these rules:
# 1. NEVER use allow_origins=["*"] - always explicit domain list
# 2. If allow_credentials=True, origins MUST be explicit (not "*")
# 3. Load allowed origins from environment variables or config
# 4. Using ["*"] with credentials enables cross-origin attacks
# 5. Be specific with allow_methods (avoid ["*"] if possible)

# SECONDARY PATTERN - Wrap database lookups in try-except with logging
@app.get("/api/data/safe/{data_id}")
def get_data_by_id_safe(data_id: int):
    try:
        return db.get_data(data_id)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
# NOTE: Database lookups should be wrapped in try-except with logging
''',
        'expected_issues': ['violates explicit allow_origins whitelist pattern', "new /api/data/{data_id} endpoint has no try-except around its database call"]
    },

    'pr_014_timing_attack': {
        'title': 'Add API key validation',
        'diff': '''
@@ -10,3 +10,10 @@ def protected_endpoint(api_key: str = Header()):
+
+
+def validate_api_key(key: str) -> bool:
+    valid_key = os.getenv("API_KEY")
+    return key == valid_key
''',
                'task_context': '''
TASK-014: Add API Key Validation

Description:
Implement validate_api_key function to check API keys for protected endpoints

Requirements:
- Accept API key as parameter
- Compare against valid API key from environment
- Return boolean indicating if key is valid

Security Requirements (CRITICAL):
- Prevent timing attacks on key comparison
- Use constant-time comparison for secrets
- Follow existing secret comparison patterns

References:
- See how passwords are verified
- See how session tokens are validated
- Follow the hmac.compare_digest() pattern for all secret comparisons

Acceptance Criteria:
✓ API key validation function implemented
✓ Constant-time comparison used
✓ Timing attack vulnerability prevented
✓ Matches hmac.compare_digest pattern from password/token verification

Additional Requirement:
- Validate the key format/length before comparison, following the input validation pattern used elsewhere
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always use hmac.compare_digest for secret comparisons
# The codebase consistently uses hmac.compare_digest() for ALL credential/token comparisons:

import hmac
import os
from fastapi import Header, HTTPException

# Example 1: Password verification with constant-time comparison
def verify_password(username: str, password: str) -> bool:
    stored_hash = db.get_password_hash(username)
    input_hash = hash_password(password)
    # REQUIRED: Use hmac.compare_digest for password comparison
    return hmac.compare_digest(input_hash, stored_hash)  # NOT ==

# Example 2: API key validation with constant-time comparison
def validate_api_key(key: str) -> bool:
    valid_key = os.getenv("API_KEY")
    if not valid_key:
        return False
    # REQUIRED: Use hmac.compare_digest for API key comparison
    return hmac.compare_digest(key, valid_key)  # NOT key == valid_key

# Example 3: Session token verification
@app.get("/api/protected")
def protected_endpoint(session_token: str = Header()):
    stored_token = db.get_session_token(session_token)
    if not stored_token:
        raise HTTPException(401, "Invalid session")
    # REQUIRED: Constant-time comparison for tokens
    if not hmac.compare_digest(session_token, stored_token):
        raise HTTPException(401, "Invalid session")
    return {"data": "protected"}

# Example 4: HMAC signature verification
def verify_webhook_signature(payload: str, signature: str) -> bool:
    secret = os.getenv("WEBHOOK_SECRET")
    expected_sig = hmac.new(secret.encode(), payload.encode(), 'sha256').hexdigest()
    # REQUIRED: Compare signatures with constant-time comparison
    return hmac.compare_digest(signature, expected_sig)

# Example 5: Reset token validation
def validate_reset_token(email: str, token: str) -> bool:
    stored_token = db.get_reset_token(email)
    if not stored_token:
        return False
    # REQUIRED: Use hmac.compare_digest for token validation
    return hmac.compare_digest(token, stored_token)

# Example 6: Two-factor authentication code verification
def verify_2fa_code(user_id: int, code: str) -> bool:
    expected_code = db.get_2fa_code(user_id)
    if not expected_code:
        return False
    # REQUIRED: Constant-time comparison for 2FA codes
    return hmac.compare_digest(code, expected_code)

# SECURITY NOTE: ALL secret/credential comparisons MUST use hmac.compare_digest():
# 1. Password hashes, API keys, session tokens, reset tokens
# 2. HMAC signatures, 2FA codes, CSRF tokens
# ❌ if key == valid_key - VULNERABLE to timing attacks
# ❌ Comparison stops at first different byte, revealing information
# ✅ hmac.compare_digest(key, valid_key) - SAFE constant-time comparison
# ✅ Always takes same time regardless of where strings differ

# SECONDARY PATTERN - Validate input format/length before using it
def validate_api_key_safe(key: str) -> bool:
    if not key or len(key) != 64:
        return False
    valid_key = os.getenv("API_KEY")
    return hmac.compare_digest(key, valid_key)
# NOTE: Validate input shape before using it in security-sensitive comparisons
''',
        'expected_issues': ['violates mandatory hmac.compare_digest pattern for secrets', 'missing input validation on the key parameter (no length/format check) before comparison']
    },

    'pr_015_unvalidated_redirect': {
        'title': 'Add OAuth callback',
        'diff': '''
@@ -20,3 +20,11 @@ def oauth_login():
     return redirect_to_oauth()
+
+
+@app.get("/api/auth/callback")
+def oauth_callback(code: str, redirect_url: str):
+    token = exchange_code_for_token(code)
+    save_token(token)
+    return RedirectResponse(url=redirect_url)
''',
                'task_context': '''
TASK-015: Add OAuth Callback Handler

Description:
Implement OAuth callback endpoint to handle redirects after authentication

Requirements:
- Accept OAuth code and redirect_url parameters
- Exchange code for access token
- Save token for user
- Redirect user to specified URL

Security Requirements (CRITICAL):
- Validate redirect URLs to prevent open redirect attacks
- Only allow redirects to trusted domains
- Follow existing redirect validation patterns

References:
- See how login endpoint handles post-auth redirects
- See how SAML callback validates relay_state
- Follow the ALLOWED_REDIRECTS whitelist pattern

Acceptance Criteria:
✓ OAuth callback functionality works
✓ Redirect URL validation implemented
✓ Only whitelisted domains allowed
✓ Matches ALLOWED_REDIRECTS pattern from other redirect endpoints

Additional Requirement:
- Validate the format of the OAuth code parameter before using it, following the input validation pattern used elsewhere
''',
'context': '''
# CRITICAL SECURITY PATTERN - Always validate redirects against ALLOWED_REDIRECTS whitelist
# The codebase consistently validates ALL redirect URLs against explicit whitelists:

from fastapi import HTTPException
from fastapi.responses import RedirectResponse

# Define allowed redirect destinations
ALLOWED_REDIRECTS = [
    "https://app.company.com",
    "https://www.company.com",
    "https://dashboard.company.com",
]

# Example 1: OAuth callback with redirect validation
@app.get("/api/oauth/callback")
def oauth_callback(code: str, state: str, redirect_url: str):
    # Exchange code for token
    token = exchange_oauth_code(code, state)
    save_user_token(token)

    # REQUIRED: Validate redirect URL against whitelist
    if redirect_url not in ALLOWED_REDIRECTS:
        raise HTTPException(400, "Invalid redirect URL")

    return RedirectResponse(url=redirect_url)

# Example 2: Post-login redirect with validation
@app.post("/api/auth/login")
def login(username: str, password: str, next_url: str = None):
    user = authenticate(username, password)
    if not user:
        raise HTTPException(401, "Invalid credentials")

    create_session(user)

    # REQUIRED: Validate next_url if provided
    if next_url:
        if next_url not in ALLOWED_REDIRECTS:
            raise HTTPException(400, "Invalid redirect")
        return RedirectResponse(url=next_url)

    # Default redirect if none provided
    return RedirectResponse(url="https://app.company.com/dashboard")

# Example 3: SAML authentication redirect with validation
@app.post("/api/saml/acs")
def saml_acs(saml_response: str, relay_state: str):
    user_info = validate_saml_response(saml_response)
    create_session(user_info)

    # REQUIRED: Validate relay_state (redirect destination)
    if relay_state not in ALLOWED_REDIRECTS:
        logger.warning(f"Invalid SAML relay_state: {relay_state}")
        raise HTTPException(400, "Invalid redirect destination")

    return RedirectResponse(url=relay_state)

# Example 4: Account verification redirect with validation
@app.get("/api/verify-email")
def verify_email(token: str, return_to: str = None):
    user_id = validate_verification_token(token)
    db.mark_email_verified(user_id)

    redirect_target = return_to or "https://app.company.com"

    # REQUIRED: Validate even default redirects
    if redirect_target not in ALLOWED_REDIRECTS:
        raise HTTPException(400, "Invalid return URL")

    return RedirectResponse(url=redirect_target)

# Example 5: Logout redirect with validation
@app.post("/api/auth/logout")
def logout(redirect_to: str = None):
    destroy_session()

    final_redirect = redirect_to or "https://www.company.com"

    # REQUIRED: Validate logout redirect
    if final_redirect not in ALLOWED_REDIRECTS:
        raise HTTPException(400, "Invalid logout destination")

    return RedirectResponse(url=final_redirect)

# SECURITY NOTE: ALL redirect operations MUST:
# 1. Define ALLOWED_REDIRECTS whitelist with explicit URLs
# 2. Validate redirect URL: if url not in ALLOWED_REDIRECTS: raise 400
# 3. NEVER trust user-provided redirect URLs without validation
# 4. Open redirects can be used for phishing attacks
# 5. Even internal redirects should be validated

# SECONDARY PATTERN - Validate input format before using it
@app.get("/api/auth/callback/safe")
def oauth_callback_safe(code: str, redirect_url: str):
    if not code or len(code) > 256:
        raise HTTPException(status_code=400, detail="Invalid code")
    token = exchange_oauth_code(code)
    return RedirectResponse(url=redirect_url)
# NOTE: Validate the shape of external input (OAuth codes, tokens) before using it
''',
        'expected_issues': ['violates mandatory ALLOWED_REDIRECTS whitelist validation', 'missing input validation on the OAuth code parameter']
    },
}


def load_pr(pr_id: str) -> dict:
    """Load a single PR by ID."""
    if pr_id not in SAMPLE_PRS:
        raise ValueError(f"PR {pr_id} not found")

    pr = SAMPLE_PRS[pr_id].copy()
    pr['id'] = pr_id
    return pr


def load_all_prs() -> list:
    """Load all sample PRs."""
    return [load_pr(pr_id) for pr_id in SAMPLE_PRS.keys()]
