"""
Toy FastAPI repository for demonstrating context retrieval.
This represents the existing codebase that PRs are modifying.

EXPANDED VERSION: Includes noise, unrelated modules, and deprecated code
to demonstrate the power of selective context retrieval.
"""

TOY_REPOSITORY = {
    'app/api/users.py': '''
from fastapi import FastAPI, HTTPException, Depends
from app.auth import get_current_user
from app.models import User, UserProfile
from app import db

app = FastAPI()


@app.get("/api/users")
def list_users():
    """List all users."""
    return db.get_all_users()


@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    """Get a single user by ID."""
    return db.get_user(user_id)


@app.get("/api/users/by-email")
def find_user_by_email(email: str):
    """Find user by email using safe parameterized query."""
    user = db.query("SELECT * FROM users WHERE email = ?", (email,))
    return user


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, current_user: User = Depends(get_current_user)):
    """Delete user - requires authentication and authorization."""
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Unauthorized")
    db.delete_user(user_id)
    return {"status": "deleted"}


@app.post("/api/users/{user_id}/email")
def update_email(user_id: int, new_email: str, current_user: User = Depends(get_current_user)):
    """Update user email - requires authentication."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot modify other users")
    db.update_user_email(user_id, new_email)
    return {"status": "updated"}


@app.post("/api/users/{user_id}/password")
def change_password(user_id: int, new_password: str, current_user: User = Depends(get_current_user)):
    """Change user password - requires authentication."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    db.update_password(user_id, hash_password(new_password))
    return {"status": "password_changed"}
''',

    'app/api/products.py': '''
from fastapi import FastAPI
from app import db

app = FastAPI()


@app.get("/api/products")
def list_products():
    """List all products."""
    return db.get_all_products()


@app.get("/api/products/search")
def search_products(name: str, category: str):
    """Search products using safe parameterized query."""
    results = db.execute(
        "SELECT * FROM products WHERE name LIKE ? AND category = ?",
        (f"%{name}%", category)
    )
    return results


@app.get("/api/products/{product_id}")
def get_product(product_id: int):
    """Get single product."""
    return db.get_product(product_id)
''',

    'app/api/posts.py': '''
from fastapi import FastAPI, Depends, HTTPException
from app.auth import get_current_user, limiter
from app.models import User, Post
from app import db
import html
import logging

app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/api/posts")
def list_posts():
    """List all posts."""
    return db.get_all_posts()


@app.get("/api/posts/by-tag")
def get_posts_by_tag(tag: str):
    """Get posts by tag using safe ORM query."""
    posts = db.query_posts(tag=tag)
    return posts


@app.post("/api/posts")
def create_post(content: str, current_user: User = Depends(get_current_user)):
    """Create a post with escaped HTML content."""
    safe_content = html.escape(content)
    post = db.create_post(user_id=current_user.id, content=safe_content)
    return post


@app.get("/api/posts/{post_id}")
def get_post(post_id: int):
    """Get a single post."""
    return db.get_post(post_id)
''',

    'app/api/auth.py': '''
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from app.auth import limiter, hash_password, verify_password
from app import db
import secrets
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Only these destinations may be used for post-auth redirects
ALLOWED_REDIRECTS = [
    "https://app.company.com",
    "https://app.company.com/dashboard",
    "https://www.company.com",
]


@app.post("/api/auth/login")
@limiter.limit("5/minute")
def login(request: Request, username: str, password: str):
    """Login endpoint with rate limiting. Never logs the raw password."""
    user = db.get_user_by_username(username)
    if not user or not verify_password(password, user.password_hash):
        logger.warning(f"Failed login attempt for username={username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_urlsafe(32)
    db.save_session(user.id, token)
    return {"token": token}


@app.post("/api/auth/register")
def register(username: str, password: str, email: str):
    """Register new user with hashed password."""
    password_hash = hash_password(password)
    user = db.create_user(username=username, password_hash=password_hash, email=email)
    return {"user_id": user.id}


@app.get("/api/oauth/callback")
@limiter.limit("10/minute")
def oauth_callback(request: Request, code: str, redirect_url: str = "https://app.company.com"):
    """OAuth callback with rate limiting and a validated redirect whitelist."""
    token = db.exchange_oauth_code(code)
    db.save_session(token.user_id, token.value)

    # REQUIRED: redirect target must be in the explicit whitelist
    if redirect_url not in ALLOWED_REDIRECTS:
        raise HTTPException(status_code=400, detail="Invalid redirect URL")

    return RedirectResponse(url=redirect_url)
''',

    'app/api/files.py': '''
from fastapi import FastAPI, HTTPException
from app import db
import os
import logging

app = FastAPI()
logger = logging.getLogger(__name__)


@app.get("/api/files/{file_id}")
def get_file(file_id: int):
    """Get file with proper error handling."""
    try:
        file_path = db.get_file_path(file_id)
        with open(file_path, 'r') as f:
            content = f.read()
        return {"content": content}
    except FileNotFoundError:
        logger.error(f"File not found: {file_id}")
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error reading file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")


@app.get("/api/downloads/{filename}")
def download_file(filename: str):
    """Download file with path validation."""
    base_dir = "/var/uploads"
    file_path = os.path.join(base_dir, filename)
    safe_path = os.path.abspath(file_path)

    if not safe_path.startswith(base_dir):
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        with open(safe_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
''',

    'app/api/inventory.py': '''
from fastapi import FastAPI, HTTPException
from app import db
from app.models import InventoryItem

app = FastAPI()


@app.post("/api/inventory/{item_id}/reserve")
def reserve_inventory(item_id: int, quantity: int):
    """Reserve inventory with proper locking."""
    with db.transaction():
        # Use FOR UPDATE to prevent race conditions
        item = db.query(
            "SELECT * FROM inventory WHERE id = ? FOR UPDATE",
            (item_id,)
        )

        if not item or item.quantity < quantity:
            raise HTTPException(status_code=400, detail="Insufficient inventory")

        db.execute(
            "UPDATE inventory SET quantity = quantity - ? WHERE id = ?",
            (quantity, item_id)
        )

        return {"status": "reserved", "remaining": item.quantity - quantity}
''',

    'app/auth.py': '''
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
import bcrypt

security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)


def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current authenticated user from token."""
    from app import db
    token = credentials.credentials
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())
''',

    'app/models.py': '''
"""Pydantic data models. All numeric fields use Field() range constraints."""
from pydantic import BaseModel, Field, EmailStr


class User(BaseModel):
    id: int
    username: str = Field(..., min_length=1, max_length=50)
    is_admin: bool = False  # REQUIRED: never accept this from client input directly


class UserProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    bio: str = Field(default="", max_length=500)
    # REQUIRED: every numeric field must have Field(ge=, le=) range constraints
    height_cm: int = Field(..., ge=50, le=300)
    weight_kg: float = Field(..., ge=20.0, le=500.0)


class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., ge=0.01)
    stock: int = Field(..., ge=0)
    rating: float = Field(..., ge=0.0, le=5.0)


class Discount(BaseModel):
    code: str = Field(..., min_length=3, max_length=20)
    percentage: int = Field(..., ge=0, le=100)
    max_uses: int = Field(..., ge=1)


class Invoice(BaseModel):
    id: int
    user_id: int
    amount: float = Field(..., ge=0.0)


class Payment(BaseModel):
    id: int
    invoice_id: int
    amount: float = Field(..., ge=0.0)


class Post(BaseModel):
    id: int
    user_id: int
    content: str = Field(..., max_length=5000)


class Notification(BaseModel):
    id: int
    user_id: int
    message: str = Field(..., max_length=1000)


class InventoryItem(BaseModel):
    id: int
    quantity: int = Field(..., ge=0)
''',

    'app/config.py': '''
"""Application configuration. All secrets are loaded from environment variables."""
import os

# REQUIRED: never hardcode credentials - always load from the environment
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET_KEY")
API_KEY = os.getenv("API_KEY")
''',

    'app/main.py': '''
"""Application entrypoint - CORS is always an explicit domain whitelist."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# REQUIRED: explicit allow_origins list loaded from config, NEVER "*"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://app.company.com").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
''',

    # NOISE: Billing module (unrelated to security issues)
    'app/api/billing.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User, Invoice, Payment
from app import db
import stripe
from decimal import Decimal

app = FastAPI()


@app.get("/api/billing/invoices")
def list_invoices(current_user: User = Depends(get_current_user)):
    """List user invoices."""
    invoices = db.query("SELECT * FROM invoices WHERE user_id = ?", (current_user.id,))
    return invoices


@app.post("/api/billing/charge")
def create_charge(amount: Decimal, token: str, current_user: User = Depends(get_current_user)):
    """Create a Stripe charge."""
    charge = stripe.Charge.create(
        amount=int(amount * 100),
        currency="usd",
        source=token,
        description=f"Charge for user {current_user.id}"
    )
    db.execute(
        "INSERT INTO payments (user_id, amount, stripe_id) VALUES (?, ?, ?)",
        (current_user.id, amount, charge.id)
    )
    return {"status": "success", "charge_id": charge.id}


@app.get("/api/billing/subscription")
def get_subscription(current_user: User = Depends(get_current_user)):
    """Get user subscription."""
    sub = db.query("SELECT * FROM subscriptions WHERE user_id = ?", (current_user.id,))
    return sub


@app.post("/api/billing/cancel")
def cancel_subscription(current_user: User = Depends(get_current_user)):
    """Cancel subscription."""
    db.execute("UPDATE subscriptions SET status = ? WHERE user_id = ?", ("cancelled", current_user.id))
    return {"status": "cancelled"}


@app.get("/api/billing/usage")
def get_usage(current_user: User = Depends(get_current_user)):
    """Get usage metrics for billing."""
    usage = db.query("SELECT * FROM usage WHERE user_id = ?", (current_user.id,))
    return usage
''',

    # NOISE: Analytics module (unrelated)
    'app/api/analytics.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db
from datetime import datetime, timedelta

app = FastAPI()


@app.get("/api/analytics/dashboard")
def get_dashboard(current_user: User = Depends(get_current_user)):
    """Get analytics dashboard data."""
    if not current_user.is_admin:
        return {"error": "Admin only"}

    stats = {
        "total_users": db.count("users"),
        "active_users": db.count("users WHERE last_login > ?", (datetime.now() - timedelta(days=30),)),
        "total_posts": db.count("posts"),
        "total_revenue": db.sum("payments", "amount")
    }
    return stats


@app.get("/api/analytics/events")
def list_events(limit: int = 100):
    """List analytics events."""
    events = db.query("SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,))
    return events


@app.post("/api/analytics/track")
def track_event(event_name: str, properties: dict, current_user: User = Depends(get_current_user)):
    """Track analytics event."""
    db.execute(
        "INSERT INTO events (user_id, event_name, properties) VALUES (?, ?, ?)",
        (current_user.id, event_name, str(properties))
    )
    return {"status": "tracked"}


@app.get("/api/analytics/reports/{report_id}")
def get_report(report_id: int, current_user: User = Depends(get_current_user)):
    """Get analytics report."""
    if not current_user.is_admin:
        return {"error": "Admin only"}
    report = db.get_report(report_id)
    return report


@app.post("/api/analytics/export")
def export_data(format: str, current_user: User = Depends(get_current_user)):
    """Export analytics data."""
    if not current_user.is_admin:
        return {"error": "Admin only"}
    data = db.get_all_analytics()
    if format == "csv":
        return export_to_csv(data)
    return {"error": "Unsupported format"}
''',

    # NOISE: Notifications module
    'app/api/notifications.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User, Notification
from app import db
import smtplib

app = FastAPI()


@app.get("/api/notifications")
def list_notifications(current_user: User = Depends(get_current_user)):
    """List user notifications."""
    notifications = db.query(
        "SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC",
        (current_user.id,)
    )
    return notifications


@app.post("/api/notifications/{notification_id}/read")
def mark_read(notification_id: int, current_user: User = Depends(get_current_user)):
    """Mark notification as read."""
    db.execute(
        "UPDATE notifications SET read = ? WHERE id = ? AND user_id = ?",
        (True, notification_id, current_user.id)
    )
    return {"status": "marked_read"}


@app.post("/api/notifications/send-email")
def send_email_notification(user_id: int, subject: str, body: str):
    """Send email notification (internal use)."""
    user = db.get_user(user_id)
    smtp = smtplib.SMTP("localhost")
    smtp.sendmail("noreply@example.com", user.email, f"Subject: {subject}\n\n{body}")
    smtp.quit()
    return {"status": "sent"}


@app.delete("/api/notifications/{notification_id}")
def delete_notification(notification_id: int, current_user: User = Depends(get_current_user)):
    """Delete notification."""
    db.execute(
        "DELETE FROM notifications WHERE id = ? AND user_id = ?",
        (notification_id, current_user.id)
    )
    return {"status": "deleted"}
''',

    # NOISE: Admin tools
    'app/api/admin.py': '''
from fastapi import FastAPI, Depends, HTTPException
from app.auth import get_current_user
from app.models import User
from app import db

app = FastAPI()


@app.get("/api/admin/users")
def list_all_users(current_user: User = Depends(get_current_user)):
    """List all users (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    users = db.query("SELECT * FROM users")
    return users


@app.post("/api/admin/users/{user_id}/ban")
def ban_user(user_id: int, current_user: User = Depends(get_current_user)):
    """Ban a user (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    db.execute("UPDATE users SET banned = ? WHERE id = ?", (True, user_id))
    return {"status": "banned"}


@app.post("/api/admin/users/{user_id}/unban")
def unban_user(user_id: int, current_user: User = Depends(get_current_user)):
    """Unban a user (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    db.execute("UPDATE users SET banned = ? WHERE id = ?", (False, user_id))
    return {"status": "unbanned"}


@app.get("/api/admin/logs")
def get_logs(limit: int = 100, current_user: User = Depends(get_current_user)):
    """Get system logs (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    logs = db.query("SELECT * FROM logs ORDER BY created_at DESC LIMIT ?", (limit,))
    return logs


@app.post("/api/admin/cleanup")
def cleanup_old_data(current_user: User = Depends(get_current_user)):
    """Cleanup old data (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    db.execute("DELETE FROM events WHERE created_at < DATE('now', '-90 days')")
    db.execute("DELETE FROM logs WHERE created_at < DATE('now', '-30 days')")
    return {"status": "cleaned"}
''',

    # NOISE: Reporting module with RED HERRING patterns
    'app/api/reports.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db

app = FastAPI()


@app.get("/api/reports/search")
def search_reports(query: str, current_user: User = Depends(get_current_user)):
    """Search reports - DEPRECATED: Use /api/reports/filter instead."""
    # RED HERRING: This looks like it might have SQL injection but it's just a bad pattern
    # This is NOT the same pattern as the security issues we're looking for
    results = db.search_reports(query)  # Uses ORM, not raw SQL
    return results


@app.post("/api/reports/generate")
def generate_report(report_type: str, filters: dict, current_user: User = Depends(get_current_user)):
    """Generate a report."""
    if not current_user.is_admin:
        return {"error": "Admin only"}

    # Complex report generation logic
    data = db.get_data_for_report(report_type, filters)
    report = compile_report(data, report_type)
    return report


@app.get("/api/reports/{report_id}/download")
def download_report(report_id: int, current_user: User = Depends(get_current_user)):
    """Download report."""
    report = db.get_report(report_id)
    if report.user_id != current_user.id and not current_user.is_admin:
        return {"error": "Not authorized"}
    return report.data


@app.delete("/api/reports/{report_id}")
def delete_report(report_id: int, current_user: User = Depends(get_current_user)):
    """Delete report."""
    if not current_user.is_admin:
        return {"error": "Admin only"}
    db.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    return {"status": "deleted"}


def compile_report(data, report_type):
    """Helper to compile report data."""
    # ... complex logic ...
    return {"type": report_type, "data": data}
''',

    # NOISE: Utilities with similar-sounding functions
    'app/utils/helpers.py': '''
"""Utility helper functions."""
from datetime import datetime
import hashlib
import json


def hash_string(s: str) -> str:
    """Hash a string using SHA256."""
    return hashlib.sha256(s.encode()).hexdigest()


def format_date(dt: datetime) -> str:
    """Format datetime as ISO string."""
    return dt.isoformat()


def parse_json_safe(s: str) -> dict:
    """Safely parse JSON string."""
    try:
        return json.loads(s)
    except:
        return {}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\\\", "_")
    # Remove other unsafe characters
    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    return filename


def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email and "." in email.split("@")[1]


def truncate_string(s: str, length: int = 100) -> str:
    """Truncate string to max length."""
    return s[:length] + "..." if len(s) > length else s


def generate_slug(title: str) -> str:
    """Generate URL slug from title."""
    slug = title.lower().replace(" ", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    return slug
''',

    # NOISE: Database utilities
    'app/utils/db_helpers.py': '''
"""Database helper utilities."""
from typing import List, Any


def build_where_clause(filters: dict) -> str:
    """Build WHERE clause from filters - uses parameterized queries."""
    conditions = []
    for key in filters.keys():
        conditions.append(f"{key} = ?")
    return " AND ".join(conditions) if conditions else "1=1"


def paginate_query(query: str, page: int, per_page: int) -> str:
    """Add pagination to query."""
    offset = (page - 1) * per_page
    return f"{query} LIMIT {per_page} OFFSET {offset}"


def bulk_insert(table: str, records: List[dict], db):
    """Bulk insert records."""
    if not records:
        return

    columns = list(records[0].keys())
    placeholders = ", ".join(["?"] * len(columns))

    for record in records:
        values = [record[col] for col in columns]
        db.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(values)
        )


def count_rows(table: str, where: str, params: tuple, db) -> int:
    """Count rows matching condition."""
    result = db.query(f"SELECT COUNT(*) FROM {table} WHERE {where}", params)
    return result[0][0] if result else 0
''',

    # NOISE: Cache utilities
    'app/utils/cache.py': '''
"""Caching utilities."""
import time
from typing import Any, Optional


class SimpleCache:
    """Simple in-memory cache."""

    def __init__(self):
        self._cache = {}
        self._timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            return self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL."""
        self._cache[key] = value
        self._timestamps[key] = time.time() + ttl

    def delete(self, key: str):
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]

    def clear_expired(self):
        """Clear expired cache entries."""
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if ts < now]
        for key in expired:
            self.delete(key)


cache = SimpleCache()
''',

    # NOISE: Email templates module
    'app/api/email_templates.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db

app = FastAPI()


@app.get("/api/email-templates")
def list_templates(current_user: User = Depends(get_current_user)):
    """List email templates."""
    if not current_user.is_admin:
        return {"error": "Admin only"}
    templates = db.query("SELECT * FROM email_templates")
    return templates


@app.post("/api/email-templates")
def create_template(name: str, subject: str, body: str, current_user: User = Depends(get_current_user)):
    """Create email template."""
    if not current_user.is_admin:
        return {"error": "Admin only"}
    db.execute(
        "INSERT INTO email_templates (name, subject, body) VALUES (?, ?, ?)",
        (name, subject, body)
    )
    return {"status": "created"}


@app.get("/api/email-templates/{template_id}")
def get_template(template_id: int):
    """Get email template."""
    template = db.query("SELECT * FROM email_templates WHERE id = ?", (template_id,))
    return template


@app.put("/api/email-templates/{template_id}")
def update_template(template_id: int, subject: str, body: str, current_user: User = Depends(get_current_user)):
    """Update email template."""
    if not current_user.is_admin:
        return {"error": "Admin only"}
    db.execute(
        "UPDATE email_templates SET subject = ?, body = ? WHERE id = ?",
        (subject, body, template_id)
    )
    return {"status": "updated"}
''',

    # NOISE: Webhooks module
    'app/api/webhooks.py': '''
from fastapi import FastAPI, Request
from app import db
import hmac
import hashlib

app = FastAPI()


@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook."""
    payload = await request.body()
    sig = request.headers.get("Stripe-Signature")

    # Verify webhook signature
    secret = "whsec_test123"
    expected_sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(sig, expected_sig):
        return {"error": "Invalid signature"}

    # Process webhook
    event = await request.json()
    db.execute(
        "INSERT INTO webhook_events (source, type, payload) VALUES (?, ?, ?)",
        ("stripe", event["type"], str(event))
    )
    return {"status": "processed"}


@app.post("/api/webhooks/github")
async def github_webhook(request: Request):
    """Handle GitHub webhook."""
    event = await request.json()
    db.execute(
        "INSERT INTO webhook_events (source, type, payload) VALUES (?, ?, ?)",
        ("github", event.get("action"), str(event))
    )
    return {"status": "processed"}


@app.get("/api/webhooks/events")
def list_webhook_events(limit: int = 100):
    """List webhook events."""
    events = db.query("SELECT * FROM webhook_events ORDER BY created_at DESC LIMIT ?", (limit,))
    return events
''',

    # NOISE: Integrations module
    'app/api/integrations.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db
import requests

app = FastAPI()


@app.get("/api/integrations")
def list_integrations(current_user: User = Depends(get_current_user)):
    """List user integrations."""
    integrations = db.query("SELECT * FROM integrations WHERE user_id = ?", (current_user.id,))
    return integrations


@app.post("/api/integrations/slack")
def connect_slack(webhook_url: str, current_user: User = Depends(get_current_user)):
    """Connect Slack integration."""
    db.execute(
        "INSERT INTO integrations (user_id, service, config) VALUES (?, ?, ?)",
        (current_user.id, "slack", webhook_url)
    )
    return {"status": "connected"}


@app.post("/api/integrations/discord")
def connect_discord(webhook_url: str, current_user: User = Depends(get_current_user)):
    """Connect Discord integration."""
    db.execute(
        "INSERT INTO integrations (user_id, service, config) VALUES (?, ?, ?)",
        (current_user.id, "discord", webhook_url)
    )
    return {"status": "connected"}


@app.delete("/api/integrations/{integration_id}")
def disconnect_integration(integration_id: int, current_user: User = Depends(get_current_user)):
    """Disconnect integration."""
    db.execute(
        "DELETE FROM integrations WHERE id = ? AND user_id = ?",
        (integration_id, current_user.id)
    )
    return {"status": "disconnected"}


@app.post("/api/integrations/test")
def test_integration(integration_id: int, current_user: User = Depends(get_current_user)):
    """Test integration."""
    integration = db.query(
        "SELECT * FROM integrations WHERE id = ? AND user_id = ?",
        (integration_id, current_user.id)
    )
    if not integration:
        return {"error": "Not found"}

    # Send test message
    if integration["service"] == "slack":
        requests.post(integration["config"], json={"text": "Test message"})

    return {"status": "tested"}
''',

    # RED HERRING: Legacy/deprecated module with BAD patterns
    'app/legacy/old_api.py': '''
"""
DEPRECATED MODULE - DO NOT USE
This module contains old patterns that have been replaced.
Kept for backwards compatibility only.
"""
from fastapi import FastAPI
from app import db

app = FastAPI()


@app.get("/api/legacy/search")
def legacy_search(q: str):
    """DEPRECATED: Old search endpoint with bad practices."""
    # This looks bad but uses ORM internally, not actual SQL injection
    results = db.legacy_search(q)
    return results


@app.get("/api/legacy/users/{user_id}")
def legacy_get_user(user_id: int):
    """DEPRECATED: No authentication required (legacy behavior)."""
    user = db.get_user(user_id)
    return user


@app.post("/api/legacy/update")
def legacy_update(data: dict):
    """DEPRECATED: Accepts arbitrary data."""
    # This looks unsafe but is actually using ORM
    db.legacy_update(data)
    return {"status": "updated"}


@app.get("/api/legacy/export")
def legacy_export(format: str):
    """DEPRECATED: Export data in various formats."""
    data = db.get_all_legacy_data()
    return {"data": data, "format": format}
''',

    # NOISE: Settings/Config module
    'app/api/settings.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db

app = FastAPI()


@app.get("/api/settings")
def get_settings(current_user: User = Depends(get_current_user)):
    """Get user settings."""
    settings = db.query("SELECT * FROM settings WHERE user_id = ?", (current_user.id,))
    return settings


@app.put("/api/settings/theme")
def update_theme(theme: str, current_user: User = Depends(get_current_user)):
    """Update theme setting."""
    db.execute(
        "UPDATE settings SET theme = ? WHERE user_id = ?",
        (theme, current_user.id)
    )
    return {"status": "updated"}


@app.put("/api/settings/language")
def update_language(language: str, current_user: User = Depends(get_current_user)):
    """Update language setting."""
    db.execute(
        "UPDATE settings SET language = ? WHERE user_id = ?",
        (language, current_user.id)
    )
    return {"status": "updated"}


@app.put("/api/settings/notifications")
def update_notification_settings(email: bool, sms: bool, push: bool, current_user: User = Depends(get_current_user)):
    """Update notification settings."""
    db.execute(
        "UPDATE settings SET email_notifications = ?, sms_notifications = ?, push_notifications = ? WHERE user_id = ?",
        (email, sms, push, current_user.id)
    )
    return {"status": "updated"}


@app.put("/api/settings/privacy")
def update_privacy(public_profile: bool, current_user: User = Depends(get_current_user)):
    """Update privacy settings."""
    db.execute(
        "UPDATE settings SET public_profile = ? WHERE user_id = ?",
        (public_profile, current_user.id)
    )
    return {"status": "updated"}
''',

    # NOISE: Comments/Reviews module
    'app/api/comments.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db
import html

app = FastAPI()


@app.get("/api/posts/{post_id}/comments")
def list_comments(post_id: int):
    """List comments on a post."""
    comments = db.query("SELECT * FROM comments WHERE post_id = ?", (post_id,))
    return comments


@app.post("/api/posts/{post_id}/comments")
def create_comment(post_id: int, content: str, current_user: User = Depends(get_current_user)):
    """Create a comment."""
    safe_content = html.escape(content)
    db.execute(
        "INSERT INTO comments (post_id, user_id, content) VALUES (?, ?, ?)",
        (post_id, current_user.id, safe_content)
    )
    return {"status": "created"}


@app.put("/api/comments/{comment_id}")
def update_comment(comment_id: int, content: str, current_user: User = Depends(get_current_user)):
    """Update a comment."""
    # Check ownership
    comment = db.query("SELECT * FROM comments WHERE id = ?", (comment_id,))
    if comment["user_id"] != current_user.id:
        return {"error": "Not authorized"}

    safe_content = html.escape(content)
    db.execute(
        "UPDATE comments SET content = ? WHERE id = ?",
        (safe_content, comment_id)
    )
    return {"status": "updated"}


@app.delete("/api/comments/{comment_id}")
def delete_comment(comment_id: int, current_user: User = Depends(get_current_user)):
    """Delete a comment."""
    comment = db.query("SELECT * FROM comments WHERE id = ?", (comment_id,))
    if comment["user_id"] != current_user.id and not current_user.is_admin:
        return {"error": "Not authorized"}

    db.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    return {"status": "deleted"}


@app.post("/api/comments/{comment_id}/like")
def like_comment(comment_id: int, current_user: User = Depends(get_current_user)):
    """Like a comment."""
    db.execute(
        "INSERT INTO comment_likes (comment_id, user_id) VALUES (?, ?)",
        (comment_id, current_user.id)
    )
    return {"status": "liked"}
''',

    # NOISE: Tags module
    'app/api/tags.py': '''
from fastapi import FastAPI, Depends
from app.auth import get_current_user
from app.models import User
from app import db

app = FastAPI()


@app.get("/api/tags")
def list_tags():
    """List all tags."""
    tags = db.query("SELECT * FROM tags ORDER BY name")
    return tags


@app.get("/api/tags/popular")
def popular_tags(limit: int = 20):
    """Get popular tags."""
    tags = db.query(
        "SELECT t.*, COUNT(pt.post_id) as count FROM tags t LEFT JOIN post_tags pt ON t.id = pt.tag_id GROUP BY t.id ORDER BY count DESC LIMIT ?",
        (limit,)
    )
    return tags


@app.post("/api/tags")
def create_tag(name: str, current_user: User = Depends(get_current_user)):
    """Create a new tag."""
    if not current_user.is_admin:
        return {"error": "Admin only"}

    db.execute("INSERT INTO tags (name) VALUES (?)", (name,))
    return {"status": "created"}


@app.delete("/api/tags/{tag_id}")
def delete_tag(tag_id: int, current_user: User = Depends(get_current_user)):
    """Delete a tag."""
    if not current_user.is_admin:
        return {"error": "Admin only"}

    db.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
    return {"status": "deleted"}


@app.post("/api/posts/{post_id}/tags/{tag_id}")
def add_tag_to_post(post_id: int, tag_id: int, current_user: User = Depends(get_current_user)):
    """Add tag to post."""
    post = db.query("SELECT * FROM posts WHERE id = ?", (post_id,))
    if post["user_id"] != current_user.id:
        return {"error": "Not authorized"}

    db.execute(
        "INSERT INTO post_tags (post_id, tag_id) VALUES (?, ?)",
        (post_id, tag_id)
    )
    return {"status": "added"}
''',
}


def get_toy_repository():
    """Get the toy repository as a dict of file paths to source code."""
    return TOY_REPOSITORY
