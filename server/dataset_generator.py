"""
Dynamic Dataset Generator
=========================
Uses an LLM to procedurally generate fresh, never-repeated dirty datasets
on every environment reset. This is the core differentiator: the agent
cannot memorise solutions — it must genuinely learn to clean data.

Each call returns a brand-new dirty dataset with:
  - A random real-world domain (HR, finance, logistics, healthcare, etc.)
  - Randomised dirty patterns appropriate for the chosen difficulty
  - A fully computed ground truth used by the grader

Falls back to a static dataset pool if the LLM is unavailable.
"""

import os
import json
import random
import hashlib
import datetime
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI

# ── LLM client (optional — env falls back gracefully) ─────────────────────
_client: Optional[OpenAI] = None

def _get_client() -> Optional[OpenAI]:
    global _client
    if _client is None:
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        if api_key:
            _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def _llm_model() -> str:
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


# ── Domain catalogue ──────────────────────────────────────────────────────
DOMAINS = {
    "easy": [
        "customer contact list for a retail company",
        "employee directory for a tech startup",
        "supplier contact list for a manufacturing firm",
        "patient appointment records for a dental clinic",
        "vendor list for a restaurant chain",
    ],
    "medium": [
        "quarterly sales records for a SaaS company",
        "payroll data for a logistics company",
        "inventory records for an e-commerce warehouse",
        "student enrollment data for a university",
        "expense reports for a consulting firm",
    ],
    "hard": [
        "international shipment records for a freight company",
        "clinical trial data for a pharmaceutical firm",
        "multi-currency transaction log for a fintech startup",
        "IoT sensor readings for a smart building system",
        "cross-border e-commerce order history",
    ],
}

DIRTY_PATTERNS = {
    "easy": [
        "exact duplicate rows",
        "phone numbers in 4+ inconsistent formats",
        "email addresses with extra whitespace or uppercase",
        "names with inconsistent capitalisation",
    ],
    "medium": [
        "missing numeric values that should be median-imputed",
        "date columns mixing MM/DD/YYYY and YYYY-MM-DD formats",
        "boolean columns mixing True/False with 'yes'/'no'/'1'/'0'",
        "string columns with inconsistent category spellings",
        "numeric columns storing values as strings with units",
    ],
    "hard": [
        "statistical outliers (values 10x the normal range)",
        "mixed unit columns (some rows in metric, some in imperial)",
        "dates mixing 5 different international formats",
        "rows violating referential integrity constraints",
        "near-duplicate rows with minor typo differences",
        "currency columns mixing USD, EUR, GBP values without normalisation",
    ],
}


# ── Generator ─────────────────────────────────────────────────────────────

def generate_dynamic_task(difficulty: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a fresh dirty dataset task for the given difficulty.
    Returns a task dict compatible with the static TASKS format.
    Uses LLM if available, falls back to static pool otherwise.
    """
    if seed is None:
        seed = random.randint(0, 999999)

    client = _get_client()
    if client is not None:
        try:
            return _generate_via_llm(client, difficulty, seed)
        except Exception as e:
            print(f"[DynamicGen] LLM generation failed ({e}), falling back to static pool")

    return _generate_from_static_pool(difficulty, seed)


def _generate_via_llm(client: OpenAI, difficulty: str, seed: int) -> Dict[str, Any]:
    """Ask the LLM to invent a plausible dirty dataset and its ground truth."""
    rng = random.Random(seed)
    domain   = rng.choice(DOMAINS[difficulty])
    patterns = rng.sample(DIRTY_PATTERNS[difficulty], k=min(2 if difficulty=="easy" else 3, len(DIRTY_PATTERNS[difficulty])))

    system_prompt = """You are a data quality engineer creating test datasets for an AI training benchmark.
You must respond with ONLY a valid JSON object — no prose, no markdown fences.

The JSON must follow this exact schema:
{
  "task_name": "<descriptive name for this cleaning task>",
  "domain": "<the dataset domain>",
  "dirty_rows": [ { ...row dict... }, ... ],
  "schema_hint": { "column_name": "expected_type_description", ... },
  "instructions": "<multi-line string describing exactly what to clean>",
  "ground_truth": {
    "clean_rows": [ { ...row dict... }, ... ],
    "notes": "<brief explanation of what was fixed>"
  }
}

Rules:
- dirty_rows must have 6-9 rows with realistic data
- Each row must have 4-6 columns
- ground_truth.clean_rows must be the fully corrected version of dirty_rows
- The cleaning instructions must be explicit and unambiguous
- Include a primary key column (id or similar)
"""

    user_prompt = (
        f"Generate a dirty dataset for this domain: {domain}\n\n"
        f"The dataset must contain these specific data quality issues:\n"
        + "\n".join(f"  - {p}" for p in patterns)
        + f"\n\nDifficulty level: {difficulty}\n"
        f"Seed (for reproducibility reference): {seed}\n\n"
        f"Generate realistic data — real-looking names, plausible values, believable context."
    )

    response = client.chat.completions.create(
        model=_llm_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.9,   # high creativity for variety
        max_tokens=1500,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    data = json.loads(raw)

    # Build a task dict compatible with static format
    task_id = f"dynamic_{difficulty}_{hashlib.md5(str(seed).encode()).hexdigest()[:8]}"

    return {
        "task_id":      task_id,
        "task_name":    data["task_name"],
        "difficulty":   difficulty,
        "instructions": data["instructions"],
        "schema_hint":  data["schema_hint"],
        "dirty_rows":   data["dirty_rows"],
        "_ground_truth": data["ground_truth"],
        "_patterns":    patterns,
        "_domain":      domain,
        "_seed":        seed,
        "_dynamic":     True,
    }


def _generate_from_static_pool(difficulty: str, seed: int) -> Dict[str, Any]:
    """
    Fallback: rotate through a hand-crafted pool of varied datasets.
    Each call with a different seed picks a different variant.
    """
    rng = random.Random(seed)

    if difficulty == "easy":
        return _static_easy(rng)
    elif difficulty == "medium":
        return _static_medium(rng)
    else:
        return _static_hard(rng)


# ── Static fallback pools ─────────────────────────────────────────────────

def _static_easy(rng: random.Random) -> Dict[str, Any]:
    """Rotates between 3 easy dataset variants."""
    variant = rng.randint(0, 2)

    if variant == 0:
        # Variant A: HR contact list
        dirty_rows = [
            {"id": 1, "name": "alice johnson",  "phone": "(800) 123-4567", "department": "Engineering"},
            {"id": 2, "name": "Bob Smith",      "phone": "800.234.5678",   "department": "Marketing"},
            {"id": 3, "name": "CAROL WHITE",    "phone": "8003456789",     "department": "Sales"},
            {"id": 2, "name": "Bob Smith",      "phone": "800.234.5678",   "department": "Marketing"},  # dup
            {"id": 4, "name": "david brown",    "phone": "+1 800 456 7890","department": "Engineering"},
            {"id": 1, "name": "alice johnson",  "phone": "(800) 123-4567", "department": "Engineering"},  # dup
            {"id": 5, "name": "Eve Davis",      "phone": "800-567-8901",   "department": "HR"},
        ]
        ground_truth_rows = [
            {"id": 1, "name": "Alice Johnson", "phone": "+1-800-123-4567", "department": "Engineering"},
            {"id": 2, "name": "Bob Smith",     "phone": "+1-800-234-5678", "department": "Marketing"},
            {"id": 3, "name": "Carol White",   "phone": "+1-800-345-6789", "department": "Sales"},
            {"id": 4, "name": "David Brown",   "phone": "+1-800-456-7890", "department": "Engineering"},
            {"id": 5, "name": "Eve Davis",     "phone": "+1-800-567-8901", "department": "HR"},
        ]
        return {
            "task_id": f"static_easy_{rng.randint(1000,9999)}",
            "task_name": "Clean HR Contact Directory",
            "difficulty": "easy",
            "instructions": (
                "This HR contact list has two problems:\n"
                "1. Remove exact duplicate rows (keep first occurrence).\n"
                "2. Standardise all phone numbers to +1-XXX-XXX-XXXX format.\n"
                "3. Normalise all names to Title Case."
            ),
            "schema_hint": {"id": "int", "name": "str (Title Case)", "phone": "str (+1-XXX-XXX-XXXX)", "department": "str"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {"clean_rows": ground_truth_rows},
            "_dynamic": False,
        }
    elif variant == 1:
        # Variant B: supplier list
        dirty_rows = [
            {"id": 10, "company": "ACME CORP",     "email": "  Contact@acme.com  ", "region": "north"},
            {"id": 11, "company": "Beta Supplies",  "email": "info@BETA.COM",        "region": "SOUTH"},
            {"id": 12, "company": "Gamma Ltd",      "email": "sales@gamma.com",      "region": "East"},
            {"id": 10, "company": "ACME CORP",      "email": "  Contact@acme.com  ", "region": "north"},  # dup
            {"id": 13, "company": "delta inc",      "email": "hello@delta.com",      "region": "WEST"},
            {"id": 11, "company": "Beta Supplies",  "email": "info@BETA.COM",        "region": "SOUTH"},  # dup
        ]
        ground_truth_rows = [
            {"id": 10, "company": "Acme Corp",    "email": "contact@acme.com",  "region": "North"},
            {"id": 11, "company": "Beta Supplies","email": "info@beta.com",     "region": "South"},
            {"id": 12, "company": "Gamma Ltd",    "email": "sales@gamma.com",   "region": "East"},
            {"id": 13, "company": "Delta Inc",    "email": "hello@delta.com",   "region": "West"},
        ]
        return {
            "task_id": f"static_easy_{rng.randint(1000,9999)}",
            "task_name": "Clean Supplier Contact List",
            "difficulty": "easy",
            "instructions": (
                "This supplier list has three problems:\n"
                "1. Remove exact duplicate rows (keep first occurrence).\n"
                "2. Normalise company names and region to Title Case.\n"
                "3. Strip whitespace and lowercase all email addresses."
            ),
            "schema_hint": {"id": "int", "company": "str (Title Case)", "email": "str (lowercase)", "region": "str (Title Case)"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {"clean_rows": ground_truth_rows},
            "_dynamic": False,
        }
    else:
        # Variant C: student list
        dirty_rows = [
            {"id": 1, "name": "PRIYA SHARMA",   "phone": "(555) 001-0001", "course": "computer science"},
            {"id": 2, "name": "james o'brien",  "phone": "555.002.0002",   "course": "MATHEMATICS"},
            {"id": 3, "name": "Lena Muller",    "phone": "5550030003",     "course": "Physics"},
            {"id": 4, "name": "WANG LEI",       "phone": "+1 555 004 0004","course": "chemistry"},
            {"id": 2, "name": "james o'brien",  "phone": "555.002.0002",   "course": "MATHEMATICS"},  # dup
            {"id": 5, "name": "Sofia Rossi",    "phone": "555-005-0005",   "course": "BIOLOGY"},
            {"id": 1, "name": "PRIYA SHARMA",   "phone": "(555) 001-0001", "course": "computer science"},  # dup
        ]
        ground_truth_rows = [
            {"id": 1, "name": "Priya Sharma",  "phone": "+1-555-001-0001", "course": "Computer Science"},
            {"id": 2, "name": "James O'Brien", "phone": "+1-555-002-0002", "course": "Mathematics"},
            {"id": 3, "name": "Lena Muller",   "phone": "+1-555-003-0003", "course": "Physics"},
            {"id": 4, "name": "Wang Lei",      "phone": "+1-555-004-0004", "course": "Chemistry"},
            {"id": 5, "name": "Sofia Rossi",   "phone": "+1-555-005-0005", "course": "Biology"},
        ]
        return {
            "task_id": f"static_easy_{rng.randint(1000,9999)}",
            "task_name": "Clean Student Enrollment Records",
            "difficulty": "easy",
            "instructions": (
                "This student enrollment list has three problems:\n"
                "1. Remove exact duplicate rows.\n"
                "2. Standardise phone numbers to +1-XXX-XXX-XXXX format.\n"
                "3. Normalise name and course to Title Case."
            ),
            "schema_hint": {"id": "int", "name": "str (Title Case)", "phone": "str (+1-XXX-XXX-XXXX)", "course": "str (Title Case)"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {"clean_rows": ground_truth_rows},
            "_dynamic": False,
        }


def _static_medium(rng: random.Random) -> Dict[str, Any]:
    """Rotates between 2 medium dataset variants."""
    variant = rng.randint(0, 1)

    if variant == 0:
        dirty_rows = [
            {"id": 1, "employee": "Alice", "age": "32 years", "salary": 95000.0,  "dept": "ENGINEERING", "active": "yes"},
            {"id": 2, "employee": "Bob",   "age": "28 years", "salary": None,      "dept": "marketing",   "active": True},
            {"id": 3, "employee": "Carol", "age": "45 years", "salary": 110000.0, "dept": "Engineering",  "active": False},
            {"id": 4, "employee": "Dave",  "age": "38 years", "salary": 88000.0,  "dept": "MARKETING",   "active": "no"},
            {"id": 5, "employee": "Eve",   "age": "30 years", "salary": None,     "dept": "sales",        "active": "yes"},
            {"id": 6, "employee": "Frank", "age": "52 years", "salary": 102000.0, "dept": "Sales",        "active": True},
        ]
        # median of [95000,110000,88000,102000] = (95000+102000)/2 = 98500
        return {
            "task_id": f"static_medium_{rng.randint(1000,9999)}",
            "task_name": "Clean Employee Payroll Records",
            "difficulty": "medium",
            "instructions": (
                "This payroll table has four issues:\n"
                "1. Strip ' years' suffix from age and cast to integer.\n"
                "2. Fill None salary with the median of non-null salaries (round to 2dp).\n"
                "3. Normalise dept to Title Case.\n"
                "4. Convert active column to boolean (yes/True→True, no/False→False)."
            ),
            "schema_hint": {"id": "int", "employee": "str", "age": "int", "salary": "float", "dept": "str (Title Case)", "active": "bool"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {
                "clean_rows": [
                    {"id": 1, "employee": "Alice", "age": 32, "salary": 95000.0,  "dept": "Engineering", "active": True},
                    {"id": 2, "employee": "Bob",   "age": 28, "salary": 98500.0,  "dept": "Marketing",   "active": True},
                    {"id": 3, "employee": "Carol", "age": 45, "salary": 110000.0, "dept": "Engineering", "active": False},
                    {"id": 4, "employee": "Dave",  "age": 38, "salary": 88000.0,  "dept": "Marketing",   "active": False},
                    {"id": 5, "employee": "Eve",   "age": 30, "salary": 98500.0,  "dept": "Sales",       "active": True},
                    {"id": 6, "employee": "Frank", "age": 52, "salary": 102000.0, "dept": "Sales",       "active": True},
                ],
                "median_salary": 98500.0,
            },
            "_dynamic": False,
        }
    else:
        dirty_rows = [
            {"id": 1, "product": "Widget A", "price": "$29.99",  "category": "HARDWARE",  "in_stock": "yes", "rating": "4.5 stars"},
            {"id": 2, "product": "Gadget B", "price": "$149.00", "category": "electronics","in_stock": "no",  "rating": "3.8 stars"},
            {"id": 3, "product": "Tool C",   "price": "$9.50",   "category": "Hardware",  "in_stock": True,  "rating": "4.9 stars"},
            {"id": 4, "product": "Part D",   "price": None,      "category": "ELECTRONICS","in_stock": "yes","rating": "4.2 stars"},
            {"id": 5, "product": "Kit E",    "price": "$75.00",  "category": "tools",     "in_stock": False, "rating": "3.5 stars"},
            {"id": 6, "product": "Set F",    "price": "$55.00",  "category": "Tools",     "in_stock": "yes", "rating": None},
        ]
        # median price of [29.99,149.00,9.50,75.00,55.00] = 55.00; median rating of [4.5,3.8,4.9,4.2,3.5] = 4.2
        return {
            "task_id": f"static_medium_{rng.randint(1000,9999)}",
            "task_name": "Clean Product Catalogue Records",
            "difficulty": "medium",
            "instructions": (
                "This product catalogue has four issues:\n"
                "1. Strip the leading '$' from price and cast to float; fill None with median price.\n"
                "2. Strip ' stars' from rating and cast to float; fill None with median rating.\n"
                "3. Normalise category to Title Case.\n"
                "4. Convert in_stock to boolean."
            ),
            "schema_hint": {"id": "int", "product": "str", "price": "float", "category": "str (Title Case)", "in_stock": "bool", "rating": "float"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {
                "clean_rows": [
                    {"id": 1, "product": "Widget A", "price": 29.99,  "category": "Hardware",   "in_stock": True,  "rating": 4.5},
                    {"id": 2, "product": "Gadget B", "price": 149.00, "category": "Electronics","in_stock": False, "rating": 3.8},
                    {"id": 3, "product": "Tool C",   "price": 9.50,   "category": "Hardware",   "in_stock": True,  "rating": 4.9},
                    {"id": 4, "product": "Part D",   "price": 55.00,  "category": "Electronics","in_stock": True,  "rating": 4.2},
                    {"id": 5, "product": "Kit E",    "price": 75.00,  "category": "Tools",      "in_stock": False, "rating": 3.5},
                    {"id": 6, "product": "Set F",    "price": 55.00,  "category": "Tools",      "in_stock": True,  "rating": 4.2},
                ],
                "median_price": 55.00,
                "median_rating": 4.2,
            },
            "_dynamic": False,
        }


def _static_hard(rng: random.Random) -> Dict[str, Any]:
    """Returns one of 2 hard dataset variants."""
    variant = rng.randint(0, 1)

    if variant == 0:
        dirty_rows = [
            {"order_id": 1,  "customer_id": 101, "product": "Widget A", "price": 29.99,   "order_date": "01/15/2024", "weight_kg": 1.5},
            {"order_id": 2,  "customer_id": 102, "product": "Gadget B", "price": 99999.0, "order_date": "2024-02-20", "weight_kg": 2500},
            {"order_id": 3,  "customer_id": 999, "product": "Thing C",  "price": 49.99,   "order_date": "15-Mar-2024","weight_kg": 0.8},
            {"order_id": 4,  "customer_id": 103, "product": "Widget A", "price": -5.00,   "order_date": "04/01/2024", "weight_kg": 1500},
            {"order_id": 5,  "customer_id": 104, "product": "Gadget B", "price": 89.50,   "order_date": "2024-05-10", "weight_kg": 0.5},
            {"order_id": 6,  "customer_id": 101, "product": "Donut D",  "price": 12.00,   "order_date": "30-Jun-2024","weight_kg": 250},
            {"order_id": 7,  "customer_id": 888, "product": "Part E",   "price": 5.00,    "order_date": "07/07/2024", "weight_kg": 0.3},
            {"order_id": 8,  "customer_id": 102, "product": "Thing C",  "price": 75.00,   "order_date": "2024-08-22", "weight_kg": 0.9},
        ]
        return {
            "task_id": f"static_hard_{rng.randint(1000,9999)}",
            "task_name": "Multi-Issue: Outliers, Dates, Units and Integrity",
            "difficulty": "hard",
            "instructions": (
                "This orders table has FOUR data quality problems:\n"
                "1. OUTLIERS: price > 10000 or < 0 → replace with None.\n"
                "2. DATES: mixed formats (MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY) → normalise to YYYY-MM-DD.\n"
                "3. UNITS: weight_kg values > 500 are in grams → divide by 1000, round to 3dp.\n"
                "4. INTEGRITY: drop rows where customer_id not in [101, 102, 103, 104]."
            ),
            "schema_hint": {"order_id": "int", "customer_id": "int", "product": "str", "price": "float|null", "order_date": "str (YYYY-MM-DD)", "weight_kg": "float"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {
                "valid_customer_ids": [101, 102, 103, 104],
                "clean_rows": [
                    {"order_id": 1, "customer_id": 101, "product": "Widget A", "price": 29.99,  "order_date": "2024-01-15", "weight_kg": 1.5},
                    {"order_id": 2, "customer_id": 102, "product": "Gadget B", "price": None,   "order_date": "2024-02-20", "weight_kg": 2.5},
                    {"order_id": 4, "customer_id": 103, "product": "Widget A", "price": None,   "order_date": "2024-04-01", "weight_kg": 1.5},
                    {"order_id": 5, "customer_id": 104, "product": "Gadget B", "price": 89.50,  "order_date": "2024-05-10", "weight_kg": 0.5},
                    {"order_id": 6, "customer_id": 101, "product": "Donut D",  "price": 12.00,  "order_date": "2024-06-30", "weight_kg": 0.25},
                    {"order_id": 8, "customer_id": 102, "product": "Thing C",  "price": 75.00,  "order_date": "2024-08-22", "weight_kg": 0.9},
                ],
            },
            "_dynamic": False,
        }
    else:
        dirty_rows = [
            {"tx_id": 1,  "account_id": 201, "amount": 1250.00,   "currency": "USD", "tx_date": "03/01/2024",   "category": "SALARY"},
            {"tx_id": 2,  "account_id": 202, "amount": -999999.0, "currency": "EUR", "tx_date": "2024-03-05",   "category": "transfer"},
            {"tx_id": 3,  "account_id": 999, "amount": 340.00,    "currency": "GBP", "tx_date": "10-Mar-2024",  "category": "Utilities"},
            {"tx_id": 4,  "account_id": 201, "amount": 85.50,     "currency": "USD", "tx_date": "15/03/2024",   "category": "groceries"},
            {"tx_id": 5,  "account_id": 203, "amount": 500000.0,  "currency": "USD", "tx_date": "2024-03-18",   "category": "RENT"},
            {"tx_id": 6,  "account_id": 202, "amount": 120.00,    "currency": "EUR", "tx_date": "20-Mar-2024",  "category": "Entertainment"},
            {"tx_id": 7,  "account_id": 888, "amount": 60.00,     "currency": "GBP", "tx_date": "22/03/2024",   "category": "transport"},
            {"tx_id": 8,  "account_id": 203, "amount": 975.00,    "currency": "USD", "tx_date": "2024-03-28",   "category": "SALARY"},
        ]
        return {
            "task_id": f"static_hard_{rng.randint(1000,9999)}",
            "task_name": "Multi-Issue: Financial Transaction Cleaning",
            "difficulty": "hard",
            "instructions": (
                "This financial transaction log has FOUR data quality problems:\n"
                "1. OUTLIERS: amount > 100000 or < 0 → replace with None.\n"
                "2. DATES: mixed formats (MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY, DD/MM/YYYY) → normalise to YYYY-MM-DD.\n"
                "3. CATEGORIES: normalise to Title Case.\n"
                "4. INTEGRITY: drop rows where account_id not in [201, 202, 203]."
            ),
            "schema_hint": {"tx_id": "int", "account_id": "int", "amount": "float|null", "currency": "str", "tx_date": "str (YYYY-MM-DD)", "category": "str (Title Case)"},
            "dirty_rows": dirty_rows,
            "_ground_truth": {
                "valid_account_ids": [201, 202, 203],
                "clean_rows": [
                    {"tx_id": 1, "account_id": 201, "amount": 1250.00, "currency": "USD", "tx_date": "2024-03-01", "category": "Salary"},
                    {"tx_id": 2, "account_id": 202, "amount": None,    "currency": "EUR", "tx_date": "2024-03-05", "category": "Transfer"},
                    {"tx_id": 4, "account_id": 201, "amount": 85.50,   "currency": "USD", "tx_date": "2024-03-15", "category": "Groceries"},
                    {"tx_id": 5, "account_id": 203, "amount": None,    "currency": "USD", "tx_date": "2024-03-18", "category": "Rent"},
                    {"tx_id": 6, "account_id": 202, "amount": 120.00,  "currency": "EUR", "tx_date": "2024-03-20", "category": "Entertainment"},
                    {"tx_id": 8, "account_id": 203, "amount": 975.00,  "currency": "USD", "tx_date": "2024-03-28", "category": "Salary"},
                ],
            },
            "_dynamic": False,
        }
