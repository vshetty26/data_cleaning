"""
Data Cleaning Environment — Task Bank & Graders

Three tasks (easy → medium → hard) each with a deterministic grader
that returns a score in (0.0, 1.0) — strictly between 0 and 1.

Task themes:
  task_001 (easy)   — Remove duplicates + standardise a phone number column
  task_002 (medium) — Handle missing values, fix data types, normalise strings
  task_003 (hard)   — Multi-issue: outliers, inconsistent dates, mixed units,
                       referential integrity, duplicate detection
"""

from typing import Any, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# TASK DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

TASKS = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    {
        "task_id": "task_001",
        "task_name": "Deduplicate & Standardise Phone Numbers",
        "difficulty": "easy",
        "instructions": (
            "This customer table has exact duplicate rows and phone numbers in "
            "inconsistent formats. Your job:\n"
            "1. Remove all exact duplicate rows (keep first occurrence).\n"
            "2. Standardise all phone numbers to the format +1-XXX-XXX-XXXX "
            "   (strip spaces, dashes, parentheses, then reformat).\n"
            "3. Return the cleaned rows in the same column order."
        ),
        "schema_hint": {
            "id": "int",
            "name": "str",
            "phone": "str (+1-XXX-XXX-XXXX)",
            "email": "str",
        },
        "dirty_rows": [
            {"id": 1, "name": "Alice",   "phone": "(555) 123-4567", "email": "alice@example.com"},
            {"id": 2, "name": "Bob",     "phone": "555.234.5678",   "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "phone": "5553456789",     "email": "charlie@example.com"},
            {"id": 2, "name": "Bob",     "phone": "555.234.5678",   "email": "bob@example.com"},   # dup
            {"id": 4, "name": "Diana",   "phone": "+1 555 456 7890","email": "diana@example.com"},
            {"id": 1, "name": "Alice",   "phone": "(555) 123-4567", "email": "alice@example.com"}, # dup
            {"id": 5, "name": "Eve",     "phone": "555-567-8901",   "email": "eve@example.com"},
        ],
        # Ground truth
        "expected_row_count": 5,
        "expected_phones": {
            1: "+1-555-123-4567",
            2: "+1-555-234-5678",
            3: "+1-555-345-6789",
            4: "+1-555-456-7890",
            5: "+1-555-567-8901",
        },
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "task_id": "task_002",
        "task_name": "Fix Missing Values, Types & Inconsistent Strings",
        "difficulty": "medium",
        "instructions": (
            "This employee salary table has several issues:\n"
            "1. Missing 'salary' values — fill with the median salary of the "
            "   non-null rows (round to 2 decimal places).\n"
            "2. The 'age' column contains strings like '30 years' — extract "
            "   the integer value only.\n"
            "3. The 'department' column has inconsistent capitalisation "
            "   (e.g. 'engineering', 'ENGINEERING') — normalise to Title Case.\n"
            "4. The 'active' column is a mix of True/False booleans and "
            "   'yes'/'no' strings — convert everything to a boolean.\n"
            "Return the cleaned rows with columns: id, name, age, department, salary, active."
        ),
        "schema_hint": {
            "id": "int",
            "name": "str",
            "age": "int",
            "department": "str (Title Case)",
            "salary": "float",
            "active": "bool",
        },
        "dirty_rows": [
            {"id": 1, "name": "Alice",   "age": "30 years", "department": "engineering",  "salary": 95000.0,  "active": True},
            {"id": 2, "name": "Bob",     "age": "25 years", "department": "MARKETING",    "salary": None,     "active": "yes"},
            {"id": 3, "name": "Charlie", "age": "40 years", "department": "Engineering",  "salary": 110000.0, "active": False},
            {"id": 4, "name": "Diana",   "age": "35 years", "department": "marketing",    "salary": 80000.0,  "active": "no"},
            {"id": 5, "name": "Eve",     "age": "28 years", "department": "ENGINEERING",  "salary": None,     "active": "yes"},
            {"id": 6, "name": "Frank",   "age": "45 years", "department": "Sales",        "salary": 90000.0,  "active": True},
        ],
        # Median of [95000, 110000, 80000, 90000] = (90000+95000)/2 = 92500.0
        "expected_median_salary": 92500.0,
        "expected_departments": {
            1: "Engineering", 2: "Marketing", 3: "Engineering",
            4: "Marketing",   5: "Engineering", 6: "Sales",
        },
        "expected_active": {1: True, 2: True, 3: False, 4: False, 5: True, 6: True},
    },

    # ── HARD ─────────────────────────────────────────────────────────────────
    {
        "task_id": "task_003",
        "task_name": "Multi-Issue: Outliers, Dates, Units & Integrity",
        "difficulty": "hard",
        "instructions": (
            "This product orders table has FOUR distinct data quality problems:\n\n"
            "1. OUTLIERS: The 'price' column has clear outliers (values > 10,000 "
            "   or < 0) — replace them with None (null).\n\n"
            "2. INCONSISTENT DATES: The 'order_date' column mixes formats "
            "   (MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY) — normalise ALL to "
            "   ISO 8601 format: YYYY-MM-DD.\n\n"
            "3. MIXED UNITS: The 'weight_kg' column has some values entered in "
            "   grams (values > 500 are grams, values <= 500 are already kg) "
            "   — convert gram entries to kg (divide by 1000, round to 3 dp).\n\n"
            "4. REFERENTIAL INTEGRITY: The 'customer_id' column references "
            "   valid customers [101, 102, 103, 104]. Rows with unknown "
            "   customer_ids must be DROPPED entirely.\n\n"
            "Return the cleaned rows with columns: "
            "order_id, customer_id, product, price, order_date, weight_kg."
        ),
        "schema_hint": {
            "order_id": "int",
            "customer_id": "int (must be in [101,102,103,104])",
            "product": "str",
            "price": "float or null",
            "order_date": "str (YYYY-MM-DD)",
            "weight_kg": "float",
        },
        "dirty_rows": [
            {"order_id": 1,  "customer_id": 101, "product": "Widget A", "price": 29.99,    "order_date": "01/15/2024",   "weight_kg": 1.5},
            {"order_id": 2,  "customer_id": 102, "product": "Gadget B", "price": 99999.0,  "order_date": "2024-02-20",   "weight_kg": 2500},   # outlier price, weight in grams
            {"order_id": 3,  "customer_id": 999, "product": "Thing C",  "price": 49.99,    "order_date": "15-Mar-2024",  "weight_kg": 0.8},    # bad customer_id → drop
            {"order_id": 4,  "customer_id": 103, "product": "Widget A", "price": -5.00,    "order_date": "04/01/2024",   "weight_kg": 1500},   # negative price outlier, weight in grams
            {"order_id": 5,  "customer_id": 104, "product": "Gadget B", "price": 89.50,    "order_date": "2024-05-10",   "weight_kg": 0.5},
            {"order_id": 6,  "customer_id": 101, "product": "Donut D",  "price": 12.00,    "order_date": "30-Jun-2024",  "weight_kg": 250},    # weight in grams
            {"order_id": 7,  "customer_id": 888, "product": "Part E",   "price": 5.00,     "order_date": "07/07/2024",   "weight_kg": 0.3},    # bad customer_id → drop
            {"order_id": 8,  "customer_id": 102, "product": "Thing C",  "price": 75.00,    "order_date": "2024-08-22",   "weight_kg": 0.9},
        ],
        # Ground truth (after all 4 operations)
        "valid_customer_ids": [101, 102, 103, 104],
        "expected_row_count": 6,  # rows 3 and 7 dropped
        "expected_dates": {
            1: "2024-01-15",
            2: "2024-02-20",
            4: "2024-04-01",
            5: "2024-05-10",
            6: "2024-06-30",
            8: "2024-08-22",
        },
        "expected_prices": {
            1: 29.99,
            2: None,    # outlier → null
            4: None,    # negative outlier → null
            5: 89.50,
            6: 12.00,
            8: 75.00,
        },
        "expected_weights_kg": {
            1: 1.5,
            2: 2.5,     # 2500g → 2.5kg
            4: 1.5,     # 1500g → 1.5kg
            5: 0.5,
            6: 0.250,   # 250g → 0.25kg
            8: 0.9,
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _rows_by_id(rows: List[Dict[str, Any]], id_col: str) -> Dict[Any, Dict]:
    return {r[id_col]: r for r in rows if id_col in r}


def _clamp_score(score: float) -> float:
    """Clamp score to strictly (0, 1) — validators reject exact 0.0 and 1.0."""
    s = round(min(score, 1.0), 4)
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    return s


# ──────────────────────────────────────────────────────────────────────────────
# GRADERS
# ──────────────────────────────────────────────────────────────────────────────

def grade_task_001(action_dict: Dict[str, Any]) -> Tuple[float, str]:
    """Easy — dedup + phone normalisation."""
    rows: List[Dict] = action_dict.get("cleaned_rows", [])
    feedback = []
    score = 0.0
    task = TASKS[0]

    # 0.4 pts: correct row count after dedup
    if len(rows) == task["expected_row_count"]:
        score += 0.4
        feedback.append(f"✅ Correct row count after dedup: {len(rows)} (+0.4)")
    else:
        feedback.append(f"❌ Expected {task['expected_row_count']} rows, got {len(rows)}")

    # 0.6 pts: phone normalisation (0.12 per row)
    by_id = _rows_by_id(rows, "id")
    correct_phones = 0
    for cid, expected_phone in task["expected_phones"].items():
        row = by_id.get(cid)
        if row and str(row.get("phone", "")).strip() == expected_phone:
            correct_phones += 1
        else:
            got = row.get("phone") if row else "row missing"
            feedback.append(f"❌ id={cid}: expected phone {expected_phone}, got {got}")
    phone_score = round(0.12 * correct_phones, 2)
    score += phone_score
    if correct_phones == len(task["expected_phones"]):
        feedback.append(f"✅ All {correct_phones} phone numbers correctly normalised (+{phone_score})")
    else:
        feedback.append(f"⚠️  {correct_phones}/{len(task['expected_phones'])} phones correct (+{phone_score})")

    return _clamp_score(score), "\n".join(feedback)


def grade_task_002(action_dict: Dict[str, Any]) -> Tuple[float, str]:
    """Medium — missing values, types, strings."""
    rows: List[Dict] = action_dict.get("cleaned_rows", [])
    feedback = []
    score = 0.0
    task = TASKS[1]
    by_id = _rows_by_id(rows, "id")

    # 0.25 pts: age is int (no 'years' suffix)
    age_ok = all(
        isinstance(r.get("age"), int)
        for r in rows
    )
    if age_ok:
        score += 0.25
        feedback.append("✅ All 'age' values are integers (+0.25)")
    else:
        feedback.append("❌ Some 'age' values still have string/suffix format")

    # 0.25 pts: department Title Case
    dept_correct = sum(
        1 for cid, expected in task["expected_departments"].items()
        if by_id.get(cid, {}).get("department") == expected
    )
    dept_score = round(0.25 * dept_correct / len(task["expected_departments"]), 2)
    score += dept_score
    if dept_correct == len(task["expected_departments"]):
        feedback.append(f"✅ All departments in Title Case (+0.25)")
    else:
        feedback.append(f"⚠️  {dept_correct}/{len(task['expected_departments'])} departments correct (+{dept_score})")

    # 0.25 pts: salary median fill
    salary_ok = 0
    for cid in [2, 5]:   # rows that had None salary
        row = by_id.get(cid)
        if row:
            sal = row.get("salary")
            try:
                if abs(float(sal) - task["expected_median_salary"]) < 0.01:
                    salary_ok += 1
                else:
                    feedback.append(f"❌ id={cid}: salary should be {task['expected_median_salary']}, got {sal}")
            except (TypeError, ValueError):
                feedback.append(f"❌ id={cid}: salary is not a number: {sal}")
    sal_score = round(0.25 * salary_ok / 2, 2)
    score += sal_score
    if salary_ok == 2:
        feedback.append(f"✅ Missing salaries filled with correct median {task['expected_median_salary']} (+0.25)")
    else:
        feedback.append(f"⚠️  {salary_ok}/2 missing salary rows correct (+{sal_score})")

    # 0.25 pts: active booleans
    active_correct = sum(
        1 for cid, expected in task["expected_active"].items()
        if by_id.get(cid, {}).get("active") == expected
    )
    act_score = round(0.25 * active_correct / len(task["expected_active"]), 2)
    score += act_score
    if active_correct == len(task["expected_active"]):
        feedback.append(f"✅ All 'active' values correctly cast to bool (+0.25)")
    else:
        feedback.append(f"⚠️  {active_correct}/{len(task['expected_active'])} 'active' values correct (+{act_score})")

    return _clamp_score(score), "\n".join(feedback)


def grade_task_003(action_dict: Dict[str, Any]) -> Tuple[float, str]:
    """Hard — outliers, dates, units, referential integrity."""
    rows: List[Dict] = action_dict.get("cleaned_rows", [])
    feedback = []
    score = 0.0
    task = TASKS[2]
    by_oid = _rows_by_id(rows, "order_id")

    # 0.25 pts: referential integrity (bad customer rows dropped)
    bad_ids_present = any(r.get("customer_id") not in task["valid_customer_ids"] for r in rows)
    correct_count   = len(rows) == task["expected_row_count"]
    if not bad_ids_present and correct_count:
        score += 0.25
        feedback.append(f"✅ Referential integrity: {len(rows)} rows, bad customer_ids dropped (+0.25)")
    else:
        if bad_ids_present:
            feedback.append("❌ Rows with invalid customer_ids still present")
        if not correct_count:
            feedback.append(f"❌ Expected {task['expected_row_count']} rows, got {len(rows)}")

    # 0.25 pts: price outliers → None
    price_ok = 0
    for oid, expected_price in task["expected_prices"].items():
        row = by_oid.get(oid)
        if not row:
            continue
        got = row.get("price")
        if expected_price is None:
            if got is None:
                price_ok += 1
            else:
                feedback.append(f"❌ order_id={oid}: price outlier should be null, got {got}")
        else:
            try:
                if abs(float(got) - expected_price) < 0.01:
                    price_ok += 1
                else:
                    feedback.append(f"❌ order_id={oid}: expected price {expected_price}, got {got}")
            except (TypeError, ValueError):
                feedback.append(f"❌ order_id={oid}: price not a number: {got}")
    price_score = round(0.25 * price_ok / len(task["expected_prices"]), 2)
    score += price_score
    if price_ok == len(task["expected_prices"]):
        feedback.append(f"✅ All price outliers replaced with null (+0.25)")
    else:
        feedback.append(f"⚠️  {price_ok}/{len(task['expected_prices'])} prices correct (+{price_score})")

    # 0.25 pts: date normalisation
    date_ok = 0
    for oid, expected_date in task["expected_dates"].items():
        row = by_oid.get(oid)
        if not row:
            continue
        got = str(row.get("order_date", "")).strip()
        if got == expected_date:
            date_ok += 1
        else:
            feedback.append(f"❌ order_id={oid}: expected date {expected_date}, got {got}")
    date_score = round(0.25 * date_ok / len(task["expected_dates"]), 2)
    score += date_score
    if date_ok == len(task["expected_dates"]):
        feedback.append(f"✅ All dates normalised to ISO 8601 (+0.25)")
    else:
        feedback.append(f"⚠️  {date_ok}/{len(task['expected_dates'])} dates correct (+{date_score})")

    # 0.25 pts: weight unit conversion
    weight_ok = 0
    for oid, expected_w in task["expected_weights_kg"].items():
        row = by_oid.get(oid)
        if not row:
            continue
        got = row.get("weight_kg")
        try:
            if abs(float(got) - expected_w) < 0.005:
                weight_ok += 1
            else:
                feedback.append(f"❌ order_id={oid}: expected weight_kg {expected_w}, got {got}")
        except (TypeError, ValueError):
            feedback.append(f"❌ order_id={oid}: weight_kg not a number: {got}")
    weight_score = round(0.25 * weight_ok / len(task["expected_weights_kg"]), 2)
    score += weight_score
    if weight_ok == len(task["expected_weights_kg"]):
        feedback.append(f"✅ All weights correctly converted to kg (+0.25)")
    else:
        feedback.append(f"⚠️  {weight_ok}/{len(task['expected_weights_kg'])} weights correct (+{weight_score})")

    return _clamp_score(score), "\n".join(feedback)


GRADERS = {
    "task_001": grade_task_001,
    "task_002": grade_task_002,
    "task_003": grade_task_003,
}
