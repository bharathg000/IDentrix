from datetime import datetime
from database import create_violation


# =====================================================
# PROCESS VIOLATION (Force Individual Entries)
# =====================================================

def process_violation(
    person_id,
    violation_type
):
    """
    Directly creates a brand-new individual database record
    for every unique violation occurrence, bypassing grouping/counting.
    """
    
    # Always create a brand-new violation row with current timestamp
    violation_id = create_violation(
        person_id,
        violation_type
    )

    return {
        "recorded": True,
        "new_record": True,
        "count": 1,
        "violation_id": violation_id
    }