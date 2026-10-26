
def check_compliance(id_card):
    rules = {
        "ID Card": bool(id_card)
    }

    issues = [
        rule
        for rule, status in rules.items()
        if not status
    ]

    if issues:
        result = "NOT COMPLIANT"
    else:
        result = "COMPLIANT"

    return result, rules, issues
