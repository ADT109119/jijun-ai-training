# Test what sub_categories the current code produces
from generate_dataset import CATEGORIES, ACCOUNTS
import random

sub_cats = CATEGORIES  # This is what the code now does: sub_categories = categories
sub_accs = ACCOUNTS

print('sub_categories count:', len(sub_cats))
print('sub_categories:', sub_cats)
print()

# Now simulate what tool_definition looks like
import json
tool_def = {
    "name": "add_record",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": sub_cats},
            "account": {"type": "string", "enum": sub_accs},
        }
    }
}
print('tool_def enum count:', len(tool_def['parameters']['properties']['category']['enum']))
print('All 15 present:', all(c in tool_def['parameters']['properties']['category']['enum'] for c in CATEGORIES))
