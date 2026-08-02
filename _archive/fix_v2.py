# Disable date_prefix in batch_generate_v2.py - replace all date_prefix calls
with open('batch_generate_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace date_prefix calls with empty prefix and reference date
content = content.replace(
    'pref, d = date_prefix(random)',
    'pref, d = "", REFERENCE_DATE'
)

with open('batch_generate_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done - date prefixes disabled')
