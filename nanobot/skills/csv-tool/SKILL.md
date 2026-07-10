---
name: csv-tool
description: Process CSV data — filter, sort, aggregate, convert, validate — using Python's built-in csv module (no extra dependencies).
metadata:
  nanobot:
    emoji: "📊"
    requires:
      bins: ["python3"]
---

# CSV Tool

Process CSV data with Python's standard library `csv` module. No extra packages needed.

> **Use cases**: filter rows, sort by column, compute aggregates, convert to JSON, validate structure, sample large files.

---

## Basic Operations

### 1. Preview structure (column names + sample)

```bash
python3 -c "
import csv, sys
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
    print('Columns:', ', '.join(rows[0].keys()))
    print(f'Rows: {len(rows)}')
    for r in rows[:3]:
        print(r)
"
```

### 2. Filter rows (column value match)

```bash
python3 -c "
import csv
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
    filtered = [r for r in rows if r['status'] == 'active']
    w = csv.DictWriter(open('active.csv','w',newline=''), fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(filtered)
    print(f'Wrote {len(filtered)} rows')
"
```

### 3. Sort by column (numeric or text)

```bash
python3 -c "
import csv
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r['price']))     # numeric ascending
    # rows.sort(key=lambda r: r['name'].lower())   # text ascending
    for r in rows[:10]:
        print(f\"{r['name']:20s} {r['price']:>8s}\")
"
```

### 4. Aggregate (group by + sum/count)

```bash
python3 -c "
import csv
from collections import defaultdict
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
    by_cat = defaultdict(lambda: {'count':0,'total':0.0})
    for r in rows:
        c = by_cat[r['category']]
        c['count'] += 1
        c['total'] += float(r['amount'])
    for cat, v in sorted(by_cat.items()):
        print(f\"{cat:20s} count={v['count']:4d}  total={v['total']:>10.2f}\")
"
```

### 5. Convert CSV → JSON

```bash
python3 -c "
import csv, json
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
    print(json.dumps(rows, ensure_ascii=False, indent=2))
" > data.json
```

### 6. Convert CSV → JSONL (one JSON object per line)

```bash
python3 -c "
import csv, json
with open('data.csv') as f, open('data.jsonl','w') as out:
    for row in csv.DictReader(f):
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
"
```

---

## Advanced

### 7. Validate structure (check column presence, non-empty required fields)

```bash
python3 -c "
import csv, sys
errors = []
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
required_cols = ['name', 'email', 'amount']
for col in required_cols:
    if col not in rows[0]:
        errors.append(f'Missing column: {col}')
for i, r in enumerate(rows, 2):
    if not r.get('email','').strip():
        errors.append(f'Row {i}: empty email')
    try:
        float(r.get('amount', ''))
    except ValueError:
        errors.append(f'Row {i}: invalid amount {r[\"amount\"]}')
if errors:
    for e in errors:
        print(f'ERROR: {e}')
    sys.exit(1)
else:
    print(f'OK: {len(rows)} rows, all valid')
"
```

### 8. Sample large file (first N rows)

```bash
head -101 data.csv > sample.csv    # header + 100 rows
```

### 9. Column stats (min, max, mean, median)

```bash
python3 -c "
import csv, statistics
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
values = [float(r['amount']) for r in rows if r['amount'].strip()]
print(f'min:    {min(values):.2f}')
print(f'max:    {max(values):.2f}')
print(f'mean:   {statistics.mean(values):.2f}')
print(f'median: {statistics.median(values):.2f}')
print(f'count:  {len(values)}')
"
```

### 10. Deduplicate by column

```bash
python3 -c "
import csv
with open('data.csv') as f:
    rows = list(csv.DictReader(f))
seen = set()
deduped = []
for r in rows:
    key = r['email'].strip().lower()
    if key not in seen:
        seen.add(key)
        deduped.append(r)
w = csv.DictWriter(open('deduped.csv','w',newline=''), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(deduped)
print(f'{len(rows)} -> {len(deduped)} rows after dedup')
"
```

---

## Tips

- Always use `newline=''` when opening a CSV file for writing with `csv.writer` / `csv.DictWriter`
- Use `csv.Sniffer().sniff(data)` to auto-detect delimiter if the file might be TSV or pipe-separated
- For **very large files**, process row-by-row instead of reading all into memory
- Use `io.StringIO(text)` to process CSV data from a string variable instead of a file
