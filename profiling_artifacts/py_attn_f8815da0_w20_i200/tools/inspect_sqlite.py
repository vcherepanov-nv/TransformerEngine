import json
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as db:
    tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print(json.dumps({'tables': tables}))
    print(json.dumps({'thread_names_schema': list(db.execute('PRAGMA table_info(ThreadNames)')), 'thread_names': list(db.execute('SELECT * FROM ThreadNames'))}))
    print(json.dumps({'schemas': list(db.execute('SELECT * FROM NVTX_PAYLOAD_SCHEMAS'))}))
    print(json.dumps({'diagnostics': list(db.execute('SELECT * FROM DIAGNOSTIC_EVENT'))}))
    print(json.dumps({'python_examples': list(db.execute('SELECT * FROM NVTX_EVENTS WHERE domainId = 3 LIMIT 6'))}, default=str))
    print(json.dumps({'domain_counts': list(db.execute('SELECT domainId, eventType, COUNT(*) FROM NVTX_EVENTS GROUP BY domainId, eventType'))}))
    print(json.dumps({'python_schema_entries': list(db.execute('SELECT * FROM NVTX_PAYLOAD_SCHEMA_ENTRIES WHERE domainId = 3'))}))
    for name in tables:
        if any(word in name.lower() for word in ('sampl', 'callchain', 'composite', 'python', 'nvtx', 'diagnostic')):
            quoted = '"' + name.replace('"', '""') + '"'
            columns = list(db.execute(f'PRAGMA table_info({quoted})'))
            print(json.dumps({'table': name, 'columns': columns, 'count': db.execute(f'SELECT COUNT(*) FROM {quoted}').fetchone()[0], 'examples': list(db.execute(f'SELECT * FROM {quoted} LIMIT 2'))}))
