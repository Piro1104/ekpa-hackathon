#!/usr/bin/env python3
"""Mock ERP server (standard library only). Usage: python3 mock_erp_server.py [--port 8080]"""
import argparse, csv, json, os, re, time, itertools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); DRAFTS = os.path.join(HERE, 'draft_orders')

def rd(name):
    with open(os.path.join(DATA, name), encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def num(v):
    try: return int(v)
    except ValueError:
        try: return float(v)
        except ValueError: return v

CUSTOMERS = rd('TRDR.csv'); ITEMS = rd('MTRL.csv'); SALESMEN = rd('PRSN.csv'); PROMOS = rd('PROMO.csv')
HEADERS = rd('SALDOC.csv'); LINES = {}
for r in rd('MTRLINES.csv'): LINES.setdefault(r['FINDOC'], []).append({k: num(v) for k, v in r.items() if k != 'FINDOC'})
HNUM = ('FINDOC','COMPANY','SOSOURCE','SERIES','TRDR','SALESMAN','ISCANCEL','NETAMNT','VATAMNT','SUMAMNT')
def header(r):
    h = {k: (num(v) if k in HNUM else v) for k, v in r.items()}; h['LINES'] = LINES.get(r['FINDOC'], []); return h
HEADERS.sort(key=lambda r: (r['TRNDATE'], int(r['FINDOC'])))
BY_FINDOC = {r['FINDOC']: r for r in HEADERS}
TRDR_IDS = {int(r['TRDR']) for r in CUSTOMERS}; MTRL_IDS = {int(r['MTRL']) for r in ITEMS}
_seq = itertools.count(1)

def conv(rows): return [{k: num(v) for k, v in r.items()} for r in rows]

def validate(o):
    errs = []
    if not isinstance(o, dict): return ['body must be a JSON object']
    for k in ('TRDR', 'SUGGESTED_DELIVERY_DATE', 'LINES'):
        if k not in o: errs.append(f'missing required key: {k}')
    if 'TRDR' in o and o['TRDR'] not in TRDR_IDS: errs.append(f"unknown TRDR: {o['TRDR']}")
    if 'SUGGESTED_DELIVERY_DATE' in o and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(o['SUGGESTED_DELIVERY_DATE'])): errs.append('SUGGESTED_DELIVERY_DATE must be YYYY-MM-DD')
    ls = o.get('LINES')
    if 'LINES' in o:
        if not isinstance(ls, list) or not ls: errs.append('LINES must be a non-empty list')
        else:
            for n, l in enumerate(ls):
                if not isinstance(l, dict): errs.append(f'LINES[{n}] must be an object'); continue
                for k in ('MTRL', 'QTY1'):
                    if k not in l: errs.append(f'LINES[{n}]: missing key {k}')
                if 'MTRL' in l and l['MTRL'] not in MTRL_IDS: errs.append(f"LINES[{n}]: unknown MTRL {l['MTRL']}")
                q = l.get('QTY1')
                if 'QTY1' in l and (isinstance(q, bool) or not isinstance(q, (int, float)) or q <= 0): errs.append(f'LINES[{n}]: QTY1 must be a number > 0')
    return errs

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query); p = u.path.rstrip('/')
        simple = {'/api/customers': CUSTOMERS, '/api/items': ITEMS, '/api/salesmen': SALESMEN, '/api/promotions': PROMOS}
        if p in simple: return self.send(200, conv(simple[p]))
        if p == '/api/saldoc':
            try:
                page = max(1, int(q.get('page', ['1'])[0])); size = min(1000, max(1, int(q.get('pageSize', ['500'])[0])))
            except ValueError: return self.send(400, {'errors': ['page/pageSize must be integers']})
            fr = q.get('from', [''])[0]; to = q.get('to', [''])[0]
            rows = [r for r in HEADERS if (not fr or r['TRNDATE'] >= fr) and (not to or r['TRNDATE'] <= to)]
            return self.send(200, {'page': page, 'pageSize': size, 'total': len(rows), 'items': [header(r) for r in rows[(page-1)*size: page*size]]})
        m = re.fullmatch(r'/api/saldoc/(\d+)', p)
        if m:
            r = BY_FINDOC.get(m.group(1))
            return self.send(200, header(r)) if r else self.send(404, {'errors': ['FINDOC not found']})
        self.send(404, {'errors': ['not found']})
    def do_POST(self):
        if urlparse(self.path).path.rstrip('/') != '/api/draft-orders': return self.send(404, {'errors': ['not found']})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode('utf-8'))
        except Exception: return self.send(400, {'errors': ['invalid JSON']})
        errs = validate(body)
        if errs: return self.send(400, {'errors': errs})
        os.makedirs(DRAFTS, exist_ok=True)
        oid = f"DRAFT-{time.strftime('%Y%m%d%H%M%S')}-{next(_seq):04d}"
        with open(os.path.join(DRAFTS, oid + '.json'), 'w', encoding='utf-8') as f: json.dump(body, f, ensure_ascii=False, indent=2)
        self.send(201, {'id': oid, 'status': 'DRAFT'})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8080); a = ap.parse_args()
    print(f'Mock ERP on http://localhost:{a.port}  (data: {DATA})')
    ThreadingHTTPServer(('0.0.0.0', a.port), H).serve_forever()
