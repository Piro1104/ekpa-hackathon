"""From forecast to suggested actions for the Account Manager.

Three kinds of suggestion (ACTION_TYPE):
  ΠΑΡΑΓΓΕΛΙΑ          items that run out before the customer's next usual visit
  ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ   the customer should have ordered again by now and did not
  ΠΡΟΣΦΟΡΑ            seasonal items the customer bought in the same window last year

Every suggestion carries an ERP payload shaped like data/sample_suggested_order.json,
plus ACTION_TYPE, REASON, SALES_NOTE and RAFI_SUGGESTION_ID (filled in by rafi.store).
Nothing here talks to the ERP: suggestions are only proposals until someone approves them.
"""
import math
from pathlib import Path

import pandas as pd

from rafi.features import (FACTOR_FLOOR, is_closed_on, likely_closer, next_working_day,
                           route_days, winter_history, working_day)
from rafi.forecast import SPLIT_CLOSER_FACTORS, _closure_status, _group_factors

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ORDER, LOSS, OFFER = "ΠΑΡΑΓΓΕΛΙΑ", "ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ", "ΠΡΟΣΦΟΡΑ"
URGENT, EXTRA_ROUTE, MARGINAL = "ΕΠΕΙΓΟΝ", "ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ", "ΟΡΙΑΚΟ"
CLOSED = "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ"

LOSS_GAP_FACTOR = 3          # silent for more than 3 x the usual order interval ...
LOSS_MIN_GAP_DAYS = 21       # ... and more than 21 days
CLOSER_EARLY_DAYS = 30       # a winter closer that stopped this much earlier than usual is a loss risk
BASKET_ORDERS = 5            # "usual basket": items of the last 5 orders ...
BASKET_MIN_ORDERS = 2        # ... that appear in at least 2 of them, with their median quantity
SEASONAL_CATEGORY = "Εποχικά"
SEASONAL_LOOKAHEAD_DAYS = 60  # offer seasonal items whose window opens within 60 days
# Assumed cost of one extra delivery run (EUR). A setting: change it from the app's settings.
EXTRA_ROUTE_COST_EUR = 60

WEEKDAYS = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]
MONTHS = ["Ιανουάριο", "Φεβρουάριο", "Μάρτιο", "Απρίλιο", "Μάιο", "Ιούνιο", "Ιούλιο",
          "Αύγουστο", "Σεπτέμβριο", "Οκτώβριο", "Νοέμβριο", "Δεκέμβριο"]
UNITS = {"ΚΙΒ": "κιβ.", "ΤΕΜ": "τεμ."}


def load_promos(data_dir=DATA_DIR):
    promos = pd.read_csv(Path(data_dir) / "PROMO.csv", encoding="utf-8-sig")
    promos["FROMDATE"] = pd.to_datetime(promos["FROMDATE"])
    promos["TODATE"] = pd.to_datetime(promos["TODATE"])
    return promos


# --- dates -----------------------------------------------------------------

def next_route_day(start, days):
    """First working day on or after `start` that is one of the customer's route weekdays."""
    d = pd.Timestamp(start).normalize()
    for _ in range(60):
        if working_day(d) and (not days or d.weekday() in days):
            return d
        d += pd.Timedelta(days=1)
    raise ValueError(f"no route day found after {start} for weekdays {days}")


def last_working_before(date):
    d = pd.Timestamp(date).normalize() - pd.Timedelta(days=1)
    while not working_day(d):
        d -= pd.Timedelta(days=1)
    return d


def route_visits(days, today):
    """(F, D1, D2): first working day after today, first route day >= F, next route day after D1."""
    first = next_working_day(today)
    d1 = next_route_day(first, days)
    d2 = next_route_day(d1 + pd.Timedelta(days=1), days)
    return first, d1, d2


def lost_sales_eur(rows, d1):
    """Sales lost if we wait for D1: for every (rate per day, depletion, price) running out
    on or before D1, rate x days without stock until D1 x price."""
    total = 0.0
    for rate, depletion, price in rows:
        days = (d1 - pd.Timestamp(depletion)).days
        if rate and days > 0:
            total += rate * days * price
    return round(total, 2)


def delivery_plan(earliest_depletion, days, today, lost_eur=math.inf):
    """Delivery date for an order, following the rules in order:

    F  = first working day after today; D1 = first route day >= F; D2 = next route day after D1
    - earliest depletion on or after D1, or nothing lost by waiting
                                         -> deliver on D1 (usual route, morning delivery)
    - lost sales until D1 > 0 but < cost of an extra route -> still D1, marked ΟΡΙΑΚΟ
    - else last working day before it    -> extra route if it is not a route weekday
    - if that is before F                -> deliver on F, urgent
    Returns (delivery, D1, D2, flags).
    """
    first, d1, d2 = route_visits(days, today)
    flags = []
    if earliest_depletion >= d1 or lost_eur <= 0:
        delivery = d1
    elif lost_eur < EXTRA_ROUTE_COST_EUR:
        delivery = d1
        flags.append(MARGINAL)
    else:
        delivery = last_working_before(earliest_depletion)
        if delivery < first:
            delivery = first
            flags.append(URGENT)
        if delivery.weekday() not in days:
            flags.append(EXTRA_ROUTE)
    return delivery, d1, d2, flags


def _next_occurrence(month_day, today):
    """First date after `today` with this 'MM-DD'."""
    d = pd.Timestamp(f"{today.year}-{month_day}")
    return d if d > today else pd.Timestamp(f"{today.year + 1}-{month_day}")


def _same_day_last_occurrence(month_day, today):
    """Most recent date on or before `today` with this 'MM-DD'."""
    d = pd.Timestamp(f"{today.year}-{month_day}")
    return d if d <= today else pd.Timestamp(f"{today.year - 1}-{month_day}")


# --- text helpers ----------------------------------------------------------

def _dm(d):
    d = pd.Timestamp(d)
    return f"{d.day}/{d.month}"


def _day(d):
    d = pd.Timestamp(d)
    return f"{WEEKDAYS[d.weekday()]} {_dm(d)}"


def _day_year(d):
    d = pd.Timestamp(d)
    return f"{_day(d)}/{d.year}"


def _num(x):
    s = f"{x:.2f}" if x < 0.1 else f"{x:.1f}"
    return s.replace(".", ",")


def _eur(x):
    return f"{x:,.0f}".replace(",", ".")


def _route_note(flags):
    if MARGINAL in flags:
        return "κανονικό δρομολόγιο"
    if URGENT in flags and EXTRA_ROUTE in flags:
        return "επείγον, έκτακτο δρομολόγιο"
    if URGENT in flags:
        return "επείγον"
    if EXTRA_ROUTE in flags:
        return "έκτακτο δρομολόγιο"
    return "κανονικό δρομολόγιο"


def _qty_list(lines):
    """'4 κιβ. Νερό ..., 2 τεμ. ...' without a trailing dot (the sentence adds its own)."""
    return ", ".join(f"{l['qty']} {l['unit']} {l['name']}" for l in lines).rstrip(".")


# --- texts (REASON / SALES_NOTE) --------------------------------------------

def template_texts(facts):
    """REASON and SALES_NOTE from fixed templates, using only the numbers in `facts`."""
    kind, lines = facts["action_type"], facts["lines"]
    delivery = facts["delivery_date"]
    hello = "Καλημέρα σας, από τη DEMO ΔΙΑΝΟΜΕΣ."

    if kind == ORDER:
        f = lines[0]
        if f["reason_code"] == "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ":
            head = f"Το «{f['name']}» έχει πιθανότατα ήδη τελειώσει"
        else:
            head = f"Το «{f['name']}» τελειώνει περίπου {_dm(f['depletion'])}"
        parts = [head]
        if f.get("rate_per_day"):
            parts.append(f"ο πελάτης καταναλώνει ~{_num(f['rate_per_day'])} {f['unit']}/ημέρα "
                         f"τον {facts['month']}")
        if len(lines) > 1:
            parts.append(f"άλλα {len(lines) - 1} είδη τελειώνουν πριν την επόμενη κανονική "
                         f"επίσκεψη ({_day(facts['next_visit'])})")
        lost = _eur(facts["lost_eur"])
        cost = _eur(facts["extra_route_cost_eur"])
        if MARGINAL in facts["flags"]:
            parts.append(f"εκτιμώμενες χαμένες πωλήσεις ~{lost} € μέχρι την κανονική επίσκεψη, "
                         f"λιγότερες από το κόστος έκτακτου δρομολογίου (~{cost} €)")
        elif URGENT in facts["flags"] or EXTRA_ROUTE in facts["flags"]:
            parts.append(f"εκτιμώμενες χαμένες πωλήσεις ~{lost} € αν περιμέναμε την κανονική "
                         f"επίσκεψη ({_day(facts['usual_visit'])}), περισσότερες από το κόστος "
                         f"έκτακτου δρομολογίου (~{cost} €)")
        parts.append(f"παράδοση {_day(delivery)} ({_route_note(facts['flags'])})")
        reason = "· ".join(parts) + "."
        promo = [l for l in lines if l.get("promo_disc")]
        if promo:
            reason += " Σε προσφορά την ημέρα παράδοσης: " + ", ".join(
                f"«{l['name']}» −{l['promo_disc']}%" for l in promo) + "."
        when = ("έχει μάλλον ήδη τελειώσει" if f["reason_code"] == "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ"
                else f"μάλλον τελειώνει γύρω στις {_dm(f['depletion'])}")
        note = f"{hello} Από τις παραγγελίες σας βλέπουμε ότι το «{f['name']}» {when}. "
        if MARGINAL in facts["flags"]:
            note += (f"Η κανονική μας επίσκεψη είναι {_day(delivery)}. Μπορείτε να μας "
                     f"επιβεβαιώσετε ότι σας φτάνει το απόθεμα μέχρι τότε; Εκείνη τη μέρα σας "
                     f"προτείνουμε: {_qty_list(lines)}.")
        else:
            note += f"Σας προτείνουμε παράδοση {_day(delivery)} με: {_qty_list(lines)}."
        if promo:
            note += " " + " ".join(f"Το «{l['name']}» έχει έκπτωση {l['promo_disc']}% εκείνη τη μέρα."
                                   for l in promo)
        note += " Να την ετοιμάσουμε;"
        return {"REASON": reason, "SALES_NOTE": note}

    if kind == LOSS:
        reason = (f"Ο πελάτης παρήγγελνε κάθε ~{facts['median_interval']:.0f} ημέρες, αλλά η "
                  f"τελευταία παράδοση ήταν {_dm(facts['last_delivery'])} "
                  f"(πριν {facts['days_silent']} ημέρες).")
        if facts.get("usual_close"):
            reason += (f" Συνήθως κλείνει για τον χειμώνα γύρω στις {_dm(facts['usual_close'])}, "
                       f"άρα σταμάτησε {facts['days_early']} ημέρες νωρίτερα από το συνηθισμένο.")
        reason += f" Αγορές τελευταίων 12 μηνών: {_eur(facts['purchases_12m_eur'])} €."
        if facts.get("closed_now"):
            reopen = facts["usual_reopen"]
            reason += (f" Το κατάστημα είναι τώρα κλειστό για τη σεζόν. Προτείνουμε τηλεφώνημα "
                       f"τώρα για να μάθουμε γιατί σταμάτησε νωρίτερα, και παραγγελία "
                       f"επαναλειτουργίας για {_day_year(delivery)} (συνήθως ανοίγει γύρω στις "
                       f"{_dm(reopen)}) με το συνηθισμένο καλάθι του ({len(lines)} είδη, "
                       f"ποσότητες για τον {MONTHS[pd.Timestamp(delivery).month - 1]}).")
            note = (f"{hello} Θέλαμε να μάθουμε πώς πήγε η σεζόν και αν είναι όλα εντάξει· "
                    f"είδαμε ότι φέτος οι παραγγελίες σταμάτησαν από τις "
                    f"{_dm(facts['last_delivery'])}. Για την επαναλειτουργία σας, να "
                    f"κλειδώσουμε από τώρα μια παραγγελία για {_day_year(delivery)} με: "
                    f"{_qty_list(lines)}; Αν θέλετε αλλαγές, τις κάνουμε μαζί.")
            return {"REASON": reason, "SALES_NOTE": note}
        reason += (f" Προτείνουμε επικοινωνία και το συνηθισμένο καλάθι του "
                   f"({len(lines)} είδη, ποσότητες προσαρμοσμένες στην εποχή) "
                   f"για {_day(delivery)}.")
        note = (f"{hello} Έχουμε να σας εξυπηρετήσουμε από τις {_dm(facts['last_delivery'])} "
                f"και θέλαμε να δούμε αν όλα πάνε καλά ή αν χρειάζεστε κάτι διαφορετικό. "
                f"Μπορούμε να σας φέρουμε {_day(delivery)} τα συνηθισμένα σας: "
                f"{_qty_list(lines)}. Τι λέτε;")
        return {"REASON": reason, "SALES_NOTE": note}

    # OFFER (seasonal items)
    season = ", ".join(f"{l['last_year_total']} {l['unit']} «{l['name']}»" for l in lines)
    first = ", ".join(f"{l['qty']} {l['unit']} «{l['name']}»" for l in lines)
    w = lines[0]
    reason = (f"Πέρσι ο πελάτης πήρε συνολικά {season} στην περίοδο "
              f"{_dm(w['window_start'])}–{_dm(w['window_end'])}. Τα εποχικά πωλούνται ξανά "
              f"από {_dm(w['window_start'])}· επειδή είναι ευπαθή, προτείνουμε όσο η περσινή "
              f"πρώτη παράδοση ({first}) με παράδοση {_day(delivery)} "
              f"({_route_note(facts['flags'])}).")
    note = (f"{hello} Ξεκινά η περίοδος για τα εποχικά. Για αρχή σας προτείνουμε {first}, "
            f"όσο και στην πρώτη παραγγελία σας πέρσι, με παράδοση {_day(delivery)}. "
            f"Να το ετοιμάσουμε;")
    return {"REASON": reason, "SALES_NOTE": note}


def write_texts(facts, text_writer=None):
    """Hook for the LLM (rafi.llm, later): `text_writer(facts)` returns REASON / SALES_NOTE.

    If there is no writer, or it fails or returns nothing, the template texts stay.
    TEXT_SOURCE says where the texts came from, so the UI can mark AI-written text.
    """
    texts = template_texts(facts)
    texts["TEXT_SOURCE"] = "ΠΡΟΤΥΠΟ"
    if text_writer is not None:
        try:
            written = text_writer(facts) or {}
        except Exception:  # an LLM problem must never block the suggestions
            written = {}
        for key in ("REASON", "SALES_NOTE"):
            if written.get(key):
                texts[key] = written[key]
                texts["TEXT_SOURCE"] = "AI"
    return texts


# --- context ---------------------------------------------------------------

class _Context:
    """Look-ups shared by the three kinds of suggestion."""

    def __init__(self, details, customers, items, deliveries, promos, today):
        self.today = pd.Timestamp(today).normalize()
        self.cutoff = self.today - pd.Timedelta(days=1)
        self.first_day = next_working_day(self.today)
        self.past = deliveries[deliveries["TRNDATE"] <= self.cutoff]
        self.customers = customers.set_index("TRDR")
        self.items = items.set_index("MTRL")
        self.promos = promos
        self.routes = route_days(self.past, self.cutoff)
        self.history = winter_history(self.past, customers, self.cutoff)

        d = details.copy()
        d["PREDICTED_DEPLETION_DATE"] = pd.to_datetime(d["PREDICTED_DEPLETION_DATE"])
        self.details = d
        self.closed = set(d.loc[d["REASON_CODE"] == CLOSED, "TRDR"])

        # seasonal factors exactly as the forecast uses them (closer / open groups)
        self.closure = _closure_status(customers, self.history, self.cutoff)
        closers = list(self.history.loc[self.history["IS_CLOSER"], "TRDR"])
        f = _group_factors(self.past, customers, items, self.cutoff, closers)
        self.factors = {(g, a, c, m): max(v, FACTOR_FLOOR) for g, a, c, m, v in
                        zip(f["GROUP"], f["AREA"], f["MTRCATEGORY"], f["MONTH"], f["FACTOR"])}

        year_ago = self.cutoff - pd.Timedelta(days=365)
        last12 = self.past[self.past["TRNDATE"] > year_ago]
        self.value_12m = last12.groupby("TRDR")["LINEVAL"].sum().round(2).to_dict()

    def route(self, trdr):
        return self.routes.get(int(trdr), [])

    def closure_dates(self, trdr):
        """Typical (last delivery before closing, first after re-opening) or (None, None)."""
        h = self.history[self.history["TRDR"] == trdr]
        if len(h):
            r = h.iloc[0]
            return (r["LAST_BEFORE_CLOSE"], r["FIRST_AFTER_OPEN"]) if r["IS_CLOSER"] else (None, None)
        likely, last, first = likely_closer(trdr, self.customers.reset_index(), self.history)
        return (last, first) if likely else (None, None)

    def factor(self, trdr, mtrl, month):
        """Seasonal factor of the forecast for this customer's group, area and item category."""
        group = ("closer" if trdr in self.closure else "open") if SPLIT_CLOSER_FACTORS else "all"
        area, cat = self.customers.at[trdr, "AREA"], self.items.at[mtrl, "MTRCATEGORY"]
        return self.factors.get((group, area, cat, month),
                                self.factors.get(("all", area, cat, month), 1.0))

    def promo(self, mtrl, date):
        """Best active discount (%) for the item on that date, or None."""
        p = self.promos
        cat = self.items.at[mtrl, "MTRCATEGORY"]
        hit = p[((p["MTRL"] == mtrl) | (p["MTRCATEGORY"] == cat))
                & (p["FROMDATE"] <= date) & (p["TODATE"] >= date)]
        return int(hit["DISC_PRC"].max()) if len(hit) else None

    def line(self, mtrl, qty, depletion=None, conf=None):
        it = self.items.loc[mtrl]
        return {"MTRL": int(mtrl), "ITEM_CODE": str(it["CODE"]), "ITEM_NAME": str(it["NAME"]),
                "QTY1": int(qty), "PRICER": float(it["PRICER"]),
                "PREDICTED_DEPLETION_DATE": (pd.Timestamp(depletion).strftime("%Y-%m-%d")
                                             if depletion is not None and pd.notna(depletion)
                                             else None),
                "CONFIDENCE": None if conf is None else round(float(conf), 2)}

    def suggestion(self, kind, trdr, delivery, lines, flags, facts, earliest, text_writer,
                   extra=None):
        c = self.customers.loc[trdr]
        facts = {"action_type": kind, "trdr": int(trdr), "customer_name": str(c["NAME"]),
                 "area": str(c["AREA"]), "delivery_date": delivery.strftime("%Y-%m-%d"),
                 "flags": list(flags), "month": MONTHS[self.today.month - 1], **facts}
        texts = write_texts(facts, text_writer)
        payload = {
            "TRDR": int(trdr), "CUSTOMER_CODE": str(c["CODE"]), "SALESMAN": int(c["SALESMAN"]),
            "SUGGESTED_DELIVERY_DATE": delivery.strftime("%Y-%m-%d"),
            "ACTION_TYPE": kind, "FLAGS": list(flags),
            "REASON": texts["REASON"], "SALES_NOTE": texts["SALES_NOTE"],
            "TEXT_SOURCE": texts["TEXT_SOURCE"],
            "RAFI_SUGGESTION_ID": None,  # set by rafi.store when saved
            **(extra or {}),
            "LINES": lines,
        }
        value = round(sum(l["QTY1"] * l["PRICER"] for l in lines), 2)
        return {"ACTION_TYPE": kind, "TRDR": int(trdr), "SALESMAN": int(c["SALESMAN"]),
                "CUSTOMER_NAME": str(c["NAME"]), "DELIVERY_DATE": payload["SUGGESTED_DELIVERY_DATE"],
                "FLAGS": list(flags), "EARLIEST_DEPLETION": earliest, "VALUE_EUR": value,
                "PRIORITY": None, "FACTS": facts, "payload": payload}


# --- (a) orders --------------------------------------------------------------

def order_suggestions(ctx, skip, text_writer=None):
    d = ctx.details
    d = d[(d["REASON_CODE"] != CLOSED) & d["PREDICTED_DEPLETION_DATE"].notna()
          & d["SUGGESTED_QTY"].notna() & ~d["TRDR"].isin(skip)]
    out = []
    for trdr, g in d.groupby("TRDR"):
        days = ctx.route(trdr)
        earliest = g["PREDICTED_DEPLETION_DATE"].min()
        _, d1, _ = route_visits(days, ctx.today)
        early = g[g["PREDICTED_DEPLETION_DATE"] <= d1]
        lost = lost_sales_eur(
            [((r.BASE_RATE * r.SEASONAL_FACTOR_NOW) if pd.notna(r.BASE_RATE) else 0.0,
              r.PREDICTED_DEPLETION_DATE, float(ctx.items.at[r.MTRL, "PRICER"]))
             for r in early.itertuples(index=False)], d1)
        delivery, d1, d2, flags = delivery_plan(earliest, days, ctx.today, lost)
        g = g[g["PREDICTED_DEPLETION_DATE"] < d2].sort_values(["PREDICTED_DEPLETION_DATE", "MTRL"])
        if g.empty:
            continue
        lines, facts_lines = [], []
        for r in g.itertuples(index=False):
            line = ctx.line(r.MTRL, max(1, int(r.SUGGESTED_QTY)), r.PREDICTED_DEPLETION_DATE,
                            r.CONFIDENCE)
            disc = ctx.promo(r.MTRL, delivery)
            if disc:
                line["PROMO_DISC_PRC"] = disc
            lines.append(line)
            rate = (r.BASE_RATE * r.SEASONAL_FACTOR_NOW) if pd.notna(r.BASE_RATE) else None
            facts_lines.append({
                "name": line["ITEM_NAME"], "unit": UNITS.get(ctx.items.at[r.MTRL, "MTRUNIT1"], ""),
                "qty": line["QTY1"], "depletion": line["PREDICTED_DEPLETION_DATE"],
                "reason_code": r.REASON_CODE, "rate_per_day": round(rate, 3) if rate else None,
                "stock_at_cutoff": None if pd.isna(r.STOCK_AT_CUTOFF) else float(r.STOCK_AT_CUTOFF),
                "promo_disc": disc})
        facts = {"next_visit": d2.strftime("%Y-%m-%d"), "usual_visit": d1.strftime("%Y-%m-%d"),
                 "lost_eur": lost, "extra_route_cost_eur": EXTRA_ROUTE_COST_EUR,
                 "lines": facts_lines}
        out.append(ctx.suggestion(ORDER, trdr, delivery, lines, flags, facts,
                                  lines[0]["PREDICTED_DEPLETION_DATE"], text_writer,
                                  extra={"LOST_EUR": lost}))
    return out


# --- (b) loss risk ---------------------------------------------------------

def loss_risks(ctx):
    """Customers silent for too long, with the numbers that show it.

    Silent: days since the last delivery > 3 x the median interval between orders, and > 21.
    Not a loss risk if a winter closure explains it (closed now, and the last delivery came
    no more than 30 days before the usual closing day).
    """
    p = ctx.past[(ctx.past["QTY"] > 0) & ~ctx.past["IS_CREDIT"]]
    out = []
    for trdr, s in p.groupby("TRDR")["TRNDATE"]:
        dates = pd.Series(sorted(s.unique()))
        if len(dates) < 2:
            continue
        last = dates.iloc[-1]
        median = float(dates.diff().dt.days.median())
        silent = (ctx.today - last).days
        if silent <= LOSS_GAP_FACTOR * median or silent <= LOSS_MIN_GAP_DAYS:
            continue
        info = {"TRDR": int(trdr), "last_delivery": last.strftime("%Y-%m-%d"),
                "median_interval": median, "days_silent": silent,
                "usual_close": None, "days_early": None, "closed_now": False,
                "usual_reopen": None}
        last_close, first_open = ctx.closure_dates(trdr)
        if last_close and is_closed_on(ctx.today, last_close, first_open):
            usual = _same_day_last_occurrence(last_close, ctx.today)
            early = (usual - last).days
            if early <= CLOSER_EARLY_DAYS:
                continue  # the usual winter closure
            # closed for the season now: the order is for the re-opening (default 1 April)
            reopen = _next_occurrence(first_open or "04-01", ctx.today)
            info.update(usual_close=usual.strftime("%Y-%m-%d"), days_early=early,
                        closed_now=True, usual_reopen=reopen.strftime("%Y-%m-%d"))
        out.append(info)
    return out


def usual_basket(ctx, trdr, delivery):
    """Items of the customer's last orders: [(MTRL, qty, share, median qty, season scale)].

    qty = median quantity x (seasonal factor of the delivery month
                             / mean factor of the months of the orders it came from),
    rounded up, at least 1. Factors are the forecast's (same closer / open group).
    """
    p = ctx.past[(ctx.past["TRDR"] == trdr) & (ctx.past["QTY"] > 0) & ~ctx.past["IS_CREDIT"]]
    seasonal = ctx.items.index[ctx.items["MTRCATEGORY"] == SEASONAL_CATEGORY]
    p = p[~p["MTRL"].isin(seasonal)]
    last_dates = sorted(p["TRNDATE"].unique())[-BASKET_ORDERS:]
    recent = p[p["TRNDATE"].isin(last_dates)]
    per_order = recent.groupby(["MTRL", "TRNDATE"])["QTY"].sum()
    out = []
    for mtrl, s in per_order.groupby(level=0):
        if len(s) < min(BASKET_MIN_ORDERS, len(last_dates)):
            continue
        months = [pd.Timestamp(t).month for t in s.index.get_level_values("TRNDATE")]
        then = sum(ctx.factor(trdr, mtrl, m) for m in months) / len(months)
        scale = ctx.factor(trdr, mtrl, delivery.month) / then
        median = float(s.median())
        qty = max(1, math.ceil(median * scale - 1e-9))
        out.append((int(mtrl), qty, len(s) / len(last_dates), median, round(scale, 3)))
    return out


def loss_suggestions(ctx, risks, text_writer=None):
    out = []
    for info in risks:
        trdr = info["TRDR"]
        start = pd.Timestamp(info["usual_reopen"]) if info["closed_now"] else ctx.first_day
        delivery = next_route_day(start, ctx.route(trdr))
        basket = usual_basket(ctx, trdr, delivery)
        if not basket:
            continue
        lines = [ctx.line(m, q, None, share) for m, q, share, _, _ in basket]
        facts = {**{k: v for k, v in info.items() if k != "TRDR"},
                 "purchases_12m_eur": float(ctx.value_12m.get(trdr, 0.0)),
                 "lines": [{"name": l["ITEM_NAME"], "qty": l["QTY1"], "median_qty": med,
                            "season_scale": scale,
                            "unit": UNITS.get(ctx.items.at[l["MTRL"], "MTRUNIT1"], "")}
                           for l, (_, _, _, med, scale) in zip(lines, basket)]}
        flags = [] if delivery.weekday() in ctx.route(trdr) else [EXTRA_ROUTE]
        out.append(ctx.suggestion(LOSS, trdr, delivery, lines, flags, facts, None, text_writer))
    return out


# --- (c) seasonal offers ---------------------------------------------------

def seasonal_windows(ctx):
    """{MTRL: (start, end, last year's start, last year's end)} for seasonal items opening soon.

    The window is last year's first and last sale date of the item, moved to this year.
    """
    seasonal = ctx.items.index[ctx.items["MTRCATEGORY"] == SEASONAL_CATEGORY]
    p = ctx.past[ctx.past["MTRL"].isin(seasonal) & (ctx.past["QTY"] > 0)]
    last_year = p[p["TRNDATE"].dt.year == ctx.today.year - 1]
    out = {}
    for mtrl, s in last_year.groupby("MTRL")["TRNDATE"]:
        start = s.min() + pd.DateOffset(years=1)
        end = s.max() + pd.DateOffset(years=1)
        if end >= ctx.first_day and start <= ctx.today + pd.Timedelta(days=SEASONAL_LOOKAHEAD_DAYS):
            out[int(mtrl)] = (start, end, s.min(), s.max())
    return out


def offer_suggestions(ctx, skip, text_writer=None):
    windows = seasonal_windows(ctx)
    p = ctx.past[ctx.past["QTY"] > 0]
    years = p["TRNDATE"].dt.year
    by_customer = {}
    for mtrl, (start, end, ly_start, ly_end) in windows.items():
        item_sales = p[(p["MTRL"] == mtrl)]
        sold_years = sorted(item_sales["TRNDATE"].dt.year.unique())
        sold_years = [y for y in sold_years if y < ctx.today.year]
        # net quantity in last year's window: returns (credits) are taken off
        net = ctx.past[ctx.past["MTRL"] == mtrl]
        ly = net[(net["TRNDATE"] >= ly_start) & (net["TRNDATE"] <= ly_end)]
        # last year's first delivery of the item to each customer inside the window
        bought = ly[ly["QTY"] > 0].groupby(["TRDR", "TRNDATE"])["QTY"].sum()
        first_qty = bought.groupby(level=0).first()
        for trdr, total in ly.groupby("TRDR")["QTY"].sum().items():
            if trdr in skip or trdr in ctx.closed or total <= 0 or trdr not in first_qty:
                continue
            delivery = next_route_day(max(start, ctx.first_day), ctx.route(trdr))
            if delivery > end:
                continue
            bought_years = set(years[(p["MTRL"] == mtrl) & (p["TRDR"] == trdr)])
            conf = len(bought_years & set(sold_years)) / max(1, len(sold_years))
            by_customer.setdefault((trdr, delivery), []).append(
                (mtrl, int(first_qty[trdr]), conf, start, end, int(total)))

    out = []
    for (trdr, delivery), entries in sorted(by_customer.items()):
        lines = [ctx.line(m, q, None, conf) for m, q, conf, _, _, _ in entries]
        facts = {"lines": [{"name": str(ctx.items.at[m, "NAME"]), "qty": q, "last_year_total": t,
                            "unit": UNITS.get(ctx.items.at[m, "MTRUNIT1"], ""),
                            "window_start": s.strftime("%Y-%m-%d"),
                            "window_end": e.strftime("%Y-%m-%d")}
                           for m, q, _, s, e, t in entries]}
        flags = [] if delivery.weekday() in ctx.route(trdr) else [EXTRA_ROUTE]
        out.append(ctx.suggestion(OFFER, trdr, delivery, lines, flags, facts, None, text_writer))
    return out


# --- all together ----------------------------------------------------------

def _rank(suggestions):
    """PRIORITY 1, 2, ...:
    1. urgent orders (earliest depletion, then value)
    2. loss risks (customer's purchases of the last 12 months, highest first)
    3. other orders (earliest depletion, then value)
    4. offers (value)
    """
    def key(s):
        dep = pd.Timestamp(s["EARLIEST_DEPLETION"]) if s["EARLIEST_DEPLETION"] else pd.Timestamp.max
        if URGENT in s["FLAGS"]:
            return (0, dep, -s["VALUE_EUR"], s["TRDR"])
        if s["ACTION_TYPE"] == LOSS:
            return (1, -s["FACTS"]["purchases_12m_eur"], s["TRDR"])
        if s["ACTION_TYPE"] == ORDER:
            return (2, dep, -s["VALUE_EUR"], s["TRDR"])
        return (3, -s["VALUE_EUR"], s["TRDR"])
    ordered = sorted(suggestions, key=key)
    for i, s in enumerate(ordered, start=1):
        s["PRIORITY"] = i
    return ordered


def build_suggestions(details, customers, items, deliveries, promos, today, text_writer=None):
    """All suggestions for `today`, sorted by PRIORITY.

    details:    output of rafi.forecast.forecast (cutoff = today - 1)
    deliveries: clean deliveries (rafi.ingest.build_deliveries)
    promos:     PROMO.csv (load_promos)
    text_writer: optional function facts -> {"REASON", "SALES_NOTE"} (the LLM hook)

    A customer at loss risk gets only the loss-risk suggestion (no order, no offer).
    """
    ctx = _Context(details, customers, items, deliveries, promos, today)
    risks = loss_risks(ctx)
    at_risk = {r["TRDR"] for r in risks}
    out = (order_suggestions(ctx, at_risk, text_writer)
           + loss_suggestions(ctx, risks, text_writer)
           + offer_suggestions(ctx, at_risk, text_writer))
    return _rank(out)
