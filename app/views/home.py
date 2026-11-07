import streamlit as st

from app import helpers as h
from app import ui
from app import warehouse as wh

TOP = 3  # rows in each "Χρειάζονται εσένα σήμερα" list

suggestions = ui.all_suggestions()
latest = max((s.get("created_at") or "" for s in suggestions), default="")
synced = f" · συγχρονισμός {h.fmt_sync(latest)}" if latest else ""

head, chip = st.columns([1, 1], vertical_alignment="center")
head.markdown(f'<div class="ne-title" style="font-size:2.1rem">Καλημέρα</div>'
              f'<p class="ne-sub">{h.fmt_date_words(h.SCENARIO_TODAY)} · '
              f'δεδομένα ERP έως 31/10</p>', unsafe_allow_html=True)
chip.markdown(f'<div style="text-align:right"><span class="ne-chip"><span class="dot"></span>'
              f'<b>DEMO ΔΙΑΝΟΜΕΣ Α.Ε.</b> · SoftOne ERP{ui.esc(synced)}</span></div>',
              unsafe_allow_html=True)

if ui.empty_state(suggestions):
    st.stop()

m = h.home_metrics(suggestions)
tabs = h.split_tabs(suggestions)
facts = ui.facts()
route_cost = h.read_settings().get("extra_route_cost_eur", 60)
orders = ui.warehouse_orders()
morning = h.morning_texts()


def morning_line(key):
    if morning.get(key):
        st.markdown(f'<p class="ne-muted">{ui.esc(morning[key])}</p>', unsafe_allow_html=True)


def open_link(col, key, on_click, args):
    with col:
        st.button("Άνοιγμα", key=f"ne_open_{key}", type="tertiary", on_click=on_click, args=args)


def row(key, tag_html, sentence, on_click, args):
    with st.container(key=f"ne_row_{key}"):
        text, link = st.columns([6, 1.1])
        text.markdown(f'<div class="ne-row">{tag_html}<div>{sentence}</div></div>',
                      unsafe_allow_html=True)
        open_link(link, key, on_click, args)


def _block_head(tag_html, title, more_label, page):
    left, right = st.columns([2.6, 1.6], vertical_alignment="center")
    left.markdown(f'<div class="ne-block-head">{tag_html}<span class="t">{title}</span></div>',
                  unsafe_allow_html=True)
    with right:
        st.page_link(page, label=more_label)


left, right = st.columns([1.35, 1], gap="medium")

with left, st.container(key="ne_block_customers"):
    _block_head(ui.tag("Πελάτες"), "Τι προτείνουμε στους πελάτες",
                f"Όλες οι {m['pending']} ειδοποιήσεις →", "views/customers.py")
    morning_line("customers")
    ui.tiles([
        ("Προς έγκριση", h.fmt_num(m["pending"]), "", ""),
        ("Έκτακτα δρομολόγια", h.fmt_num(m["extra_routes"]),
         f"{sum(1 for s in tabs[h.TAB_NEEDS_YOU] if h.URGENT in s['payload'].get('FLAGS', []))} "
         f"επείγον", "warn"),
        ("Κίνδυνος απώλειας", h.fmt_num(m["loss_risks"]), "", "loss"),
        ("Οριακά · τηλεφώνημα",
         h.fmt_num(sum(1 for s in suggestions if s["status"] == h.PROPOSED
                       and h.MARGINAL in s["payload"].get("FLAGS", []))), "", ""),
        ("Αξία σε κίνδυνο", h.fmt_eur(m["at_risk_eur"]), "", ""),
        ("Προσφορές", h.fmt_num(m["offers"]), h.fmt_eur(m["offers_eur"]), ""),
    ])
    st.markdown('<div class="ne-h">Χρειάζονται εσένα σήμερα</div>', unsafe_allow_html=True)
    urgent = tabs[h.TAB_NEEDS_YOU]
    if not urgent:
        st.markdown('<p class="ne-muted">Τίποτα επείγον σήμερα.</p>', unsafe_allow_html=True)
    for s in urgent[:TOP]:
        name, _ = ui.customer(s["trdr"])
        text, kind = h.card_badge(s)
        row(f"s{s['id']}", ui.tag(text.split(" · ")[0], kind),
            h.home_sentence(s, name, s["trdr"], facts.get(s["id"]), route_cost),
            ui.open_suggestion, (s["id"],))
    rest = urgent[TOP:]
    if rest:
        names = ", ".join(f"{ui.customer(s['trdr'])[0]} ({s['trdr']})" for s in rest)
        st.markdown(f'<div class="ne-muted" style="border-top:1px solid #E3E8E5;padding-top:.7rem">'
                    f'Ακόμη {len(rest)}: {ui.esc(names)}.</div>', unsafe_allow_html=True)

with right, st.container(key="ne_block_warehouse"):
    _block_head(ui.tag("Αποθήκη", "wh"), "Τι πρέπει να αγοράσουμε", "Αναλυτικά →",
                "views/warehouse_page.py")
    morning_line("warehouse")
    if orders is None:
        st.markdown(f'<p class="ne-muted">{wh.SOON}</p>', unsafe_allow_html=True)
    else:
        w = wh.metrics(orders)
        ui.tiles([
            ("Παραγγελίες σε προμηθευτές", h.fmt_num(w["orders"]), "αυτή την εβδομάδα", ""),
            ("Δεν φτάνει η αποθήκη", h.fmt_num(w["short_items"]),
             f"{'είδος' if w['short_items'] == 1 else 'είδη'} · "
             f"{w['short_deliveries']} παραδόσεις", "warn" if w["short_items"] else ""),
        ], cols=2)
        st.markdown('<div class="ne-h">Χρειάζονται εσένα σήμερα</div>', unsafe_allow_html=True)
        if not orders:
            st.markdown('<p class="ne-muted">Καμία παραγγελία σε προμηθευτή αυτή την εβδομάδα.</p>',
                        unsafe_allow_html=True)
        for o in wh.sort_orders(orders)[:TOP]:
            key = wh.order_key(o)
            t = ui.tag("Δεν φτάνει", "short") if o.get("short") else ui.tag("Αγορά", "buy")
            row(f"w{ui.safe_key(key)}", t, wh.home_sentence(o), ui.open_purchase, (key,))

st.write("")
with st.container(key="ne_block_chat"):
    st.markdown('<div class="ne-block-head"><span class="t">Ρώτα το Never Empty</span></div>',
                unsafe_allow_html=True)
    ui.chat_box("chat_general",
                lambda: {**h.general_context(suggestions, ui.customers_map()),
                         "warehouse_week": wh.week_context(orders)},
                placeholder="π.χ. Ποιοι πελάτες επηρεάζονται αν δεν έρθουν οι μπύρες;",
                quick=("Με ποιον να ξεκινήσω σήμερα;", "Τι παραγγέλνουμε αυτή την εβδομάδα;",
                       "Πλάνο Πέμπτης στη Θεσσαλονίκη"),
                footnote="Ο βοηθός απαντά μόνο με τα νούμερα της εφαρμογής.")
