import streamlit as st

from app import helpers as h
from app import ui
from app import warehouse as wh

ui.page_header("Αποθήκη", "wh", "Αποθήκη",
               "Παραγγελίες προς τους προμηθευτές μας για αυτή την εβδομάδα")

funcs = wh.agent()
orders = ui.warehouse_orders() if funcs else None
ui.show_flash("warehouse")
if orders is None:
    st.info(wh.SOON)
    st.stop()

state = st.session_state
orders = wh.sort_orders(orders)
everything = ui.warehouse_orders(wh.ALL_DAYS) or []
m = wh.metrics(orders)
ui.tiles([
    ("Παραγγελίες προς έγκριση", h.fmt_num(m["orders"]), "", ""),
    ("Ελλείψεις αποθήκης", h.fmt_num(m["short_items"]),
     f"{'είδος' if m['short_items'] == 1 else 'είδη'} · {m['short_deliveries']} "
     f"παραδόσεις σε πελάτες δεν καλύπτονται", "warn" if m["short_items"] else ""),
    ("Αργήσαμε", h.fmt_num(m["late_items"]),
     f"{'είδος' if m['late_items'] == 1 else 'είδη'} · πελάτες καλυμμένοι, κάτω από το "
     f"απόθεμα ασφαλείας", ""),
    ("Αξία σε τιμή καταλόγου", h.fmt_eur(m["value_eur"]), "", ""),
], cols=4)

keys = [wh.order_key(o) for o in orders]
if state.get("selected_po") not in keys:
    state["selected_po"] = keys[0] if keys else None
selected = next((o for o in orders if wh.order_key(o) == state["selected_po"]), None)


def pick(key):
    state["selected_po"] = key
    state.pop("po_reject", None)


left, right = st.columns([5, 7], gap="medium")

with left, st.container(key="ne_list"):
    if not orders:
        st.markdown('<p class="ne-muted">Καμία παραγγελία σε προμηθευτή αυτή την εβδομάδα.</p>',
                    unsafe_allow_html=True)
    for o in orders:
        key = wh.order_key(o)
        t = ui.tag(*wh.tag_of(o))
        ui.card(key, t, o.get("supplier_name") or o.get("supplier_id"), "", wh.card_line(o),
                key == state["selected_po"], pick, (key,))
    st.markdown(f'<p class="ne-muted">Μία παραγγελία ανά προμηθευτή και ημερομηνία. '
                f'Προγραμματισμένες για αργότερα: {wh.later_count(everything, orders)}.</p>',
                unsafe_allow_html=True)

with right:
    if selected is None:
        st.stop()
    o = selected
    key = wh.order_key(o)
    supplier = o.get("supplier_name") or o.get("supplier_id")
    ai_note = h.is_ai(o.get("note_source"))
    kind = wh.status(o)

    with st.container(key="ne_panel_warehouse"):
        tags = ui.tag("Αποθήκη", "wh") + (ui.tag(*wh.tag_of(o)) if kind else "")
        st.markdown(f'{tags}<div class="ne-title">{ui.esc(supplier)}</div>',
                    unsafe_allow_html=True)
        ui.numbers([("Παράγγειλε έως", h.fmt_day(o.get("order_by"))),
                    ("Φτάνει", h.fmt_day(o.get("arrival"))),
                    ("Αξία (τιμή καταλόγου)", h.fmt_eur(o.get("total_value_eur")))])

        ui.label_row("Τι προτείνουμε και γιατί")
        st.markdown(f'<div class="ne-reason">{ui.esc(o.get("reason") or "—")}</div>',
                    unsafe_allow_html=True)

        ui.label_row("Είδη")
        df = wh.lines_frame(o)
        ui.table(df, height=38 + 35 * max(1, len(df)))

        affected = o.get("affected_deliveries") or []
        if kind == wh.SHORT and affected:
            st.markdown(
                f'<div class="ne-warn"><b>{wh.affected_phrase(len(affected))} '
                f'πριν τις {h.dm(o.get("arrival"))}</b><br>{ui.esc(wh.affected_text(o))}<br>'
                f'<span class="small">Δεν καλύπτονται από το απόθεμα. Με ό,τι υπάρχει '
                f'εξυπηρετούνται πρώτα τα επείγοντα και οι μεγαλύτεροι πελάτες.</span></div>',
                unsafe_allow_html=True)
            with st.expander("Όλες οι παραδόσεις που δεν καλύπτονται"):
                for a in sorted(affected, key=lambda a: str(a.get("date"))):
                    st.markdown(f"{ui.esc(a.get('customer_name') or a.get('trdr'))} · "
                                f"{h.fmt_day(a.get('date'))} · {h.fmt_qty(a.get('qty'))}")
            st.write("")
        elif kind == wh.LATE:
            st.markdown(f'<div class="ne-reason"><b>Αργήσαμε.</b> {ui.esc(wh.late_text(o))}</div>',
                        unsafe_allow_html=True)
            st.write("")

        ui.label_row("Μήνυμα προς τον προμηθευτή", ai_note)
        st.markdown(f'<div class="ne-note blue">{ui.esc(o.get("note") or "—")}</div>',
                    unsafe_allow_html=True)
        st.write("")

        rejecting = state.get("po_reject") == key
        with st.container(horizontal=True):
            approve = st.button("Εγκρίνω την παραγγελία", type="primary", key=f"po_ok_{key}")
            if st.button("Απορρίπτω", key=f"ne_reject_po_{ui.safe_key(key)}"):
                state["po_reject"] = None if rejecting else key
                st.rerun()

        if approve:
            try:
                funcs["approve_purchase_order"](o["supplier_id"], o["order_by"], h.store_user())
                ui.clear_warehouse()
                done = f"Η παραγγελία προς {supplier} εγκρίθηκε."
                note = h.notify_supplier_after_approval(
                    o["supplier_id"], o["order_by"], h.read_settings().get("after_approval_mode"))
                kind = note[0] if note else "success"
                ui.flash("warehouse", kind, done + (f" {note[1]}" if note else ""))
                st.rerun()
            except Exception as e:
                st.error(h.humanize(e))

        if rejecting:
            with st.form(f"po_reject_form_{ui.safe_key(key)}", border=True):
                reason = st.text_input("Λόγος απόρριψης (υποχρεωτικός)")
                do_reject = st.form_submit_button("Απόρριψη παραγγελίας")
            if do_reject:
                if not reason.strip():
                    st.error("Γράψε λόγο απόρριψης.")
                else:
                    try:
                        funcs["reject_purchase_order"](o["supplier_id"], o["order_by"],
                                                       reason.strip(), h.store_user())
                        ui.clear_warehouse()
                        state.pop("po_reject", None)
                        ui.flash("warehouse", "info",
                                 f"Η παραγγελία προς {supplier} απορρίφθηκε: {reason.strip()}")
                        st.rerun()
                    except Exception as e:
                        st.error(h.humanize(e))

        st.markdown('<div class="ne-h" style="margin-top:1rem"></div>', unsafe_allow_html=True)
        ui.label_row("Ρώτα για αυτή την παραγγελία")
        ui.chat_box(f"chat_po_{ui.safe_key(key)}", lambda: wh.order_context(o),
                    placeholder="π.χ. Τι γίνεται αν παραγγείλουμε την Τετάρτη;",
                    footnote="Ο βοηθός βλέπει μόνο τα νούμερα αυτής της παραγγελίας.")
