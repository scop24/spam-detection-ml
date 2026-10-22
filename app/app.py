"""
app/app.py
----------
Spam Detection Filter — Streamlit demo app.

Tabs:
  Classifier     -- paste a message, get a spam probability + verdict,
                    see WHICH words pushed the decision, experiment with
                    the decision threshold, or batch-classify a CSV.
  Model Insights -- leaderboard vs baselines, confusion matrix, ROC/PR
                    curves, threshold analysis, cross-domain (SMS<->email)
                    results, robustness tests, misclassified examples.
  Data Explorer  -- class balance, message lengths, top words per class.

The app loads the pipeline saved by train.py and starts instantly.
Run from the project root:  streamlit run app/app.py
"""

import html
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.features  # noqa: E402,F401  (needed to unpickle the pipeline)
from src.explain import word_contributions  # noqa: E402
from src.features import clean_text  # noqa: E402

st.set_page_config(page_title="Spam Detection Filter", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(160deg, #0d0f1d 0%, #171233 55%, #2a1030 100%); color: #eceaff; }
section[data-testid="stSidebar"] { background: rgba(10,10,26,.88); border-right: 1px solid rgba(255,255,255,.08); }
section[data-testid="stSidebar"] * { color: #d9d4ff !important; }
h1, h2, h3 { color: #f5f3ff !important; font-weight: 700 !important; }
p, span, label { color: #cfc9f2; }
.hero {
    background: linear-gradient(120deg, rgba(124,58,237,.85), rgba(219,39,119,.70));
    border: 1px solid rgba(255,255,255,.25); border-radius: 24px;
    padding: 26px 32px; margin: 4px 0 18px 0; box-shadow: 0 16px 44px rgba(5,2,20,.55);
}
.hero .t { font-size: 2.2rem; font-weight: 800; color: #fff; }
.hero .s { color: rgba(255,255,255,.88); font-size: 1.02rem; margin-top: 4px; }
.kpi { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14);
       border-radius: 16px; padding: 15px 18px; height: 100%; }
.kpi .label { font-size: .8rem; text-transform: uppercase; letter-spacing: 1.1px; color: #b3a8ec; font-weight: 600; }
.kpi .value { font-size: 1.8rem; font-weight: 800; color: #fff; margin: 3px 0 1px 0; }
.kpi .note { font-size: .9rem; color: #c4bcf0; }
.verdict-spam { background: linear-gradient(120deg, #dc2626, #991b1b); border-radius: 18px;
    padding: 20px 26px; color: #fff; font-size: 1.7rem; font-weight: 800; text-align: center;
    box-shadow: 0 12px 34px rgba(220,38,38,.35); }
.verdict-ham { background: linear-gradient(120deg, #16a34a, #166534); border-radius: 18px;
    padding: 20px 26px; color: #fff; font-size: 1.7rem; font-weight: 800; text-align: center;
    box-shadow: 0 12px 34px rgba(22,163,74,.30); }
.msgbox { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.13);
    border-radius: 14px; padding: 16px 18px; font-size: 1.05rem; line-height: 1.75; color: #e9e6ff; }
.hl-spam { background: rgba(239,68,68,.50); border-radius: 5px; padding: 0 4px; color: #fff; font-weight: 600; }
.hl-ham { background: rgba(34,197,94,.38); border-radius: 5px; padding: 0 4px; color: #fff; font-weight: 600; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #fff; margin: 14px 0 6px 0; }
button[data-baseweb="tab"] { font-size: 1.03rem; font-weight: 600; color: #b3a8ec; padding: 10px 16px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #fff; background: rgba(255,255,255,.08); }
.stButton > button { background: linear-gradient(120deg, #7c3aed, #db2777); color: #fff; border: none;
    border-radius: 11px; font-weight: 600; box-shadow: 0 6px 16px rgba(219,39,119,.30); }
.footer { text-align: center; color: #8f86c4; font-size: .85rem; margin: 28px 0 4px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#e6e1ff", family="Outfit"),
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    if height:
        fig.update_layout(height=height)
    return fig


def kpi(col, label, value, note):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="note">{note}</div></div>',
                 unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all():
    with open(os.path.join(ROOT, "models", "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    model = joblib.load(os.path.join(ROOT, "models", "spam_pipeline.joblib"))
    return metadata, metrics, model


try:
    METADATA, METRICS, MODEL = load_all()
except Exception:
    st.error("Trained model not found. Please run `python train.py` first, then reload.")
    st.stop()

WINNER = METADATA["winner"]
TUNED_THRESHOLD = float(METADATA["threshold"])
WORD_SCORES = METRICS.get("word_scores", {})


@st.cache_data
def load_dataset(which):
    path = os.path.join(ROOT, "data", "processed",
                        "sms.csv" if which == "SMS" else "email.csv")
    return pd.read_csv(path)


def predict_proba(text):
    return float(MODEL.predict_proba(pd.DataFrame({"text": [text]}))[0][1])


def highlighted_html(text, model):
    """The message as the model sees it (cleaned), with spam-pushing words
    in red and ham-pushing words in green."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    contrib = word_contributions(model, text)
    scores = {}
    for i, tok in enumerate(tokens):
        s = contrib.get(tok, WORD_SCORES.get(tok, 0.0))
        if i + 1 < len(tokens):
            pair = f"{tok} {tokens[i + 1]}"
            ps = contrib.get(pair, WORD_SCORES.get(pair, 0.0))
            s = max(s, ps) if ps > 0 else min(s, ps) if ps < 0 else s
        scores[i] = s
    parts = []
    for i, tok in enumerate(tokens):
        esc = html.escape(tok)
        s = scores.get(i, 0.0)
        if s >= 0.8:
            parts.append(f'<span class="hl-spam">{esc}</span>')
        elif s <= -0.8:
            parts.append(f'<span class="hl-ham">{esc}</span>')
        else:
            parts.append(esc)
    return " ".join(parts)


EXAMPLES = {
    "🚨 KYC scam (spam)": "Dear Customer, your SBI account will be BLOCKED today due to incomplete KYC. Update immediately here: bit.ly/sbi-kyc-update or account will be suspended in 24 hrs.",
    "🎰 Lottery scam (spam)": "Congratulations! You have won Rs 25,00,000 in KBC Lottery. To claim, pay processing fee Rs 5,000 and WhatsApp your bank details to +91 98765 43210.",
    "📦 Delivery scam (spam)": "Sir your parcel is held at customs. Pay Rs 49 clearance charge to release your delivery today: india-post-delivery.link/pay",
    "🔐 Genuine OTP (ham)": "Your OTP for login is 482913. It is valid for 10 minutes. Please do not share it with anyone. - Team CampusPortal",
    "👨‍👩‍👦 Family message (ham)": "Hi beta, reached hostel safely? Call when you get time. Papa is asking about your exam dates.",
    "🎓 College notice (ham)": "Reminder: DBMS lab viva is tomorrow at 10 AM in Lab 3. Please bring your record file and hall ticket. - Class Representative",
}

# ---------------------------------------------------------------------------
# Sidebar + hero
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Spam Detection Filter")
    st.markdown("An ML filter that reads a message and decides: **spam or not** — "
                "and shows its reasoning.")
    st.markdown("---")
    st.markdown(f"**Model:** {WINNER}  \n"
                f"**Decision threshold:** {TUNED_THRESHOLD} (tuned on validation)  \n"
                f"**Test F1:** {METRICS['threshold']['test_at_threshold']['f1']}  \n"
                f"**Training data:** {METRICS['data']['sms_total']} SMS messages")
    st.markdown("---")
    st.caption("UCI SMS Spam Collection + SpamAssassin emails · scikit-learn / XGBoost · Streamlit")

st.markdown(
    '<div class="hero"><div class="t">🛡️ Spam Detection Filter</div>'
    '<div class="s">Machine learning that catches spam — and explains itself. '
    'Paste any SMS or email below and watch it decide.</div></div>',
    unsafe_allow_html=True)

tab_cls, tab_ins, tab_data = st.tabs(["🔍 Classifier", "📊 Model Insights", "🗂️ Data Explorer"])

# ============================================================ CLASSIFIER ====
with tab_cls:
    st.markdown('<div class="section-title">Try an example</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (label, msg) in enumerate(EXAMPLES.items()):
        if cols[i % 3].button(label, key=f"ex{i}"):
            st.session_state["msg_input"] = msg

    text = st.text_area("✉️ Paste a message (SMS or email)",
                        height=130, key="msg_input",
                        placeholder="e.g. Congratulations! You have won a FREE prize...")
    if text and len(text) > 5000:
        st.warning("Very long message — only the first 5,000 characters were analysed.")
        text = text[:5000]

    threshold = st.slider(
        "Decision threshold — messages with spam probability above this are flagged",
        0.05, 0.99, TUNED_THRESHOLD, 0.01,
        help="Higher threshold = fewer real messages wrongly flagged (higher precision), "
             "but more spam slips through (lower recall).")

    pr = METRICS["pr_curve_val"]
    idx = int(np.argmin([abs(t - threshold) for t in pr["thresholds"]]))
    v1, v2, v3 = st.columns(3)
    kpi(v1, "At this threshold (validation)", f"P {pr['precisions'][idx]:.2f} · R {pr['recalls'][idx]:.2f}",
        "Precision / recall measured on held-out validation data")
    kpi(v2, "Project default", f"{TUNED_THRESHOLD}", "Chosen on validation: best F1 with precision ≥ 0.95")
    kpi(v3, "Why not 0.50?", "Real messages matter", "A false alarm (real mail in spam folder) costs more than a missed spam")

    if text and text.strip():
        proba = predict_proba(text)
        is_spam = proba >= threshold
        gcol, vcol = st.columns([1, 1.4])
        with gcol:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                number=dict(suffix="%", font=dict(size=34, color="#fff")),
                title=dict(text="Spam probability", font=dict(color="#cfc9f2")),
                gauge=dict(axis=dict(range=[0, 100], tickcolor="#8f86c4"),
                           bar=dict(color="#ef4444" if is_spam else "#22c55e"),
                           bgcolor="rgba(255,255,255,0.06)",
                           steps=[dict(range=[0, threshold * 100], color="rgba(34,197,94,0.15)"),
                                  dict(range=[threshold * 100, 100], color="rgba(239,68,68,0.15)")],
                           threshold=dict(line=dict(color="#fff", width=3),
                                          value=threshold * 100))))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=250,
                                margin=dict(t=40, b=10, l=20, r=20),
                                font=dict(color="#e6e1ff", family="Outfit"))
            st.plotly_chart(gauge, use_container_width=True)
        with vcol:
            st.markdown(
                f'<div class="{"verdict-spam" if is_spam else "verdict-ham"}">'
                f'{"🚨 SPAM" if is_spam else "✅ NOT SPAM"}</div>',
                unsafe_allow_html=True)
            st.markdown("**How the model sees your message** — red words pushed it "
                        "towards spam, green words towards genuine:")
            st.markdown(f'<div class="msgbox">{highlighted_html(text, MODEL)}</div>',
                        unsafe_allow_html=True)
            st.caption("URLs, phone numbers and digits are replaced by tokens before "
                       "scoring — a message full of them is itself a spam signal.")
    else:
        st.info("👆 Pick an example or paste your own message to see the filter in action.")

    st.markdown('<div class="section-title">📥 Batch mode — classify a CSV</div>',
                unsafe_allow_html=True)
    st.write("Upload a CSV with a message column (named `text` or `message`). "
             "You get the same file back with spam probabilities and verdicts.")
    upload = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv")
    if upload is not None:
        try:
            bdf = pd.read_csv(upload).head(5000)
            col = next((c for c in bdf.columns
                        if c.lower() in ("text", "message", "msg", "sms")), bdf.columns[0])
            probs = MODEL.predict_proba(bdf[[col]].rename(columns={col: "text"}))[:, 1]
            bdf["spam_probability"] = np.round(probs, 3)
            bdf["verdict"] = np.where(probs >= threshold, "SPAM", "NOT SPAM")
            st.success(f"Classified {len(bdf)} messages from column '{col}': "
                       f"{int((bdf['verdict'] == 'SPAM').sum())} spam, "
                       f"{int((bdf['verdict'] == 'NOT SPAM').sum())} not spam.")
            st.dataframe(bdf, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download results CSV",
                               bdf.to_csv(index=False).encode(),
                               "spam_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process that file: {e}")

# ============================================================== INSIGHTS ====
with tab_ins:
    st.markdown('<div class="section-title">🏆 Leaderboard — test set (untouched during training)</div>',
                unsafe_allow_html=True)
    lb = pd.DataFrame(METRICS["leaderboard"])
    show_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    st.dataframe(lb[show_cols], use_container_width=True, hide_index=True)
    st.write(f"**Winner: {WINNER}** — selected by cross-validation PR-AUC "
             f"({METRICS['winner_cv_pr_auc']}) on the training split alone. Note the "
             "majority baseline: 87% accuracy while catching **zero** spam — accuracy "
             "alone is meaningless on imbalanced data.")

    thr = METRICS["threshold"]
    st.markdown('<div class="section-title">🎚️ Threshold tuning (false positives cost more)</div>',
                unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    kpi(t1, "Chosen threshold", f"{thr['threshold']}",
        f"Validation precision {thr['val_precision']} · recall {thr['val_recall']}")
    kpi(t2, "Test @ tuned threshold",
        f"F1 {thr['test_at_threshold']['f1']}",
        f"Precision {thr['test_at_threshold']['precision']} · Recall {thr['test_at_threshold']['recall']}")
    kpi(t3, "Test @ default 0.5", f"F1 {thr['test_at_0.5']['f1']}",
        f"Precision {thr['test_at_0.5']['precision']} · Recall {thr['test_at_0.5']['recall']}")
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(ROOT, "reports", "figures", "threshold_analysis.png"))
    with c2:
        st.image(os.path.join(ROOT, "reports", "figures", "confusion_matrix.png"))

    st.markdown('<div class="section-title">📈 ROC & precision-recall curves</div>',
                unsafe_allow_html=True)
    st.image(os.path.join(ROOT, "reports", "figures", "roc_pr_curves.png"))

    if METRICS["cross_domain"]:
        st.markdown('<div class="section-title">🌍 Cross-domain test — train on one world, test on another</div>',
                    unsafe_allow_html=True)
        cd = pd.DataFrame([
            {"trained → tested": k, "accuracy": v["accuracy"], "precision": v["precision"],
             "recall": v["recall"], "f1": v["f1"], "pr_auc": v.get("pr_auc", None)}
            for k, v in METRICS["cross_domain"].items()])
        st.dataframe(cd, use_container_width=True, hide_index=True)
        st.image(os.path.join(ROOT, "reports", "figures", "cross_domain.png"))
        st.write("Performance drops sharply across domains: 2003-era emails and "
                 "short SMS messages use different vocabulary, lengths and tricks. "
                 "A filter trained on one distribution does not automatically "
                 "understand the other — this is why real spam systems retrain "
                 "continuously on fresh data.")

    st.markdown('<div class="section-title">🥷 Robustness — obfuscated spam & the Indian test set</div>',
                unsafe_allow_html=True)
    rob = METRICS["robustness"]
    rrows = [{"variant": k.replace("_", " "), "messages": v["n"],
              "recall @ tuned threshold": v["recall_at_threshold"],
              "recall @ 0.5": v["recall_at_0.5"]} for k, v in rob.items()]
    st.dataframe(pd.DataFrame(rrows), use_container_width=True, hide_index=True)
    im = METRICS["indian_metrics"]
    st.write(f"**Hand-written Indian test set (30 modern messages):** accuracy "
             f"{im['accuracy']}, spam recall {im['recall']}, precision {im['precision']}. "
             "Honest result: today's Indian scams use vocabulary (KYC, UPI, courier "
             "fees) that barely exists in the 2000s-era training data, so several slip "
             "through — and a few genuine bank/OTP-style messages get flagged. The fix "
             "in production would be training on recent local data.")
    with st.expander("See all 30 Indian test messages and their verdicts"):
        idf = pd.DataFrame(METRICS["indian_details"])
        idf["actual"] = np.where(idf["label"] == 1, "spam", "ham")
        idf["model said"] = np.where(idf["predicted"] == 1, "SPAM", "not spam")
        idf["correct"] = np.where(idf["label"] == idf["predicted"], "✅", "❌")
        st.dataframe(idf[["text", "actual", "model said", "spam_probability", "correct"]],
                     use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">🧠 What the model learned — top words</div>',
                unsafe_allow_html=True)
    st.image(os.path.join(ROOT, "reports", "figures", "top_words.png"))
    st.caption(f"Source: {METRICS['top_words'].get('source', '')}.")
    if METRICS.get("lime_generated"):
        st.markdown('<div class="section-title">🔍 LIME — explaining one prediction</div>',
                    unsafe_allow_html=True)
        st.image(os.path.join(ROOT, "reports", "figures", "lime_example.png"))
        st.caption("LIME perturbs the message (removing words) and watches the "
                   "probability move, revealing each word's local influence.")

    st.markdown('<div class="section-title">⚠️ Error analysis — actual mistakes</div>',
                unsafe_allow_html=True)
    st.write(METRICS["errors"]["plain_language"])
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**False positives** (real messages flagged as spam):")
        for t in METRICS["errors"]["false_positives"]:
            st.markdown(f'<div class="msgbox">{html.escape(t)}</div>', unsafe_allow_html=True)
    with e2:
        st.markdown("**False negatives** (spam that slipped through):")
        for t in METRICS["errors"]["false_negatives"]:
            st.markdown(f'<div class="msgbox">{html.escape(t)}</div>', unsafe_allow_html=True)

# ================================================================ DATA =====
with tab_data:
    which = st.radio("Dataset", ["SMS", "Email"], horizontal=True,
                     help="SMS = UCI collection · Email = SpamAssassin corpus")
    d = load_dataset(which)
    d["class"] = np.where(d["label"] == 1, "spam", "ham")
    d["length"] = d["text"].astype(str).str.len()

    st.markdown('<div class="section-title">⚖️ Class distribution</div>',
                unsafe_allow_html=True)
    counts = d["class"].value_counts().reset_index()
    counts.columns = ["class", "count"]
    fig = px.bar(counts, x="class", y="count", color="class",
                 color_discrete_map={"spam": "#ef4444", "ham": "#22c55e"}, text="count")
    st.plotly_chart(style_fig(fig, height=320), use_container_width=True)
    spam_rate = (d["label"] == 1).mean()
    st.caption(f"Only {spam_rate:.0%} of {which} messages are spam — the imbalance "
               "that makes accuracy a misleading metric.")

    st.markdown('<div class="section-title">📏 Message length by class</div>',
                unsafe_allow_html=True)
    fig2 = px.histogram(d[d["length"] <= d["length"].quantile(0.99)],
                        x="length", color="class", nbins=50, barmode="overlay",
                        opacity=0.75, labels={"length": "Characters"},
                        color_discrete_map={"spam": "#ef4444", "ham": "#22c55e"})
    st.plotly_chart(style_fig(fig2, height=340), use_container_width=True)

    st.markdown('<div class="section-title">🔤 Most frequent words per class</div>',
                unsafe_allow_html=True)
    from sklearn.feature_extraction.text import CountVectorizer
    cleaned = d["text"].astype(str).map(clean_text)
    vec = CountVectorizer(max_features=5000, stop_words="english")
    Xc = vec.fit_transform(cleaned)
    vocab = np.array(vec.get_feature_names_out())
    w1, w2 = st.columns(2)
    for col, lab, color, title in [(w1, 1, "#ef4444", "Spam vocabulary"),
                                   (w2, 0, "#22c55e", "Ham vocabulary")]:
        sums = np.asarray(Xc[d["label"].to_numpy() == lab].sum(axis=0)).ravel()
        top = np.argsort(sums)[-15:][::-1]
        tdf = pd.DataFrame({"word": vocab[top], "count": sums[top]})
        f = px.bar(tdf, x="count", y="word", orientation="h",
                   color_discrete_sequence=[color])
        f.update_layout(yaxis=dict(autorange="reversed"), title=title)
        col.plotly_chart(style_fig(f, height=420), use_container_width=True)

    st.markdown('<div class="section-title">🧾 Sample messages</div>',
                unsafe_allow_html=True)
    sample_class = st.selectbox("Show samples of", ["spam", "ham"], key="sample_class")
    for t in d[d["class"] == sample_class]["text"].head(5):
        st.markdown(f'<div class="msgbox">{html.escape(str(t)[:400])}</div>',
                    unsafe_allow_html=True)

st.markdown('<div class="footer">🛡️ Spam Detection Filter · ' + WINNER +
            f' · Test F1 {METRICS["threshold"]["test_at_threshold"]["f1"]} @ threshold '
            f'{TUNED_THRESHOLD} · ML Lab Project</div>', unsafe_allow_html=True)
