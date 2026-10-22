"""
src/data.py
-----------
Downloads and caches the datasets (project runs offline afterwards):

1. SMS  -- UCI SMS Spam Collection (5,572 messages).
2. Email -- SpamAssassin public corpus (spam + easy_ham tarballs),
   parsed from raw email files to plain body text, for the
   cross-domain (SMS <-> email) generalisation test.
3. Indian test set -- ~30 hand-written modern Indian-style messages
   (KYC/bank/lottery/loan/OTP/delivery scams vs genuine messages),
   used only as an extra robustness check, never for training.

Exact duplicate messages are removed BEFORE any split: identical copies
landing in both train and test would let the model "memorise" test items
and inflate every metric.
"""

import email
import html
import io
import os
import re
import tarfile
from email import policy

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
SA_BASE = "https://spamassassin.apache.org/old/publiccorpus"
SA_FILES = {
    "spam": ["20030228_spam.tar.bz2", "20030228_spam_2.tar.bz2"],
    "ham": ["20030228_easy_ham.tar.bz2"],
}
MAX_EMAIL_PER_CLASS = 2000   # cap for balance + reasonable size


# ---------------------------------------------------------------------------
# SMS (UCI)
# ---------------------------------------------------------------------------
def load_sms():
    cache = os.path.join(PROCESSED_DIR, "sms.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    os.makedirs(RAW_DIR, exist_ok=True)
    import zipfile
    r = requests.get(SMS_URL, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.lower().startswith("smsspamcollection")][0]
    df = pd.read_csv(io.BytesIO(z.read(name)), sep="\t", header=None,
                     names=["label", "text"], quoting=3)
    df["label"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    df["source"] = "sms"
    df, removed = dedupe(df)
    print(f"SMS: {len(df)} messages after removing {removed} exact duplicates")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Email (SpamAssassin)
# ---------------------------------------------------------------------------
def _email_body(raw_bytes):
    """Best-effort plain-text body of a raw email file."""
    try:
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        body = msg.get_body(preferencelist=("plain", "html"))
        if body is not None:
            text = body.get_content()
        else:
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        parts.append(part.get_content())
                    except Exception:
                        pass
            text = "\n".join(parts)
    except Exception:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip any HTML tags
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_emails():
    cache = os.path.join(PROCESSED_DIR, "email.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for label_name, files in SA_FILES.items():
        for fname in files:
            url = f"{SA_BASE}/{fname}"
            print(f"Downloading {url} ...")
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            tf = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:bz2")
            for member in tf.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                text = _email_body(f.read())
                if len(text) >= 20:
                    rows.append({"label": 1 if label_name == "spam" else 0,
                                 "text": text, "source": "email"})
    df = pd.DataFrame(rows)
    # Balance the classes by capping the larger one.
    counts = df["label"].value_counts()
    cap = int(min(counts.min(), MAX_EMAIL_PER_CLASS))
    frames = [g.sample(n=min(len(g), cap), random_state=42)
              for _, g in df.groupby("label")]
    df = pd.concat(frames, ignore_index=True)
    df, removed = dedupe(df)
    print(f"Email: {len(df)} messages after removing {removed} exact duplicates "
          f"(spam={int((df['label'] == 1).sum())}, ham={int((df['label'] == 0).sum())})")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Shared helpers + Indian test set
# ---------------------------------------------------------------------------
def dedupe(df):
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df, before - len(df)


INDIAN_MESSAGES = [
    # ---- spam (label 1) ----
    (1, "Dear Customer, your SBI account will be BLOCKED today due to incomplete KYC. Update your KYC immediately here: bit.ly/sbi-kyc-update or your account will be suspended within 24 hrs."),
    (1, "Congratulations! You have won Rs 25,00,000 in KBC Lottery 2026. To claim your prize money, pay processing fee of Rs 5,000 and WhatsApp your bank details to +91 98765 43210."),
    (1, "Urgent: Your HDFC credit card reward points worth Rs 8,450 will expire TODAY. Redeem now at hdfc-rewards-claim.xyz before midnight. Hurry!"),
    (1, "Get instant personal loan of Rs 5 lakh at 0% interest, no documents, no CIBIL check. Approval in 5 minutes. Apply now: quickloan-india.com/apply"),
    (1, "Your electricity bill is overdue. Your connection will be disconnected tonight at 9 PM. To avoid disconnection pay immediately via link: ebill-pay.now/verify"),
    (1, "Part time job offer! Earn Rs 3,000-5,000 daily from home by just liking YouTube videos. No investment needed. Join our Telegram channel now."),
    (1, "Sir your parcel is held at customs. Pay Rs 49 clearance charge to release your delivery today: india-post-delivery.link/pay"),
    (1, "You have received a refund of Rs 12,999 from Income Tax Department. Click here to verify your account and receive money instantly: it-refund2026.in"),
    (1, "Double your money in 30 days! Invest in our guaranteed crypto scheme, 200% returns assured. Limited seats. Transfer to UPI id crypto-double@okaxis."),
    (1, "Your PhonePe account is at risk. Share the OTP sent to your mobile to verify your identity and prevent account freeze. Do not share with anyone else."),
    (1, "Work from home data entry job. Salary Rs 45,000/month. Registration fee only Rs 999. Pay now and start earning from tomorrow."),
    (1, "FREE iPhone 16 Pro! You are our lucky 1,00,000th visitor. Claim your gift by paying shipping charges Rs 199 only: lucky-winner-prize.com"),
    (1, "Your SIM will be deactivated in 2 hours as per TRAI rules. Complete e-KYC now by calling 1800-419-XXXX from your registered number."),
    (1, "Stock tip: Buy XYZ Infra today, sure shot multibagger, 500% returns in 3 months guaranteed. Join premium group @StockGuruTips."),
    (1, "Dear customer your Aadhaar is linked with illegal activities. Police case registered. Press 1 to talk to cyber cell officer immediately."),
    (1, "Get Rs 500 cashback instantly! Just download our app and enter code WIN500. Offer valid for first 100 users only. Download: cashback-app-download.apk"),
    # ---- ham (label 0) ----
    (0, "Hi beta, reached hostel safely? Call when you get time. Papa is asking about your exam dates."),
    (0, "Your OTP for login is 482913. It is valid for 10 minutes. Please do not share it with anyone. - Team CampusPortal"),
    (0, "Your order #402-9182736 (Running Shoes) has been delivered. Rate your purchase on the app. Thank you for shopping with us."),
    (0, "Reminder: DBMS lab viva is tomorrow at 10 AM in Lab 3. Please bring your record file and hall ticket. - Class Representative"),
    (0, "Bhai kal subah 6 baje ground pe milte hain, cricket practice hai. Ball main le aaunga, tu wickets le aana."),
    (0, "Your account XX4821 has been debited with Rs 1,250.00 on 20-Oct-2026 at AMAZON PAY. Available balance Rs 8,432.10. Not you? Call bank helpline."),
    (0, "Meeting rescheduled to 3:30 PM today. Please find the updated agenda attached. Let me know if the time works for everyone."),
    (0, "Happy birthday dear! Wishing you a wonderful year ahead. Party kab de raha hai?"),
    (0, "The train 12951 Mumbai Rajdhani will depart from platform 4 at 17:00 hrs. Please report 30 minutes early. PNR 8642019573."),
    (0, "Mummy ne kaha hai ki Sunday ko ghar aa jana, tumhare liye special khana banega. Train book kar lena time se."),
    (0, "Your package with tracking ID 7829104856 is out for delivery and will arrive by 8 PM today. Please keep Rs 0 ready, it is prepaid."),
    (0, "Results for Semester 5 internal assessment are published on the student portal. Check your marks and report discrepancies by Friday."),
    (0, "Can you send me the notes of yesterday's operating systems lecture? I missed it because of the dentist appointment."),
    (0, "Gym membership renewal due on 1st November. Pay at the front desk to continue without interruption. - FitLife Gym"),
]

INDIAN_COLUMNS = ["label", "text"]


def load_indian_test_set():
    cache = os.path.join(PROCESSED_DIR, "indian_test_set.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    df = pd.DataFrame(INDIAN_MESSAGES, columns=INDIAN_COLUMNS)
    df["source"] = "indian_handwritten"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    sms = load_sms()
    print(sms["label"].value_counts().to_dict())
    try:
        em = load_emails()
        print(em["label"].value_counts().to_dict())
    except Exception as e:
        print(f"Email download failed (will continue with SMS only): {e}")
    ind = load_indian_test_set()
    print(f"Indian test set: {len(ind)} messages "
          f"(spam={int((ind.label == 1).sum())}, ham={int((ind.label == 0).sum())})")
