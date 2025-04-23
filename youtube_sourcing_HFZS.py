import os
import shutil
import time
import re
import json
import datetime
import pickle
import requests
import xml.etree.ElementTree as ET
from transformers import pipeline
from google.oauth2 import service_account
from googleapiclient.discovery import build

# -------------------------
# ABSOLUTE PATH SETUP
# -------------------------
INPUT_DIR = os.getenv("DATA_DIR", ".")
WORK_DIR = os.getenv("WORK_DIR", ".")

SEEN_FILE_READ          = os.path.join(INPUT_DIR, "seen_channels.pickle")
CACHE_FILE_READ         = os.path.join(INPUT_DIR, "search_cache.json")
SERVICE_ACCOUNT_FILE_IN = os.path.join(INPUT_DIR, "service_account.json")

SEEN_FILE               = os.path.join(WORK_DIR, "seen_channels.pickle")
CACHE_FILE              = os.path.join(WORK_DIR, "search_cache.json")
SERVICE_ACCOUNT_FILE    = os.path.join(WORK_DIR, "service_account.json")

SPREADSHEET_ID = "12ZCiyliodaReN7PxByGMDKberiWP9kHuozK50hd_8jg"

for src, dst in [
    (SEEN_FILE_READ, SEEN_FILE),
    (CACHE_FILE_READ, CACHE_FILE),
    (SERVICE_ACCOUNT_FILE_IN, SERVICE_ACCOUNT_FILE)
]:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

# -------------------------
# CONFIGURATION
# -------------------------
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/spreadsheets"
]
CURATED_SEARCH_TERMS = [
    "Technology & Gadgets", "Personal Finance & Investing", "Health & Wellness",
    "Beauty & Fashion", "Gaming", "Education & How-To Content",
    "Business & Entrepreneurship", "Automotive", "Lifestyle & Vlogging",
    "Food & Cooking", "Travel", "Parenting & Family", "Home & DIY",
    "News & Commentary", "Music & Performance", "Movies & TV Commentary",
    "Science & Curiosity", "Luxury & High-End Lifestyle",
    "Real Estate & Investing", "Motivational & Self-Development"
]
MAX_PAGES = 4
MIN_SUBSCRIBERS = 1000
CATEGORY_TABS = CURATED_SEARCH_TERMS + ["Unassigned"]
OUTREACH_TABS = ["GeneralCreators - Outreach", "LongFormCreators - Outreach"]
ALL_TABS = CATEGORY_TABS + OUTREACH_TABS
CATEGORIES = CURATED_SEARCH_TERMS + ["Unassigned", "Exclude"]

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

def truncate(text, max_chars=500):
    return text[:max_chars] + ("…" if len(text) > max_chars else "")

def build_sequence(meta):
    return (
        f"Channel Title: {meta.get('title','')}\n"
        f"Handle: {meta.get('customUrl','')}\n"
        f"About: {truncate(meta.get('desc',''))}\n"
        f"Tags: {', '.join(meta.get('tags',[]))}\n"
        f"Latest Video Title: {meta.get('latestTitle','')}\n"
        f"Latest Video Description: {truncate(meta.get('latestVideoDesc',''))}\n"
        f"Channel Category: {meta.get('channelCategory','')}"
    )

def chunkify(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def classify_zero_shot_batch(metas, batch_size=16):
    sequences = [build_sequence(m) for m in metas]
    labels = []
    for seq_batch in chunkify(sequences, batch_size):
        out = classifier(
            sequences=seq_batch,
            candidate_labels=CURATED_SEARCH_TERMS,
            hypothesis_template="This channel is about {}.",
            multi_label=False
        )
        if isinstance(out, dict):
            labels.append(out['labels'][0])
        else:
            labels.extend([item['labels'][0] for item in out])
    return labels

def get_credentials():
    return service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

def load_seen():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, 'rb') as f:
            return pickle.load(f)
    return set()

def save_seen(seen):
    with open(SEEN_FILE, 'wb') as f:
        pickle.dump(seen, f)

def load_cache():
    try:
        return json.load(open(CACHE_FILE))
    except:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def cache_valid(entry):
    return (time.time() - entry.get('timestamp', 0)) < 86400

def extract_email(text):
    m = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m[0] if m else ''

def extract_instagram(text):
    m = re.findall(r"(?:https?://)?(?:www\.)?instagram\.com/([\w\.]+)", text)
    return '@' + m[0] if m else ''

def format_currency(x):
    return f"${int(x):,}"

def channel_age_months(published_at):
    try:
        dt = datetime.datetime.fromisoformat(published_at.replace('Z', '+00:00'))
    except:
        return 1.0
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(1.0, (now - dt).days / 30.0)

def build_link(cid, custom_url=None):
    if custom_url and custom_url.startswith('@'):
        return f"https://www.youtube.com/{custom_url.lower()}"
    return f"https://www.youtube.com/channel/{cid}"

def ensure_sheet(creds, tab):
    svc = build('sheets', 'v4', credentials=creds)
    ss = svc.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    names = [s['properties']['title'] for s in ss['sheets']]
    if tab not in names:
        svc.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests':[{'addSheet':{'properties':{'title':tab,'gridProperties':{'columnCount':18}}}}]}
        ).execute()

def append_rows(creds, tab, rows):
    svc = build('sheets', 'v4', credentials=creds)
    svc.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{tab}!A2:R",
        valueInputOption='RAW',
        body={'values':rows}
    ).execute()

def get_latest_video_description_rss(cid):
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {'atom':'http://www.w3.org/2005/Atom','media':'http://search.yahoo.com/mrss/'}
        ent = root.find('atom:entry', ns)
        if ent:
            desc = ent.find('media:group/media:description', ns)
            return desc.text if desc is not None else ''
    except:
        pass
    return ''

def update_spreadsheet_with_rss(creds, sheet):
    svc = build("sheets", "v4", credentials=creds)
    rng = f"{sheet}!A2:R"
    data = svc.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=rng).execute().get("values",[])
    updated = []
    for row in data:
        if len(row) < 18:
            row += [""] * (18 - len(row))
        cid = row[17]
        rss = get_latest_video_description_rss(cid)
        if rss:
            em = extract_email(rss)
            ins = extract_instagram(rss)
            if em: row[7] = em
            if ins: row[5] = ins
        updated.append(row)
        time.sleep(1)
    svc.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=rng,
        valueInputOption="RAW",
        body={"values": updated}
    ).execute()

def search_channels(youtube, term):
    cache = load_cache()
    if term in cache and cache_valid(cache[term]):
        return cache[term]['items']
    items, token = [], None
    for _ in range(MAX_PAGES):
        resp = youtube.search().list(
            part='snippet', type='channel',
            q=term, maxResults=50,
            order='relevance', pageToken=token
        ).execute()
        items += resp.get('items', [])
        token = resp.get('nextPageToken')
        if not token:
            break
        time.sleep(1.5)
    cache[term] = {'items': items, 'timestamp': time.time()}
    save_cache(cache)
    return items

def fetch_details(youtube, ids):
    info = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        resp = youtube.channels().list(
            part='snippet,statistics,brandingSettings,topicDetails',
            id=','.join(chunk)
        ).execute()
        for ch in resp.get('items', []):
            cid = ch['id']
            sn = ch['snippet']; st = ch['statistics']; bs = ch.get('brandingSettings', {}).get('channel', {})
            td = ch.get('topicDetails', {})
            info[cid] = {
                'title': sn.get('title',''),
                'customUrl': sn.get('customUrl',''),
                'desc': sn.get('description',''),
                'tags': bs.get('keywords','').split(',') if bs.get('keywords') else [],
                'channelCategory': ' '.join(td.get('topicCategories',[])),
                'subs': int(st.get('subscriberCount',0)),
                'views': int(st.get('viewCount',0)),
                'publishedAt': sn.get('publishedAt',''),
                'latestTitle':'','latestVideoDesc':''
            }
    return info

def main():
    creds = get_credentials()
    yt = build('youtube','v3',credentials=creds)
    seen = load_seen()
    total = 0
    category_counts = {c: 0 for c in CATEGORY_TABS}

    for term in CURATED_SEARCH_TERMS:
        print(f"🔍 Searching YouTube for category: {term}")
        items = search_channels(yt, term)
        ids = [it['snippet']['channelId'] for it in items]
        details = fetch_details(yt, ids)
        pairs = [(cid, d) for cid, d in details.items() if cid.lower() not in seen]
        if not pairs:
            print(f" → No new channels for {term}")
            continue

        metas = [d for _, d in pairs]
        labels = classify_zero_shot_batch(metas, batch_size=16)
        category_rows = {c: [] for c in CATEGORY_TABS}

        for (cid, d), cat in zip(pairs, labels):
            print(f" • Evaluating {cid} → Classified as: {cat}")
            if cat == 'Exclude':
                continue
            if d['subs'] < MIN_SUBSCRIBERS:
                continue
            m = channel_age_months(d['publishedAt'])
            low = (d['views']/m)/1000 * 2.5 * 0.75
            high = (d['views']/m)/1000 * 2.5 * 1.25
            if low < 3500 or high > 350000:
                continue
            row = [
                time.strftime("%-m/%-d/%Y"),
                "YouTube API",
                build_link(cid, d['customUrl']),
                format_currency(low),
                format_currency(high),
                extract_instagram(d['desc']),
                "",
                extract_email(d['desc']),
                *['']*8,
                "",
                cid
            ]
            category_rows.setdefault(cat, []).append(row)
            seen.add(cid.lower())
            total += 1
            category_counts[cat] += 1

        for cat, rows in category_rows.items():
            if rows:
                ensure_sheet(creds, cat)
                append_rows(creds, cat, rows)

        save_seen(seen)

    print(f"🎯 Total Channels Added: {total}")
    for cat, count in category_counts.items():
        if count:
            print(f"  • {cat}: {count}")

    time.sleep(5)
    for tab in CATEGORY_TABS:
        update_spreadsheet_with_rss(creds, tab)

if __name__ == '__main__':
    main()
