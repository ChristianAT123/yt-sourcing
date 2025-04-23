import os
import shutil
import time
import re
import json
import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from transformers import pipeline
from google.oauth2 import service_account
from googleapiclient.discovery import build

# -------------------------
# ABSOLUTE PATH SETUP
# -------------------------
INPUT_DIR = os.getenv("DATA_DIR", ".")
WORK_DIR  = os.getenv("WORK_DIR", ".")

for name in ("search_cache.json", "service_account.json", "iab_taxonomy.json"):
    src = os.path.join(INPUT_DIR, name)
    dst = os.path.join(WORK_DIR, name)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

CACHE_FILE = os.path.join(WORK_DIR, "search_cache.json")
SA_FILE    = os.path.join(WORK_DIR, "service_account.json")
IAB_FILE   = os.path.join(WORK_DIR, "iab_taxonomy.json")
TERM_HIST  = os.path.join(WORK_DIR, "term_history.json")

# Print path so you can verify where the file is written
print("TERM_HIST path =", TERM_HIST)

# -------------------------
# CONFIGURATION
# -------------------------
SPREADSHEET_ID   = "12ZCiyliodaReN7PxByGMDKberiWP9kHuozK50hd_8jg"
SCOPES           = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/spreadsheets"
]

CURATED_SEARCH_TERMS = [
    "Technology & Gadgets","Personal Finance & Investing","Health & Wellness",
    "Beauty & Fashion","Gaming","Education & How-To Content",
    "Business & Entrepreneurship","Automotive","Lifestyle & Vlogging",
    "Food & Cooking","Travel","Parenting & Family","Home & DIY",
    "News & Commentary","Music & Performance","Movies & TV Commentary",
    "Science & Curiosity","Luxury & High-End Lifestyle",
    "Real Estate & Investing","Motivational & Self-Development"
]
CATEGORY_TABS = CURATED_SEARCH_TERMS + ["Unassigned"]
OUTREACH_TABS = ["GeneralCreators - Outreach","LongFormCreators - Outreach"]
ALL_TABS      = CATEGORY_TABS + OUTREACH_TABS

MAX_PAGES       = 4
PAGES_PER_SEED  = MAX_PAGES       # ← fetch all 4 pages per seed
SEEDS_PER_RUN   = 50
MIN_SUBSCRIBERS = 1000

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

# -------------------------
# AUTOCOMPLETE + IAB SEEDS
# -------------------------
def fetch_suggestions(term):
    url    = "http://suggestqueries.google.com/complete/search"
    params = {"client":"firefox","ds":"yt","q":term}
    r      = requests.get(url, params=params, timeout=5)
    return r.json()[1] if r.status_code == 200 else []

def load_json_file(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except:
        return {}

def load_term_history():
    if os.path.exists(TERM_HIST):
        return json.load(open(TERM_HIST, encoding="utf-8"))
    th = {}
    iab = load_json_file(IAB_FILE)
    for cat in CURATED_SEARCH_TERMS:
        seeds = []
        seeds += fetch_suggestions(cat)
        seeds += iab.get(cat, [])
        th[cat] = {"suggestions": seeds, "last_index": 0}
    json.dump(th, open(TERM_HIST, "w"), indent=2)
    return th

def save_term_history(th):
    json.dump(th, open(TERM_HIST, "w"), indent=2)

# -------------------------
# DE-DUPE VIA SHEETS
# -------------------------
def normalize_link(link):
    p      = urlparse(link)
    domain = p.netloc.lower().replace("www.","")
    path   = p.path.rstrip("/").lower()
    return f"{domain}{path}"

def load_seen_links_from_sheet(creds):
    svc  = build("sheets","v4",credentials=creds)
    seen = set()
    for tab in ALL_TABS:
        try:
            vals = svc.spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID, range=f"{tab}!C2:C"
            ).execute().get("values",[])
        except:
            continue
        for row in vals:
            link = row[0] if row else ""
            if link:
                seen.add(normalize_link(link))
    return seen

# -------------------------
# EXISTING HELPERS
# -------------------------
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
        SA_FILE, scopes=SCOPES
    )

def load_cache():
    try:
        return json.load(open(CACHE_FILE, encoding="utf-8"))
    except:
        return {}

def save_cache(cache):
    json.dump(cache, open(CACHE_FILE, "w"), indent=2)

def cache_valid(entry):
    return (time.time() - entry.get("timestamp", 0)) < 86400

def extract_email(text):
    m = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m[0] if m else ""

def extract_instagram(text):
    m = re.findall(r"(?:https?://)?(?:www\.)?instagram\.com/([\w\.]+)", text)
    return '@' + m[0] if m else ""

def format_currency(x):
    return f"${int(x):,}"

def channel_age_months(published_at):
    try:
        dt   = datetime.datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        now  = datetime.datetime.now(datetime.timezone.utc)
        return max(1.0, (now - dt).days / 30.0)
    except:
        return 1.0

def build_link(cid, custom_url=None):
    base = "https://www.youtube.com"
    if custom_url and custom_url.startswith("@"):
        return f"{base}/{custom_url.lower()}"
    return f"{base}/channel/{cid}"

def ensure_sheet(creds, tab):
    svc   = build('sheets','v4',credentials=creds)
    names = [s['properties']['title'] for s in svc.spreadsheets().get(
               spreadsheetId=SPREADSHEET_ID).execute()['sheets']]
    if tab not in names:
        svc.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests':[{'addSheet':{'properties':{'title':tab}}}]}
        ).execute()

def append_rows(creds, tab, rows):
    svc = build('sheets','v4',credentials=creds)
    svc.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{tab}!A2:R",
        valueInputOption='RAW',
        body={'values':rows}
    ).execute()

def get_latest_video_description_rss(cid):
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
    try:
        r    = requests.get(url, timeout=10); r.raise_for_status()
        root = ET.fromstring(r.content)
        ns   = {'atom':'http://www.w3.org/2005/Atom','media':'http://search.yahoo.com/mrss/'}
        ent  = root.find('atom:entry', ns)
        if ent:
            desc = ent.find('media:group/media:description', ns)
            return desc.text if desc is not None else ""
    except:
        pass
    return ""

def update_spreadsheet_with_rss(creds, sheet):
    svc  = build("sheets","v4",credentials=creds)
    rng  = f"{sheet}!A2:R"
    data = svc.spreadsheets().values().get(
               spreadsheetId=SPREADSHEET_ID, range=rng
           ).execute().get("values", [])
    updated = []
    for row in data:
        if len(row) < 18:
            row += [""] * (18 - len(row))
        cid = row[17]
        rss = get_latest_video_description_rss(cid)
        if rss:
            em  = extract_email(rss)
            ins = extract_instagram(rss)
            if em:  row[7] = em
            if ins: row[5] = ins
        updated.append(row)
        time.sleep(1)
    svc.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=rng,
        valueInputOption="RAW",
        body={"values":updated}
    ).execute()

def search_channels(youtube, term, max_pages=None):
    pages = max_pages or MAX_PAGES
    cache = load_cache()
    key   = f"{term}__{pages}"
    if key in cache and cache_valid(cache[key]):
        return cache[key]["items"]
    items, token = [], None
    for _ in range(pages):
        resp = youtube.search().list(
            part="snippet", type="channel",
            q=term, maxResults=50,
            order="relevance", pageToken=token
        ).execute()
        items += resp.get("items", [])
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(1.5)
    cache[key] = {"items":items,"timestamp":time.time()}
    save_cache(cache)
    return items

def fetch_details(youtube, ids):
    info = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        resp  = youtube.channels().list(
            part="snippet,statistics,brandingSettings,topicDetails",
            id=",".join(chunk)
        ).execute()
        for ch in resp.get("items", []):
            cid, sn, st = ch["id"], ch["snippet"], ch["statistics"]
            bs = ch.get("brandingSettings", {}).get("channel", {})
            td = ch.get("topicDetails", {})
            info[cid] = {
                "title": sn.get("title",""),
                "customUrl": sn.get("customUrl",""),
                "desc": sn.get("description",""),
                "tags": bs.get("keywords","").split(",") if bs.get("keywords") else [],
                "channelCategory": " ".join(td.get("topicCategories",[])),
                "subs": int(st.get("subscriberCount",0)),
                "views": int(st.get("viewCount",0)),
                "publishedAt": sn.get("publishedAt",""),
                "latestTitle":"",
                "latestVideoDesc":""
            }
    return info

def main():
    creds        = get_credentials()
    seen_urls    = load_seen_links_from_sheet(creds)
    term_history = load_term_history()
    yt           = build("youtube","v3",credentials=creds)

    # round-robin seed selection until we hit SEEDS_PER_RUN
    seeds = []
    while len(seeds) < SEEDS_PER_RUN:
        for cat, data in term_history.items():
            idx  = data["last_index"]
            seed = data["suggestions"][idx]
            seeds.append((cat, seed))
            data["last_index"] = (idx + 1) % len(data["suggestions"])
            if len(seeds) >= SEEDS_PER_RUN:
                break
    save_term_history(term_history)

    total = 0
    category_counts = {c: 0 for c in CATEGORY_TABS}

    for cat, seed in seeds:
        print(f"🔍 {cat} → “{seed}”")
        items   = search_channels(yt, seed, max_pages=PAGES_PER_SEED)
        ids     = [it["snippet"]["channelId"] for it in items]
        details = fetch_details(yt, ids)

        rows_by_cat = {c: [] for c in CATEGORY_TABS}
        metas       = list(details.values())
        labels      = classify_zero_shot_batch(metas, batch_size=16)

        for (cid, d), label in zip(details.items(), labels):
            link = build_link(cid, d["customUrl"])
            norm = normalize_link(link)
            if norm in seen_urls or d["subs"] < MIN_SUBSCRIBERS:
                continue

            m   = channel_age_months(d["publishedAt"])
            low = (d["views"]/m)/1000 * 2.5 * 0.75
            high= (d["views"]/m)/1000 * 2.5 * 1.25
            if low < 3500 or high > 350000:
                continue

            row = [
                time.strftime("%-m/%-d/%Y"),
                "YouTube API",
                link,
                format_currency(low),
                format_currency(high),
                extract_instagram(d["desc"]),
                "",
                extract_email(d["desc"]),
                *[""]*8,
                cid
            ]
            rows_by_cat[label].append(row)
            seen_urls.add(norm)
            total += 1
            category_counts[label] += 1

        for label, rows in rows_by_cat.items():
            if rows:
                ensure_sheet(creds, label)
                append_rows(creds, label, rows)

    print(f"🎯 Total Channels Added: {total}")
    for cat, cnt in category_counts.items():
        if cnt:
            print(f"  • {cat}: {cnt}")

    time.sleep(5)
    for tab in CATEGORY_TABS:
        update_spreadsheet_with_rss(creds, tab)

if __name__ == '__main__':
    main()
