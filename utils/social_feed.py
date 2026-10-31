"""Live social-media feed for the blue ticker of the Akal University admissions assistant.

Fetches the latest posts of the official Facebook / Twitter(X) / YouTube / Instagram pages, keeps ONLY the ones
about events, webinars, competitions, workshops etc., and returns them newest first.

IMPORTANT (why tokens are needed): Facebook, Instagram and Twitter(X) do not allow anonymous reading of their
pages. They must be read through their OFFICIAL APIs with a token that the university's social-media admin
creates. Put the tokens in the .env file (see the names below). A platform without a token is simply skipped
(nothing is ever invented). YouTube works WITHOUT any token (public RSS feed).

.env names:
    FACEBOOK_ACCESS_TOKEN=...      (Page access token)      FACEBOOK_PAGE_ID=me  (or the numeric page id)
    INSTAGRAM_ACCESS_TOKEN=...     (Instagram Business / Creator token)   INSTAGRAM_USER_ID=me
    TWITTER_BEARER_TOKEN=...       (X API v2 bearer token)  TWITTER_USERNAME=UniversityAkal
    YOUTUBE_CHANNEL_ID=...         (optional - found automatically if empty)
    SOCIAL_MAX_AGE_DAYS=45         (older posts are not shown)
    SOCIAL_PER_SOURCE=4            (max items per platform)
    SOCIAL_MAX_ITEMS=12            (max items in the stripe)
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

CACHE_PATH = BASE_DIR / "cache" / "social_feed.json"
TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (AkalUniversityAdmissionsBot/1.0)"}

FACEBOOK_URL = "https://www.facebook.com/AkalUniversity/"
TWITTER_DEFAULT_USER = "UniversityAkal"
INSTAGRAM_URL = "https://www.instagram.com/akaluniversity/"
YOUTUBE_USER = "akaluniversityts"
YOUTUBE_URL = "https://www.youtube.com/akaluniversityts"

SOURCE_ORDER = ["Facebook", "Twitter", "YouTube", "Instagram"]
SOURCE_ICON = {"Facebook": "📘", "Twitter": "🐦", "YouTube": "▶️", "Instagram": "📸"}
SOURCE_HOME = {"Facebook": FACEBOOK_URL, "Twitter": f"https://twitter.com/{TWITTER_DEFAULT_USER}",
               "YouTube": YOUTUBE_URL, "Instagram": INSTAGRAM_URL}

# Only posts that talk about one of these are shown (webinars, competitions, events ...).
EVENT_RE = re.compile(
    r"\b(webinars?|web[- ]?talk|seminars?|workshops?|competitions?|contests?|events?|conferences?|symposi\w+|"
    r"fests?|festivals?|lectures?|talks?|sessions?|camps?|inaugurat\w+|ceremon\w+|orientation|hackathons?|"
    r"quiz|celebrat\w+|declamation|cultural|convocation|panel|exhibition|expo|rally|drive|tournament|"
    r"matches?|olympiad|bootcamp|training|induction|felicitat\w+|guest|visit|awareness|MoU|"
    r"admissions? (?:open|started|are open))\b"
    r"|वेबिनार|कार्यशाला|प्रतियोगिता|समारोह|सेमिनार|कार्यक्रम|संगोष्ठी|शिविर|व्याख्यान|उद्घाटन"
    r"|ਵੈਬੀਨਾਰ|ਵਰਕਸ਼ਾਪ|ਮੁਕਾਬਲਾ|ਸਮਾਗਮ|ਸੈਮੀਨਾਰ|ਕੈਂਪ|ਉਦਘਾਟਨ",
    re.IGNORECASE,
)


def _env(name, default=""):
    return os.environ.get(name, "").strip() or default


def _int_env(name, default):
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# ---------------- small helpers ----------------
def _ts(value):
    """Any ISO / RFC-2822 date text -> epoch seconds (UTC), or None."""
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = parsedate_to_datetime(str(value))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _clean(text, limit=170):
    """One short, readable line: no links, no hashtag clutter, no line breaks."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text or "")
    text = re.sub(r"(?:\s*#\w+)+\s*$", "", text.strip())
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-–—") + "…"
    return text


def _date_text(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d %b %Y")


def _item(source, raw_text, url, ts, force_event=False):
    text = _clean(raw_text)
    if not text or ts is None:
        return None
    return {
        "source": source,
        "text": text,
        "url": url or SOURCE_HOME[source],
        "ts": float(ts),
        "event": bool(force_event or EVENT_RE.search(raw_text or "")),
    }


def _get_json(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------- Facebook (Graph API) ----------------
def _facebook():
    token = _env("FACEBOOK_ACCESS_TOKEN")
    if not token:
        return None
    page = _env("FACEBOOK_PAGE_ID", "me")
    base = f"https://graph.facebook.com/v19.0/{page}"
    items = []
    data = _get_json(f"{base}/posts", {
        "fields": "message,created_time,permalink_url", "limit": 30, "access_token": token})
    for p in data.get("data", []):
        it = _item("Facebook", p.get("message"), p.get("permalink_url"), _ts(p.get("created_time")))
        if it:
            items.append(it)
    try:  # official Facebook Events of the page (webinars / programmes that are coming up)
        ev = _get_json(f"{base}/events", {
            "fields": "name,start_time,place", "time_filter": "upcoming", "limit": 10, "access_token": token})
        for e in ev.get("data", []):
            ts = _ts(e.get("start_time"))
            if not e.get("name") or ts is None:
                continue
            place = (e.get("place") or {}).get("name")
            text = f"Upcoming: {e['name']} - {_date_text(ts)}" + (f", {place}" if place else "")
            it = _item("Facebook", text, f"https://www.facebook.com/events/{e.get('id', '')}", ts, force_event=True)
            if it:
                items.append(it)
    except Exception as err:
        print(f"[social] Facebook events skipped: {str(err)[:150]}")
    return items


# ---------------- Instagram (Graph API) ----------------
def _instagram():
    token = _env("INSTAGRAM_ACCESS_TOKEN")
    if not token:
        return None
    user = _env("INSTAGRAM_USER_ID", "me")
    data = _get_json(f"https://graph.instagram.com/{user}/media", {
        "fields": "caption,permalink,timestamp", "limit": 30, "access_token": token})
    items = []
    for p in data.get("data", []):
        it = _item("Instagram", p.get("caption"), p.get("permalink"), _ts(p.get("timestamp")))
        if it:
            items.append(it)
    return items


# ---------------- Twitter / X (API v2) ----------------
_TW_ID_CACHE = {}


def _twitter():
    token = _env("TWITTER_BEARER_TOKEN")
    if not token:
        return None
    username = _env("TWITTER_USERNAME", TWITTER_DEFAULT_USER)
    auth = {"Authorization": f"Bearer {token}"}
    if username not in _TW_ID_CACHE:
        r = requests.get(f"https://api.twitter.com/2/users/by/username/{username}", headers=auth, timeout=TIMEOUT)
        r.raise_for_status()
        _TW_ID_CACHE[username] = r.json()["data"]["id"]
    r = requests.get(
        f"https://api.twitter.com/2/users/{_TW_ID_CACHE[username]}/tweets",
        params={"max_results": 20, "tweet.fields": "created_at", "exclude": "retweets,replies"},
        headers=auth, timeout=TIMEOUT,
    )
    r.raise_for_status()
    items = []
    for t in r.json().get("data", []):
        it = _item("Twitter", t.get("text"), f"https://twitter.com/{username}/status/{t.get('id')}",
                   _ts(t.get("created_at")))
        if it:
            items.append(it)
    return items


# ---------------- YouTube (public RSS feed, no token needed) ----------------
_YT_CHANNEL_CACHE = {}
_NS = {"a": "http://www.w3.org/2005/Atom", "m": "http://search.yahoo.com/mrss/"}


def _youtube_channel_id():
    if _env("YOUTUBE_CHANNEL_ID"):
        return _env("YOUTUBE_CHANNEL_ID")
    if "id" in _YT_CHANNEL_CACHE:
        return _YT_CHANNEL_CACHE["id"]
    for url in (YOUTUBE_URL, f"https://www.youtube.com/@{YOUTUBE_USER}", f"https://www.youtube.com/c/{YOUTUBE_USER}"):
        try:
            r = requests.get(url, headers={**HEADERS, "Cookie": "CONSENT=YES+1"}, timeout=TIMEOUT)
            m = (re.search(r'"externalId":"(UC[\w-]{22})"', r.text)
                 or re.search(r"channel_id=(UC[\w-]{22})", r.text)
                 or re.search(r"/channel/(UC[\w-]{22})", r.text))
            if m:
                _YT_CHANNEL_CACHE["id"] = m.group(1)
                return m.group(1)
        except Exception:
            continue
    return None


def _youtube():
    feeds = []
    cid = _youtube_channel_id()
    if cid:
        feeds.append(f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}")
    feeds.append(f"https://www.youtube.com/feeds/videos.xml?user={YOUTUBE_USER}")
    last_err = None
    for feed in feeds:
        try:
            r = requests.get(feed, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            last_err = e
            continue
        items = []
        for e in root.findall("a:entry", _NS):
            title = e.findtext("a:title", default="", namespaces=_NS)
            link = e.find("a:link", _NS)
            desc = e.findtext("m:group/m:description", default="", namespaces=_NS) or ""
            it = _item("YouTube", title, link.get("href") if link is not None else None,
                       _ts(e.findtext("a:published", namespaces=_NS)))
            if it:
                it["event"] = bool(EVENT_RE.search(title + " " + desc[:200]))
                items.append(it)
        return items
    raise last_err or RuntimeError("YouTube feed not reachable")


# ---------------- cache of the last good result (used only if a platform is temporarily down) ----------------
def _load_cache():
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[social] could not save cache: {e}")


# ---------------- public function ----------------
def fetch_live_items():
    """Latest event / webinar / competition posts of all four platforms, newest first.
    Returns [{'source', 'text', 'url', 'ts', 'event'}, ...] (empty list when nothing could be read)."""
    fetchers = {"Facebook": _facebook, "Twitter": _twitter, "YouTube": _youtube, "Instagram": _instagram}
    cache = _load_cache()
    by_source = {}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fn): name for name, fn in fetchers.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[social] {name} failed: {str(e)[:200]}")
                by_source[name] = cache.get(name, [])  # last good copy (still age-filtered below)
                continue
            if result is None:
                print(f"[social] {name}: no access token set in .env - skipped")
                by_source[name] = []
                continue
            by_source[name] = result
            cache[name] = result
    _save_cache(cache)

    now = time.time()
    max_age = _int_env("SOCIAL_MAX_AGE_DAYS", 45) * 86400
    per_source = _int_env("SOCIAL_PER_SOURCE", 4)
    max_items = _int_env("SOCIAL_MAX_ITEMS", 12)

    chosen, seen = [], set()
    for name in SOURCE_ORDER:
        fresh = [i for i in by_source.get(name, []) if i.get("event") and now - i["ts"] <= max_age]
        fresh.sort(key=lambda i: -i["ts"])
        for i in fresh[:per_source]:
            key = i["text"].lower()
            if key not in seen:
                seen.add(key)
                chosen.append(i)
    chosen.sort(key=lambda i: -i["ts"])
    return chosen[:max_items]


if __name__ == "__main__":  # quick test:  python -m utils.social_feed
    for it in fetch_live_items():
        print(it["source"], "|", _date_text(it["ts"]), "|", it["text"], "|", it["url"])
