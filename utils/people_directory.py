"""Verified directory of the University's people (officials, office-bearers, faculty).

WHY THIS FILE EXISTS
    "Who is the Registrar?" used to return information about *registration*, and "Who is the VC?" got an
    "I don't have it". Searching free text for such questions is unreliable, so the people of the University are
    kept as STRUCTURED, SOURCE-TAGGED records and the questions about them are answered from those records:

      Designation | Name | Email | Mobile | Department | Note | Source page

    - data/university_officials.txt       verified records (copied from official pages of https://auts.ac.in/)
    - data/university_officials_live.txt  written by refresh_live() when the website is read again (optional)

    Nothing is hard-coded in the code: the code only matches the question to the records. Different wordings
    ("VC", "Vice Chancellor", "वाइस चांसलर", "कुलपति", "who heads the university" ...) lead to the SAME records and
    therefore to the SAME answer. A role / person that is not in the records is reported honestly as "not
    verified" - never guessed.

USED BY rag_pipeline.py:  handle(query, language, last_query)  and  unsupported_names(answer, reference_text)

TEST FROM THE TERMINAL
    python -m utils.people_directory --ask "who is the registrar"
    python -m utils.people_directory --list
    python -m utils.people_directory --refresh          (reads the official website, needs internet)
"""
import difflib
import os
import re
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SEED_FILE = "university_officials.txt"
LIVE_FILE = "university_officials_live.txt"
GOVERNANCE_URL = "https://auts.ac.in/governance/"
FACULTY_URL = "https://auts.ac.in/akal-faculty/"
# The chat assistant is called "Dr. Sukhjeet Singh" and an official with the same name is listed on the website.
# When a question is about that name, the answer is about the OFFICIAL, and a one-line note explains the overlap.
PERSONA_NAME = os.environ.get("ASSISTANT_PERSON_NAME", "Sukhjeet Singh")

# =====================================================================================================
#  SPELLING / SCRIPT HELPERS
# =====================================================================================================
_HONORIFICS = {
    "dr", "prof", "professor", "mr", "mrs", "ms", "miss", "shri", "shree", "sri", "smt", "sardar", "sardarni",
    "baba", "sant", "maj", "major", "gen", "general", "retd", "retired", "col", "capt", "brig", "honorable",
    "honourable", "hon", "ble", "he", "padma", "late", "md", "mbbs", "phd", "ji", "sahib", "saab", "sir",
    "madam", "mam", "ma", "er", "adv",
}
_ACADEMIC_TITLES = {"dr": "Dr", "prof": "Prof", "professor": "Prof"}
_GENERIC_NAME_WORDS = ("singh", "kaur", "kour", "kumar", "sharma", "devi", "lal", "rani", "sinh")

_STOP_EN = {
    "who", "whom", "whose", "what", "which", "when", "where", "why", "how", "is", "are", "was", "were", "the",
    "of", "a", "an", "about", "tell", "me", "us", "please", "pls", "plz", "give", "show", "share", "name", "names",
    "detail", "details", "info", "information", "regarding", "email", "mail", "id", "phone", "mobile", "number",
    "contact", "address", "office", "akal", "university", "au", "auts", "current", "present", "new", "now",
    "currently", "in", "at", "on", "for", "and", "or", "to", "from", "with", "his", "her", "their", "this", "that",
    "do", "does", "did", "can", "you", "i", "my", "your", "our", "by", "as", "be", "been", "will", "would",
    "should", "hello", "hi", "hey", "talwandi", "sabo", "kaun", "kon", "kaunsa", "hai", "hain", "ka", "ki", "ke",
    "ko", "kya", "batao", "bataiye", "bataye", "bataiyen", "mujhe", "jankari", "sab", "all", "every", "list",
    "complete", "full", "head", "heads", "headed", "incharge", "charge", "sir", "madam", "ji", "ka", "ke", "ki",
    "wala", "wale", "hota", "hoti", "hote", "kaun", "email", "ids", "numbers", "sabhi", "teacher", "teachers",
    "faculty", "professor", "professors", "staff", "member", "members", "department", "dept",
}
_STOP_NATIVE = {
    "कौन", "है", "हैं", "का", "की", "के", "को", "क्या", "बताओ", "बताइए", "बताएं", "मुझे", "ईमेल", "फोन", "नंबर",
    "अकाल", "यूनिवर्सिटी", "विश्वविद्यालय", "में", "से", "और", "बारे", "जी", "श्री", "डॉ", "डॉक्टर", "प्रोफेसर",
    "सर", "मैडम", "कृपया", "नाम", "कौनसा", "कौन-सा", "आज", "अभी", "वर्तमान", "हैं।", "है।", "सभी", "सारे",
    "ਕੌਣ", "ਹੈ", "ਹਨ", "ਦਾ", "ਦੀ", "ਦੇ", "ਨੂੰ", "ਕੀ", "ਦੱਸੋ", "ਮੈਨੂੰ", "ਈਮੇਲ", "ਫ਼ੋਨ", "ਨੰਬਰ", "ਅਕਾਲ",
    "ਯੂਨੀਵਰਸਿਟੀ", "ਵਿੱਚ", "ਤੋਂ", "ਅਤੇ", "ਬਾਰੇ", "ਜੀ", "ਸ੍ਰੀ", "ਡਾ", "ਪ੍ਰੋ", "ਸਰ", "ਨਾਮ", "ਸਾਰੇ", "ਕਿਰਪਾ",
}
_TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F\u0A00-\u0A7F]+")

# ---- Devanagari / Gurmukhi -> Latin (rough, only used to compare names) ----
_DEV_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "ड़": "r", "ढ़": "rh", "ज़": "z", "फ़": "f",
}
_DEV_VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
            "ओ": "o", "औ": "au", "ऑ": "o"}
_DEV_MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
              "ृ": "ri", "ॉ": "o"}
_GUR_CONS = {
    "ਕ": "k", "ਖ": "kh", "ਗ": "g", "ਘ": "gh", "ਙ": "n", "ਚ": "ch", "ਛ": "chh", "ਜ": "j", "ਝ": "jh", "ਞ": "n",
    "ਟ": "t", "ਠ": "th", "ਡ": "d", "ਢ": "dh", "ਣ": "n", "ਤ": "t", "ਥ": "th", "ਦ": "d", "ਧ": "dh", "ਨ": "n",
    "ਪ": "p", "ਫ": "ph", "ਬ": "b", "ਭ": "bh", "ਮ": "m", "ਯ": "y", "ਰ": "r", "ਲ": "l", "ਵ": "v", "ੜ": "r",
    "ਸ": "s", "ਹ": "h", "\u0a36": "sh", "\u0a59": "kh", "\u0a5a": "g", "\u0a5b": "z", "\u0a5e": "f",
}
_GUR_VOW = {"ਅ": "a", "ਆ": "aa", "ਇ": "i", "ਈ": "ee", "ਉ": "u", "ਊ": "oo", "ਏ": "e", "ਐ": "ai", "ਓ": "o",
            "ਔ": "au", "ੳ": "", "ੲ": ""}
_GUR_MATRA = {"ਾ": "aa", "ਿ": "i", "ੀ": "ee", "ੁ": "u", "ੂ": "oo", "ੇ": "e", "ੈ": "ai", "ੋ": "o", "ੌ": "au"}


def _translit(word):
    """A Hindi / Punjabi word -> rough Latin spelling (a Latin word is returned unchanged)."""
    if not re.search(r"[\u0900-\u097F\u0A00-\u0A7F]", word):
        return word
    out, i, n = [], 0, len(word)
    dev = bool(re.search(r"[\u0900-\u097F]", word))
    while i < n:
        c = word[i]
        if dev and c in _DEV_CONS:
            out.append(_DEV_CONS[c])
            j = i + 1
            if j < n and word[j] == "़":
                j += 1
            nxt = word[j] if j < n else ""
            if nxt in _DEV_MATRA:
                out.append(_DEV_MATRA[nxt])
                i = j + 1
                continue
            if nxt == "्":
                i = j + 1
                continue
            if j < n:  # an inherent "a" in the middle of a word (the vowel sign is dropped at the end)
                out.append("a")
            i = j
            continue
        if dev and c in _DEV_VOW:
            out.append(_DEV_VOW[c])
        elif dev and c in "ंँ":
            out.append("n")
        elif dev and c == "ः":
            out.append("h")
        elif (not dev) and c in _GUR_CONS:
            out.append(_GUR_CONS[c])
        elif (not dev) and c in _GUR_VOW:
            out.append(_GUR_VOW[c])
        elif (not dev) and c in _GUR_MATRA:
            out.append(_GUR_MATRA[c])
        elif (not dev) and c in "ਂੰ":
            out.append("n")
        elif (not dev) and c == "ੱ" and i + 1 < n and word[i + 1] in _GUR_CONS:
            out.append(_GUR_CONS[word[i + 1]][0])
        i += 1
    latin = "".join(out)
    return re.sub(r"nh$", "ngh", latin)


def _pkey(token):
    """Phonetic key of a name word: 'Sukhjeet' / 'Sukhjit' -> 'sukjit', 'Swarn' / 'Sawarn' -> 'svarn'."""
    t = re.sub(r"[^a-z]", "", _translit(token).lower())
    t = t.replace("w", "v").replace("ph", "f").replace("ck", "k")
    t = re.sub(r"(ee|ii|ea|ie|ey)", "i", t)
    t = re.sub(r"(oo|uu|ou)", "u", t)
    t = t.replace("aa", "a")
    t = re.sub(r"(ai|ay|ei)", "e", t)
    t = t.replace("h", "")
    return re.sub(r"(.)\1+", r"\1", t)


def _skeleton(key):
    return re.sub(r"[aeiou]", "", key)


def _tok_match(a, b):
    """True when two phonetic keys are the same name word (spelling variants allowed)."""
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    sa = _skeleton(a)
    if len(sa) >= 3 and sa == _skeleton(b) and ratio >= 0.70:
        return True
    return ratio >= 0.86


_GENERIC_KEYS = {_pkey(w) for w in _GENERIC_NAME_WORDS}


def _name_tokens(name):
    """Alphabetic words of a name without honorifics ('Prof. Sawarn Singh' -> ['sawarn', 'singh'])."""
    return [t.lower() for t in re.findall(r"[A-Za-z]+", name or "") if t.lower() not in _HONORIFICS]


def _title_set(name):
    return {_ACADEMIC_TITLES[t.lower()] for t in re.findall(r"[A-Za-z]+", name or "") if t.lower() in _ACADEMIC_TITLES}


# =====================================================================================================
#  ROLES  (the same office named in English / Hindi / Hinglish / Punjabi / abbreviation -> ONE key)
# =====================================================================================================
def _spec(key, q, d, en, hi, pa):
    return {"key": key, "q": re.compile(q, re.IGNORECASE), "d": re.compile(d, re.IGNORECASE),
            "label": {"English": en, "Hindi": hi, "Punjabi": pa}}


ROLE_SPECS = [
    _spec("pro_vc",
          r"pro[\s-]*vice[\s-]*chancellor|(?<![a-z0-9])pro[\s-]*v\.?\s?c\.?(?![a-z0-9])|प्रो[\s-]*वाइस[\s-]*चांसलर|"
          r"ਪ੍ਰੋ[\s-]*ਵਾਈਸ[\s-]*ਚਾਂਸਲਰ",
          r"pro[\s-]*vice[\s-]*chancellor", "Pro Vice-Chancellor", "प्रो वाइस-चांसलर", "ਪ੍ਰੋ ਵਾਈਸ-ਚਾਂਸਲਰ"),
    _spec("vc",
          r"vice[\s-]*chancellor|(?<![a-z0-9])v\.?\s?c\.?(?![a-z0-9])|वाइस[\s-]*चांसलर|वाइस[\s-]*चान्सलर|वीसी|"
          r"वी\.\s?सी\.?|कुलपति|ਵਾਈਸ[\s-]*ਚਾਂਸਲਰ|ਵੀਸੀ|ਉਪ[\s-]*ਕੁਲਪਤੀ|ਕੁਲਪਤੀ",
          r"vice[\s-]*chancellor", "Vice-Chancellor", "वाइस-चांसलर (कुलपति)", "ਵਾਈਸ-ਚਾਂਸਲਰ"),
    _spec("founder",
          r"founders?|founded|founding|संस्थापक|ਸੰਸਥਾਪਕ",
          r"founder(?:[\s-]*(?:chancellor|president))?", "Founder Chancellor", "संस्थापक चांसलर", "ਸੰਸਥਾਪਕ ਚਾਂਸਲਰ"),
    _spec("chancellor",
          r"chancellor|कुलाधिपति|चांसलर|चान्सलर|ਚਾਂਸਲਰ|ਕੁਲਾਧਿਪਤੀ",
          r"chancellor(?:\s*/\s*president)?", "Chancellor", "चांसलर (कुलाधिपति)", "ਚਾਂਸਲਰ"),
    _spec("registrar",
          r"(?<![a-z])registrars?(?![a-z])|रजिस्ट्रार|कुलसचिव|ਰਜਿਸਟਰਾਰ",
          r"registrar", "Registrar", "रजिस्ट्रार (कुलसचिव)", "ਰਜਿਸਟਰਾਰ"),
    _spec("appellate",
          r"appellate\s+authority|अपीलीय\s*प्राधिकारी|ਅਪੀਲੀ",
          r"appellate\s+authority", "Appellate Authority", "अपीलीय प्राधिकारी", "ਅਪੀਲੀ ਅਥਾਰਟੀ"),
    _spec("coe",
          r"(?<![a-z0-9])c\.?\s?o\.?\s?e\.?(?![a-z0-9])|controller\s+of\s+exam\w*|परीक्षा\s*नियंत्रक|"
          r"ਪ੍ਰੀਖਿਆ\s*ਕੰਟਰੋਲਰ",
          r"controller\s+of\s+exam\w*", "Controller of Examination", "परीक्षा नियंत्रक", "ਪ੍ਰੀਖਿਆ ਕੰਟਰੋਲਰ"),
    _spec("finance",
          r"finance\s+officer|accounts?\s+officer|(?<![a-z])fdo(?![a-z])|treasurer|वित्त\s*अधिकारी|"
          r"ਵਿੱਤ\s*(?:ਅਫ਼ਸਰ|ਅਧਿਕਾਰੀ)",
          r"finance\s+officer|accounts?\s+officer|treasurer", "Finance Officer", "वित्त अधिकारी", "ਵਿੱਤ ਅਧਿਕਾਰੀ"),
    _spec("cvo",
          r"chief\s+vigilance\s+officer|(?<![a-z])cvo(?![a-z])|मुख्य\s*सतर्कता\s*अधिकारी|ਮੁੱਖ\s*ਚੌਕਸੀ",
          r"chief\s+vigilance\s+officer", "Chief Vigilance Officer", "मुख्य सतर्कता अधिकारी", "ਮੁੱਖ ਚੌਕਸੀ ਅਧਿਕਾਰੀ"),
    _spec("ombud",
          r"ombuds\w*|लोकपाल|ओम्बड्स\w*|ਓਮਬਡਸ\w*",
          r"ombuds\w*", "Ombudsperson", "लोकपाल (ओम्बड्सपर्सन)", "ਓਮਬਡਸਪਰਸਨ"),
    _spec("tpo",
          r"training\s*(?:and|&)?\s*placement\s*(?:officer|head|in-?charge)?|(?<![a-z])tpo(?![a-z])|"
          r"placement\s+officer|प्लेसमेंट\s*अधिकारी|ਪਲੇਸਮੈਂਟ\s*ਅਫ਼ਸਰ",
          r"training\s*(?:and|&)?\s*placement|(?<![a-z])tpo(?![a-z])", "Training and Placement Officer (TPO)",
          "ट्रेनिंग एवं प्लेसमेंट ऑफिसर (TPO)", "ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਪਲੇਸਮੈਂਟ ਅਫ਼ਸਰ (TPO)"),
    _spec("dsw",
          r"(?<![a-z0-9])d\.?\s?s\.?\s?w\.?(?![a-z0-9])|dean\s*(?:of\s*)?students?'?s?\s*welfare|छात्र\s*कल्याण",
          r"dean\s*(?:of\s*)?students?'?s?\s*welfare|(?<![a-z])dsw(?![a-z])", "Dean Student Welfare",
          "डीन स्टूडेंट वेलफेयर", "ਡੀਨ ਸਟੂਡੈਂਟ ਵੈਲਫੇਅਰ"),
    _spec("dean",
          r"(?<![a-z])deans?(?![a-z])|डीन|अधिष्ठाता|ਡੀਨ",
          r"(?<![a-z])deans?(?![a-z])", "Dean", "डीन", "ਡੀਨ"),
    _spec("director",
          r"(?<![a-z])directors?(?![a-z])|निदेशक|डायरेक्टर|ਡਾਇਰੈਕਟਰ",
          r"(?<![a-z])directors?(?![a-z])", "Director", "निदेशक", "ਡਾਇਰੈਕਟਰ"),
    _spec("hod",
          r"(?<![a-z0-9])h\.?\s?o\.?\s?d\.?(?![a-z0-9])|head\s+of\s+(?:the\s+)?(?:dept\.?|department)|"
          r"department\s+head|विभागाध्यक्ष|ਵਿਭਾਗ\s*(?:ਦੇ\s*)?ਮੁਖੀ",
          r"head\s+of\s+(?:the\s+)?(?:dept\.?|department)|(?<![a-z])hod(?![a-z])|^head\b|\bhead,",
          "Head of Department", "विभागाध्यक्ष", "ਵਿਭਾਗ ਮੁਖੀ"),
    _spec("proctor",
          r"proctor|प्रॉक्टर|प्रोक्टर|ਪ੍ਰੌਕਟਰ", r"proctor", "Proctor", "प्रॉक्टर", "ਪ੍ਰੌਕਟਰ"),
    _spec("principal",
          r"principal|प्रिंसिपल|प्राचार्य|ਪ੍ਰਿੰਸੀਪਲ", r"principal", "Principal", "प्रिंसिपल", "ਪ੍ਰਿੰਸੀਪਲ"),
    _spec("librarian",
          r"librarian|लाइब्रेरियन|ਲਾਇਬ੍ਰੇਰੀਅਨ", r"librarian", "Librarian", "लाइब्रेरियन", "ਲਾਇਬ੍ਰੇਰੀਅਨ"),
]
_SPEC_BY_KEY = {s["key"]: s for s in ROLE_SPECS}
_LIST_ORDER = ["chancellor", "vc", "registrar", "finance", "coe", "cvo", "ombud", "tpo", "appellate"]
_DEPT_ROLES = {"dean", "hod", "director"}

# departments (to answer "HOD of physics" / "faculty of chemistry" honestly)
_DEPTS = {
    "Physics": r"physics|भौतिक|ਭੌਤਿਕ",
    "Chemistry": r"chemistry|रसायन|ਰਸਾਇਣ",
    "Mathematics": r"math(?:s|ematics)?|गणित|ਗਣਿਤ",
    "Commerce": r"commerce|कॉमर्स|ਕਾਮਰਸ",
    "Management": r"management|business|mba|प्रबंध|ਪ੍ਰਬੰਧ",
    "English": r"english|अंग्रेज़ी|अंग्रेजी|ਅੰਗਰੇਜ਼ੀ",
    "Punjabi": r"punjabi|पंजाबी|ਪੰਜਾਬੀ",
    "Economics": r"economics|अर्थशास्त्र|ਅਰਥ",
    "Education": r"education|शिक्षा|ਸਿੱਖਿਆ",
    "Psychology": r"psychology|मनोविज्ञान|ਮਨੋਵਿਗਿਆਨ",
    "Social Science": r"social\s+science",
    "Physiotherapy": r"physiotherapy|फिजियोथेरेपी|ਫਿਜ਼ੀਓ",
    "Botany": r"botany",
    "Zoology": r"zoology",
    "Life Sciences": r"life\s+sciences?|biotech\w*",
    "Computer Science": r"computer|(?<![a-z])cse(?![a-z])|कंप्यूटर|ਕੰਪਿਊਟਰ",
    "Guru Granth Sahib Studies": r"guru\s+granth|(?<![a-z])sggss(?![a-z])",
    "Medical Lab Technology": r"medical\s+lab|(?<![a-z])mlt(?![a-z])",
}
_DEPT_RES = {k: re.compile(v, re.IGNORECASE) for k, v in _DEPTS.items()}

_SELF_RE = re.compile(
    r"(?i)\b(your\s+name|who\s+are\s+you|are\s+you\s+(?:a\s+)?(?:real|human|bot|ai|robot|person|dr)|"
    r"what\s+are\s+you|tum\s+kaun|aap\s+kaun|tumhara\s+naam|aapka\s+naam)\b|आपका\s+नाम|तुम्हारा\s+नाम|आप\s+कौन|"
    r"तुम\s+कौन|ਤੁਹਾਡਾ\s+ਨਾਮ|ਤੁਸੀਂ\s+ਕੌਣ"
)
_PHONE_RE = re.compile(r"(?i)\b(phone|mobile|number|contact|call|whatsapp|reach)\b|फोन|मोबाइल|नंबर|संपर्क|ਫ਼ੋਨ|ਮੋਬਾਈਲ|ਨੰਬਰ|ਸੰਪਰਕ")
_DUTY_RE = re.compile(
    r"(?i)\b(duties|duty|responsibilit\w*|powers?|functions?|role\s+of|what\s+does\s+(?:the\s+)?\w+\s+do|work\s+of|"
    r"kya\s+karte|kaam)\b|कर्तव्य|जिम्मेदारी|ज़िम्मेदारी|काम\s+क्या|ਡਿਊਟੀ|ਜ਼ਿੰਮੇਵਾਰੀ"
)
_LIST_RE = re.compile(
    r"(?i)\b(administration|administrators?|officials?|authorities|leadership|key\s+(?:people|persons|officers)|"
    r"office[\s-]*bearers?|top\s+management)\b|प्रशासन|पदाधिकारी|अधिकारी|ਪ੍ਰਸ਼ਾਸਨ|ਪਦਾਧਿਕਾਰੀ|ਅਧਿਕਾਰੀ"
)
_FACULTY_RE = re.compile(
    r"(?i)\b(faculty|professors?|teachers?|staff|lecturers?|teaches|teach|taught)\b|प्रोफेसर|शिक्षक|अध्यापक|ਅਧਿਆਪਕ|ਪ੍ਰੋਫੈਸਰ|ਫੈਕਲਟੀ|फैकल्टी"
)
_WHO_RE = re.compile(r"(?i)\b(who|whom|whose|kaun|kon|about|name|list|all|show|tell)\b|कौन|ਕੌਣ|सभी|ਸਾਰੇ|बताओ|बताइए")
_FOLLOWUP_RE = re.compile(r"(?i)\b(his|her|their|its|uska|unka|uski|unki|that|this)\b|उसका|उनका|उनकी|उसकी|ਉਸਦਾ|ਉਨ੍ਹਾਂ")


def _roles_in(text, field):
    """The offices named in `text` (field 'q' = question wording, 'd' = designation wording) and the text that is
    left when their words are removed (so 'vice chancellor' does not also count as 'chancellor')."""
    rest, out = text or "", []
    for spec in ROLE_SPECS:
        rx = spec[field]
        if rx.search(rest):
            out.append(spec["key"])
            rest = rx.sub(" ", rest)
    return out, rest


def _dept_hits(text):
    return [name for name, rx in _DEPT_RES.items() if rx.search(text or "")]


# =====================================================================================================
#  LOADING THE RECORDS
# =====================================================================================================
_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")
_CACHE = {"stamp": None, "people": [], "records": 0}


def _source_rank(url):
    u = (url or "").lower()
    for i, key in enumerate(("/governance", "/ugc-mandatory", "/akal-faculty", "/about/chancellors", "/board-of-studies")):
        if key in u:
            return i
    return 9


def _source_label(url):
    u = (url or "").lower()
    for key, label in (("/governance", "Governance page"), ("/ugc-mandatory", "UGC Mandatory Disclosures page"),
                       ("/akal-faculty", "Akal Faculty page"), ("/about/chancellors", "University Leadership page"),
                       ("/board-of-studies", "Board of Studies page")):
        if key in u:
            return label
    return "official website"


def _read_records():
    records = []
    for fname in (SEED_FILE, LIVE_FILE):
        path = DATA_DIR / fname
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [p.strip() for p in ln.split("|")]
            parts += [""] * (7 - len(parts))
            desig, name, email, phone, dept, note, source = parts[:7]
            if not desig or not name:
                continue
            records.append({"designation": desig, "name": name, "email": email if _EMAIL_RE.match(email) else "",
                            "phone": phone, "department": dept, "note": note, "source": source})
    return records


def _build_people(records):
    groups = {}
    for r in sorted(records, key=lambda r: _source_rank(r["source"])):
        toks = [t for t in _name_tokens(r["name"]) if len(t) >= 2]
        if not toks:
            continue
        keys = tuple(sorted({_pkey(t) for t in toks}))
        p = groups.get(keys)
        if p is None:
            p = {"keys": list(keys), "name": r["name"], "variants": [], "desigs": [], "emails": [], "phones": [],
                 "notes": []}
            groups[keys] = p
        if r["name"] not in [v[0] for v in p["variants"]]:
            p["variants"].append((r["name"], r["source"]))
        roles, _ = _roles_in(r["designation"], "d")
        sig = (r["designation"].lower(), r["department"].lower())
        if sig not in [(d["desig"].lower(), d["dept"].lower()) for d in p["desigs"]]:
            p["desigs"].append({"desig": r["designation"], "dept": r["department"], "roles": roles,
                                "source": r["source"]})
        if r["email"] and r["email"].lower() not in [e.lower() for e in p["emails"]]:
            p["emails"].append(r["email"])
        if r["phone"] and r["phone"] not in p["phones"]:
            p["phones"].append(r["phone"])
        if r["note"] and (r["note"], bool(roles)) not in p["notes"]:
            p["notes"].append((r["note"], bool(roles)))
    people = list(groups.values())
    for p in people:  # official-domain addresses first
        p["emails"].sort(key=lambda e: 0 if e.lower().endswith("@auts.ac.in") else 1)
    return people


def _people():
    stamp = []
    for fname in (SEED_FILE, LIVE_FILE):
        path = DATA_DIR / fname
        try:
            stamp.append(path.stat().st_mtime_ns)
        except OSError:
            stamp.append(None)
    if _CACHE["stamp"] != stamp:
        records = _read_records()
        _CACHE.update(stamp=stamp, people=_build_people(records), records=len(records))
    return _CACHE["people"]


def summary():
    people = _people()
    return f"[people] {_CACHE['records']} verified records, {len(people)} people in the official directory"


# =====================================================================================================
#  TEXTS (English / Hindi / Punjabi)
# =====================================================================================================
_T = {
    "English": {
        "is": "The **{role}** of Akal University is **{name}**.",
        "was": "The **{role}** of Akal University was **{name}**.",
        "many": "The **{role}** of Akal University:",
        "p_lead": "**{name}** is the {roles} of Akal University.",
        "p_lead_was": "**{name}** was the {roles} of Akal University.",
        "p_fac": "**{name}** is {desig} in the {dept} at Akal University.",
        "p_fac_nodept": "**{name}** is {desig} at Akal University.",
        "and": " and ",
        "also": "Also",
        "email": "Email", "mobile": "Mobile", "dept": "Department", "qual": "Qualifications", "details": "Details",
        "title_note": "Note: another page of the University website writes this title as “{alt}” ({page}).",
        "persona": "Note: this chat assistant is also called “Dr. Sukhjeet Singh”. The details above are of the "
                   "official listed on the University website.",
        "missing": "I could not find a verified **{role}** in the University's official information that I have, so "
                   "I will not guess. Please check the official Governance page ({url}) or ask the Admission Office.",
        "list_head": "Here are the key officials of Akal University as listed on its official website:",
        "list_cols": ["Position", "Name", "Email"],
        "ambiguous": "More than one person matches your question. Please tell me the full name or the position:",
        "fac_head": "Faculty of **{dept}**, as listed on the official Akal Faculty page:",
        "fac_missing": "I could not find a verified faculty list for the **{dept}** in the official information that "
                       "I have, so I will not guess. Please check the official Akal Faculty page ({url}).",
        "fac_note": "This is the list shown on the website page; the page may not show every faculty member.",
    },
    "Hindi": {
        "is": "अकाल यूनिवर्सिटी के **{role}** **{name}** हैं।",
        "was": "अकाल यूनिवर्सिटी के **{role}** **{name}** थे।",
        "many": "अकाल यूनिवर्सिटी के **{role}**:",
        "p_lead": "**{name}** अकाल यूनिवर्सिटी के {roles} हैं।",
        "p_lead_was": "**{name}** अकाल यूनिवर्सिटी के {roles} थे।",
        "p_fac": "**{name}** अकाल यूनिवर्सिटी में {dept} के {desig} हैं।",
        "p_fac_nodept": "**{name}** अकाल यूनिवर्सिटी में {desig} हैं।",
        "and": " और ",
        "also": "इसके अलावा",
        "email": "ईमेल", "mobile": "मोबाइल", "dept": "विभाग", "qual": "योग्यता", "details": "विवरण",
        "title_note": "ध्यान दें: यूनिवर्सिटी की वेबसाइट के एक और पेज पर यह पदवी “{alt}” लिखी है ({page})।",
        "persona": "ध्यान दें: इस चैट असिस्टेंट का नाम भी “डॉ. सुखजीत सिंह” है। ऊपर की जानकारी यूनिवर्सिटी की "
                   "वेबसाइट पर सूचीबद्ध अधिकारी की है।",
        "missing": "यूनिवर्सिटी की मेरे पास मौजूद आधिकारिक जानकारी में **{role}** की पुष्टि की हुई जानकारी नहीं मिली, "
                   "इसलिए मैं अंदाज़ा नहीं लगाऊँगा। कृपया आधिकारिक गवर्नेंस पेज ({url}) देखें या एडमिशन ऑफिस से पूछें।",
        "list_head": "अकाल यूनिवर्सिटी की आधिकारिक वेबसाइट पर सूचीबद्ध प्रमुख अधिकारी:",
        "list_cols": ["पद", "नाम", "ईमेल"],
        "ambiguous": "आपके सवाल से एक से ज़्यादा लोग मेल खाते हैं। कृपया पूरा नाम या पद बताइए:",
        "fac_head": "**{dept}** की फैकल्टी, जैसी आधिकारिक अकाल फैकल्टी पेज पर सूचीबद्ध है:",
        "fac_missing": "मेरे पास मौजूद आधिकारिक जानकारी में **{dept}** की फैकल्टी की पुष्टि की हुई सूची नहीं मिली, "
                       "इसलिए मैं अंदाज़ा नहीं लगाऊँगा। कृपया आधिकारिक अकाल फैकल्टी पेज ({url}) देखें।",
        "fac_note": "यह वेबसाइट के पेज पर दिखाई गई सूची है; हो सकता है पेज पर हर फैकल्टी सदस्य न दिखे।",
    },
    "Punjabi": {
        "is": "ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਦੇ **{role}** **{name}** ਹਨ।",
        "was": "ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਦੇ **{role}** **{name}** ਸਨ।",
        "many": "ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਦੇ **{role}**:",
        "p_lead": "**{name}** ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਦੇ {roles} ਹਨ।",
        "p_lead_was": "**{name}** ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਦੇ {roles} ਸਨ।",
        "p_fac": "**{name}** ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਵਿੱਚ {dept} ਦੇ {desig} ਹਨ।",
        "p_fac_nodept": "**{name}** ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਵਿੱਚ {desig} ਹਨ।",
        "and": " ਅਤੇ ",
        "also": "ਇਸ ਤੋਂ ਇਲਾਵਾ",
        "email": "ਈਮੇਲ", "mobile": "ਮੋਬਾਈਲ", "dept": "ਵਿਭਾਗ", "qual": "ਯੋਗਤਾ", "details": "ਵੇਰਵਾ",
        "title_note": "ਧਿਆਨ ਦਿਓ: ਯੂਨੀਵਰਸਿਟੀ ਦੀ ਵੈੱਬਸਾਈਟ ਦੇ ਇੱਕ ਹੋਰ ਪੰਨੇ 'ਤੇ ਇਹ ਪਦਵੀ “{alt}” ਲਿਖੀ ਹੈ ({page})।",
        "persona": "ਧਿਆਨ ਦਿਓ: ਇਸ ਚੈਟ ਅਸਿਸਟੈਂਟ ਦਾ ਨਾਮ ਵੀ “ਡਾ. ਸੁਖਜੀਤ ਸਿੰਘ” ਹੈ। ਉੱਪਰ ਦੀ ਜਾਣਕਾਰੀ ਯੂਨੀਵਰਸਿਟੀ ਦੀ "
                   "ਵੈੱਬਸਾਈਟ 'ਤੇ ਸੂਚੀਬੱਧ ਅਧਿਕਾਰੀ ਦੀ ਹੈ।",
        "missing": "ਯੂਨੀਵਰਸਿਟੀ ਦੀ ਮੇਰੇ ਕੋਲ ਮੌਜੂਦ ਅਧਿਕਾਰਤ ਜਾਣਕਾਰੀ ਵਿੱਚ **{role}** ਦੀ ਪੁਸ਼ਟੀ ਕੀਤੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ, "
                   "ਇਸ ਲਈ ਮੈਂ ਅੰਦਾਜ਼ਾ ਨਹੀਂ ਲਗਾਵਾਂਗਾ। ਕਿਰਪਾ ਕਰਕੇ ਅਧਿਕਾਰਤ ਗਵਰਨੈਂਸ ਪੰਨਾ ({url}) ਵੇਖੋ ਜਾਂ ਦਾਖਲਾ ਦਫ਼ਤਰ ਤੋਂ ਪੁੱਛੋ।",
        "list_head": "ਅਕਾਲ ਯੂਨੀਵਰਸਿਟੀ ਦੀ ਅਧਿਕਾਰਤ ਵੈੱਬਸਾਈਟ 'ਤੇ ਸੂਚੀਬੱਧ ਮੁੱਖ ਅਧਿਕਾਰੀ:",
        "list_cols": ["ਅਹੁਦਾ", "ਨਾਮ", "ਈਮੇਲ"],
        "ambiguous": "ਤੁਹਾਡੇ ਸਵਾਲ ਨਾਲ ਇੱਕ ਤੋਂ ਵੱਧ ਵਿਅਕਤੀ ਮੇਲ ਖਾਂਦੇ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਪੂਰਾ ਨਾਮ ਜਾਂ ਅਹੁਦਾ ਦੱਸੋ:",
        "fac_head": "**{dept}** ਦੀ ਫੈਕਲਟੀ, ਜਿਵੇਂ ਅਧਿਕਾਰਤ ਅਕਾਲ ਫੈਕਲਟੀ ਪੰਨੇ 'ਤੇ ਸੂਚੀਬੱਧ ਹੈ:",
        "fac_missing": "ਮੇਰੇ ਕੋਲ ਮੌਜੂਦ ਅਧਿਕਾਰਤ ਜਾਣਕਾਰੀ ਵਿੱਚ **{dept}** ਦੀ ਫੈਕਲਟੀ ਦੀ ਪੁਸ਼ਟੀ ਕੀਤੀ ਸੂਚੀ ਨਹੀਂ ਮਿਲੀ, ਇਸ ਲਈ ਮੈਂ "
                       "ਅੰਦਾਜ਼ਾ ਨਹੀਂ ਲਗਾਵਾਂਗਾ। ਕਿਰਪਾ ਕਰਕੇ ਅਧਿਕਾਰਤ ਅਕਾਲ ਫੈਕਲਟੀ ਪੰਨਾ ({url}) ਵੇਖੋ।",
        "fac_note": "ਇਹ ਵੈੱਬਸਾਈਟ ਦੇ ਪੰਨੇ 'ਤੇ ਦਿਖਾਈ ਗਈ ਸੂਚੀ ਹੈ; ਹੋ ਸਕਦਾ ਹੈ ਪੰਨੇ 'ਤੇ ਹਰ ਫੈਕਲਟੀ ਮੈਂਬਰ ਨਾ ਦਿਖੇ।",
    },
}


def _tx(language):
    return _T.get(language, _T["English"])


def _label(key, language):
    spec = _SPEC_BY_KEY.get(key)
    if not spec:
        return key
    return spec["label"].get(language) or spec["label"]["English"]


def _is_late(name):
    return bool(re.search(r"(?i)\blate\b", name or ""))


# =====================================================================================================
#  ANSWER BUILDING
# =====================================================================================================
def _details_lines(p, language, want_phone, include_notes=True):
    T = _tx(language)
    lines = []
    if p["emails"]:
        lines.append(f"- {T['email']}: " + "; ".join(p["emails"]))
    if want_phone and p["phones"]:
        lines.append(f"- {T['mobile']}: " + "; ".join(p["phones"]))
    if include_notes:
        for note, is_role in p["notes"]:
            lines.append(f"- {T['details'] if is_role else T['qual']}: {note}")
    return lines


def _title_note(p, language):
    """If the official pages write the person's title differently (Prof. here, Dr. there), say so."""
    if len(p["variants"]) < 2:
        return ""
    base = _title_set(p["variants"][0][0])
    for name, source in p["variants"][1:]:
        other = _title_set(name)
        if other and base and other != base:
            return _tx(language)["title_note"].format(alt=name, page=_source_label(source))
    return ""


def _persona_note(p, language):
    pk = {_pkey(t) for t in _name_tokens(PERSONA_NAME)}
    if pk and pk <= set(p["keys"]):
        return _tx(language)["persona"]
    return ""


def _holders(role, depts=None):
    out = []
    for p in _people():
        for d in p["desigs"]:
            if role in d["roles"]:
                if depts and role in _DEPT_ROLES:
                    if not any(_DEPT_RES[x].search(d["dept"] + " " + d["desig"]) for x in depts):
                        continue
                out.append(p)
                break
    return out


def _role_text(roles, language, want_phone, depts):
    T = _tx(language)
    parts, notes, missing, found_any = [], [], [], False
    for role in roles[:6]:
        holders = _holders(role, depts)
        label = _label(role, language)
        if role in _DEPT_ROLES and depts:
            label += " (" + ", ".join(depts) + ")"
        if not holders:
            missing.append(role)
            parts.append(T["missing"].format(role=label, url=GOVERNANCE_URL))
            continue
        found_any = True
        if len(holders) == 1:
            p = holders[0]
            tpl = T["was"] if _is_late(p["name"]) else T["is"]
            block = [tpl.format(role=label, name=p["name"])]
            lines = _details_lines(p, language, want_phone, include_notes=True)
            if lines:
                block.append("\n".join(lines))
            parts.append("\n\n".join(block))
            for n in (_title_note(p, language), _persona_note(p, language)):
                if n and n not in notes:
                    notes.append(n)
        else:
            block = [T["many"].format(role=label)]
            for p in holders:
                lines = _details_lines(p, language, want_phone, include_notes=False)
                block.append(f"- **{p['name']}**" + (" — " + "; ".join(p["emails"]) if p["emails"] else ""))
            parts.append("\n".join(block))
    text = "\n\n".join(parts)
    if notes:
        text += "\n\n" + "\n\n".join(f"*{n}*" for n in notes)
    return text, found_any, missing


def _person_text(p, language, want_phone):
    T = _tx(language)
    lead = [d for d in p["desigs"] if d["roles"]]
    fac = [d for d in p["desigs"] if not d["roles"]]
    head = ""
    if lead:
        names = []
        for d in lead:
            for r in d["roles"][:1]:
                lab = _label(r, language)
                if lab not in names:
                    names.append(lab)
        tpl = T["p_lead_was"] if _is_late(p["name"]) else T["p_lead"]
        head = tpl.format(name=p["name"], roles=T["and"].join(names))
    elif fac:
        d = fac[0]
        if d["dept"]:
            head = T["p_fac"].format(name=p["name"], desig=d["desig"], dept=d["dept"])
        else:
            head = T["p_fac_nodept"].format(name=p["name"], desig=d["desig"])
    lines = []
    if lead and fac:
        lines.append(f"- {T['also']}: " + "; ".join(f"{d['desig']}" + (f", {d['dept']}" if d["dept"] else "") for d in fac))
    lines += _details_lines(p, language, want_phone, include_notes=True)
    text = head + ("\n\n" + "\n".join(lines) if lines else "")
    for n in (_title_note(p, language), _persona_note(p, language)):
        if n:
            text += f"\n\n*{n}*"
    return text


def _list_text(language):
    T = _tx(language)
    rows = []
    for role in _LIST_ORDER:
        for p in _holders(role):
            rows.append((_label(role, language), p["name"], "; ".join(p["emails"]) or "-"))
    if not rows:
        return None
    cols = T["list_cols"]
    table = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        table.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
    return T["list_head"] + "\n\n" + "\n".join(table)


def _faculty_text(dept, language):
    T = _tx(language)
    members = []
    for p in _people():
        for d in p["desigs"]:
            if not d["roles"] and _DEPT_RES[dept].search(d["dept"]):
                members.append((p, d))
                break
    if not members:
        return T["fac_missing"].format(dept=dept, url=FACULTY_URL)
    lines = [T["fac_head"].format(dept=dept), ""]
    for p, d in members:
        lines.append(f"- **{p['name']}** — {d['desig']}")
    lines += ["", f"*{T['fac_note']}*"]
    return "\n".join(lines)


def _ai_block(roles, people, extra_instruction):
    lines = ["[verified-people-directory]",
             "These records come from the University's official website. They are the ONLY source for names, "
             "designations, emails and phone numbers. Never use another name."]
    for p in people:
        desig = "; ".join(d["desig"] + (f" ({d['dept']})" if d["dept"] else "") for d in p["desigs"])
        bits = [p["name"], desig]
        if p["emails"]:
            bits.append("email: " + ", ".join(p["emails"]))
        if p["phones"]:
            bits.append("mobile: " + ", ".join(p["phones"]))
        for note, _ in p["notes"]:
            bits.append(note)
        lines.append("- " + " | ".join(bits))
    lines.append(extra_instruction)
    return "\n".join(lines)


_DUTY_INSTRUCTION = (
    "The student also asks what this office / person does. Describe duties ONLY if other lines of the Context state "
    "them. If they do not, say plainly that the detailed duties are not part of the verified information you have, "
    "and refer to the Akal University Act 2015 / the official website. Never invent duties."
)
_NAME_INSTRUCTION = (
    "Answer the student's question using only these verified records and the rest of the Context. If the question "
    "asks for something that is not stated, say that it is not available in the verified information."
)


# =====================================================================================================
#  THE QUESTION HANDLER
# =====================================================================================================
def _name_keys(text):
    keys = []
    for raw in _TOKEN_RE.findall(text or ""):
        low = raw.lower()
        if low in _STOP_EN or raw in _STOP_NATIVE or low in _HONORIFICS:
            continue
        lat = _translit(raw).lower()
        if len(re.sub(r"[^a-z]", "", lat)) < 2:
            continue
        k = _pkey(raw)
        if len(k) >= 2:
            keys.append(k)
    return keys


def _handle_name(rest, language, want_phone, wants_block):
    people = _people()
    qkeys = _name_keys(rest)
    specific = [k for k in qkeys if k not in _GENERIC_KEYS]
    if not specific:
        return None
    full, partial = [], []
    for p in people:
        hit = sum(1 for k in specific if any(_tok_match(k, pk) for pk in p["keys"]))
        if hit == len(specific):
            full.append(p)
        elif hit:
            exact = any(k in p["keys"] for k in specific)
            if exact:
                partial.append(p)
    if full:
        generic = [k for k in qkeys if k in _GENERIC_KEYS]
        if generic and len(full) > 1:  # "Atul Kumar": the surname narrows the list
            narrowed = [p for p in full if all(any(_tok_match(g, pk) for pk in p["keys"]) for g in generic)]
            full = narrowed or full
        if len(full) == 1:
            p = full[0]
            if wants_block:
                return {"direct": None, "block": _ai_block([], [p], _NAME_INSTRUCTION), "extractive": _person_text(p, language, want_phone),
                        "roles": [], "kind": "person"}
            return {"direct": _person_text(p, language, want_phone), "block": None, "roles": [], "kind": "person"}
        T = _tx(language)
        lines = [T["ambiguous"], ""]
        for p in full[:8]:
            d = p["desigs"][0]
            lines.append(f"- **{p['name']}** — {d['desig']}" + (f", {d['dept']}" if d["dept"] else ""))
        return {"direct": "\n".join(lines), "block": None, "roles": [], "kind": "ambiguous"}
    if partial:  # a name plus other words: the AI answers, but only with the verified records in front of it
        return {"direct": None, "block": _ai_block([], partial[:3], _NAME_INSTRUCTION),
                "extractive": _person_text(partial[0], language, want_phone), "roles": [], "kind": "person"}
    return None


def _handle_text(q, language):
    roles, rest = _roles_in(q, "q")
    want_phone = bool(_PHONE_RE.search(q))
    want_duty = bool(_DUTY_RE.search(q))
    depts = _dept_hits(q)
    if roles:
        text, found_any, _missing = _role_text(roles, language, want_phone, depts)
        if want_duty and found_any:
            holders = []
            for r in roles:
                for p in _holders(r, depts):
                    if p not in holders:
                        holders.append(p)
            return {"direct": None, "block": _ai_block(roles, holders, _DUTY_INSTRUCTION), "extractive": text,
                    "roles": roles, "kind": "role_duty"}
        return {"direct": text, "block": None, "roles": roles, "kind": "role"}
    n_words = len(_TOKEN_RE.findall(q))
    if _LIST_RE.search(q) and (_WHO_RE.search(q) or n_words <= 5):
        text = _list_text(language)
        if text:
            return {"direct": text, "block": None, "roles": [], "kind": "list"}
    if depts and _FACULTY_RE.search(q):
        return {"direct": _faculty_text(depts[0], language), "block": None, "roles": [], "kind": "faculty"}
    res = _handle_name(rest, language, want_phone, wants_block=want_duty)
    return res


def handle(query, language="English", last_query=""):
    """Returns None (not a question about a person / office) or a dict:
         {'direct': text or None, 'block': text for the AI or None, 'extractive': text, 'roles': [...], 'kind': ...}
       'direct' is a complete, verified answer (no AI involved)."""
    q = (query or "").strip()
    if not q or _SELF_RE.search(q) or not _people():
        return None
    res = _handle_text(q, language)
    if res is None and last_query and len(_TOKEN_RE.findall(q)) <= 6 and (_PHONE_RE.search(q) or _FOLLOWUP_RE.search(q)):
        res = _handle_text(f"{last_query} {q}", language)  # "and his email?" after "who is the VC?"
    return res


# =====================================================================================================
#  CHECK OF NAMES IN AN AI ANSWER (zero hallucination)
# =====================================================================================================
_NAME_IN_TEXT_RE = re.compile(
    r"\b(?:Prof(?:essor)?|Dr|Mr|Mrs|Ms|Shri|Smt|Maj(?:or)?)\.?\s+"
    r"((?:[A-Z][A-Za-z.'’-]+\s+){0,3}[A-Z][A-Za-z.'’-]+)"
)


def unsupported_names(answer, reference_text):
    """Person names written after Dr. / Prof. / Mr. ... in `answer` that do not occur in `reference_text`
    (the verified data given to the AI + the student's question). The assistant's own name is always allowed."""
    ref_keys = {_pkey(t) for t in re.findall(r"[A-Za-z]+", reference_text or "")}
    ref_keys |= {_pkey(t) for t in _name_tokens(PERSONA_NAME)}
    ref_keys.discard("")
    bad = []
    for m in _NAME_IN_TEXT_RE.finditer(answer or ""):
        name = m.group(1)
        toks = [t for t in re.findall(r"[A-Za-z]+", name) if len(t) >= 3 and t.lower() not in _HONORIFICS]
        for t in toks:
            if t.lower() in ("the", "and", "for", "who", "has", "was", "with", "from", "this", "that"):
                continue
            k = _pkey(t)
            if len(k) >= 2 and not any(_tok_match(k, r) for r in ref_keys):
                bad.append(name.strip())
                break
    return list(dict.fromkeys(bad))


# =====================================================================================================
#  READING THE OFFICIAL WEBSITE AGAIN  (optional;  python -m utils.people_directory --refresh)
#  Pages are read with requests + BeautifulSoup. Three layouts are understood:
#    1. an HTML table with the columns Name / Position (Designation) / Email,
#    2. blocks like  "Registrar" / "Name: ..." / "Email: ..." / "Mobile: ...",
#    3. the faculty page: "Department of X" / "Dr. Name" / "Assistant Professor" / qualifications.
#  Only records that pass simple checks are written to data/university_officials_live.txt.
# =====================================================================================================
_CRAWL_PAGES = [
    "https://auts.ac.in/governance/",
    "https://auts.ac.in/ugc-mandatory-disclosures/",
    "https://auts.ac.in/about/chancellors-message/",
    "https://auts.ac.in/akal-faculty/",
    "https://auts.ac.in/board-of-studies/",
]
_DESIG_WORDS = re.compile(
    r"(?i)\b(professor|lecturer|head|dean|director|registrar|chancellor|officer|controller|ombudsperson|"
    r"coordinator|incharge|in-charge|librarian|principal|proctor|authority|secretary|president|manager)\b"
)


def _valid_record(desig, name, email):
    toks = _name_tokens(name)
    if not toks or len(" ".join(toks)) < 3 or len(name) > 90:
        return False
    if not desig or len(desig) > 90 or "http" in desig.lower() or "http" in name.lower():
        return False
    if email and not _EMAIL_RE.match(email):
        return False
    return True


def _parse_tables(soup, url):
    out = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        header = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])]
        col = {}
        for i, h in enumerate(header):
            if "name" in h and "name" not in col:
                col["name"] = i
            elif re.search(r"position|designation|post|role", h):
                col["desig"] = i
            elif re.search(r"e-?mail", h):
                col["email"] = i
            elif re.search(r"mobile|phone|contact", h):
                col["phone"] = i
            elif re.search(r"department|dept", h):
                col["dept"] = i
        if "name" not in col or "desig" not in col:
            continue
        for tr in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]

            def cell(key):
                i = col.get(key)
                return cells[i] if i is not None and i < len(cells) else ""

            rec = {"designation": cell("desig"), "name": cell("name"), "email": cell("email"),
                   "phone": cell("phone"), "department": cell("dept"), "note": "", "source": url}
            if _valid_record(rec["designation"], rec["name"], rec["email"]):
                out.append(rec)
    return out


def _parse_label_blocks(lines, url):
    out = []
    for i, ln in enumerate(lines):
        m = re.match(r"(?i)^\s*name\s*[:：]\s*(.+)$", ln)
        if not m:
            continue
        name = m.group(1).strip()
        desig = ""
        for back in range(1, 4):
            if i - back < 0:
                break
            prev = lines[i - back].strip()
            if prev and ":" not in prev and len(prev) <= 70 and not prev.lower().startswith(("click", "profile")):
                desig = prev
                break
        email = phone = ""
        for nxt in lines[i + 1:i + 5]:
            e = re.match(r"(?i)^\s*e-?mail\s*[:：]\s*(\S+)", nxt)
            ph = re.match(r"(?i)^\s*(?:mobile|phone)\s*[:：]\s*(.+)$", nxt)
            if e and not email:
                email = e.group(1).strip().strip(",;")
            if ph and not phone:
                phone = ph.group(1).strip()
        rec = {"designation": desig, "name": name, "email": email, "phone": phone, "department": "", "note": "",
               "source": url}
        if _valid_record(desig, name, email) and _DESIG_WORDS.search(desig):
            out.append(rec)
    return out


def _parse_faculty(lines, url):
    out, dept, i = [], "", 0
    name_re = re.compile(r"^(?:Dr|Prof|Mr|Ms|Mrs|Maj)\b\.?")
    while i < len(lines):
        ln = lines[i].strip()
        if re.match(r"(?i)^(?:department|dept\.?)\s+of\s+", ln) and len(ln) < 80:
            dept = ln
        elif name_re.match(ln) and len(ln) < 80 and dept:
            desig = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if not re.search(r"(?i)professor|lecturer|head|faculty", desig) or len(desig) > 60:
                desig, note = "Faculty member", ""
            else:
                note = lines[i + 2].strip() if i + 2 < len(lines) else ""
                if re.match(r"(?i)^(?:department|dept\.?)\s+of\s+", note) or name_re.match(note) or "|" in note:
                    note = ""
            rec = {"designation": desig, "name": ln, "email": "", "phone": "", "department": dept,
                   "note": note.replace("|", "/")[:200], "source": url}
            if _valid_record(desig, ln, ""):
                out.append(rec)
        i += 1
    return out


def refresh_live(debug=False):
    """Reads the official pages and writes data/university_officials_live.txt. Returns the number of records."""
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (AkalUniversityAdmissionsBot/1.0)"}
    records = []
    for url in _CRAWL_PAGES:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[people] {url} not readable: {str(e)[:100]}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = _parse_tables(soup, url)
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        blocks = _parse_label_blocks(lines, url)
        faculty = _parse_faculty(lines, url) if "faculty" in url else []
        if debug:
            print(f"[people] {url}: {len(tables)} table rows, {len(blocks)} labelled blocks, {len(faculty)} faculty")
        records += tables + blocks + faculty
    seen, out_lines = set(), []
    for r in records:
        key = (r["designation"].lower(), r["name"].lower(), r["source"])
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(" | ".join([r["designation"], r["name"], r["email"], r["phone"], r["department"],
                                     r["note"], r["source"]]))
    if out_lines:
        header = ("# Written by utils/people_directory.py (refresh_live) on " + time.strftime("%d %b %Y %H:%M") +
                  " from the official website. Same format as university_officials.txt.\n")
        (DATA_DIR / LIVE_FILE).write_text(header + "\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[people] {len(out_lines)} records read from the official website")
    return len(out_lines)


def refresh_if_stale(max_age_days=7):
    path = DATA_DIR / LIVE_FILE
    try:
        if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
            return 0
    except OSError:
        pass
    return refresh_live()


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if "--refresh" in argv:
        refresh_live(debug=True)
    if "--list" in argv:
        print(summary())
        for p in _people():
            print("-", p["name"], "|", "; ".join(d["desig"] for d in p["desigs"]), "|", ", ".join(p["emails"]))
    if "--ask" in argv:
        q = argv[argv.index("--ask") + 1]
        for lang in ("English", "Hindi", "Punjabi"):
            r = handle(q, lang)
            print(f"--- {lang} ---")
            print((r or {}).get("direct") or (r or {}).get("block") or "(not a people question)")
