import streamlit as st
import streamlit.components.v1 as components

from rag_pipeline import (
    APPLY_URL,
    UNIVERSITY_NAME,
    build_index,
    index_exists,
    index_is_stale,
    prepare,
    stream_reply,
    warm_up,
)
from utils.leads import normalize_phone, save_lead, valid_email

st.set_page_config(page_title=f"{UNIVERSITY_NAME} Admissions Assistant", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
        text-align: left;
        white-space: normal;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CONTACT_FOOTER = (
    "\n\n---\n"
    "📞 **Admission Helpline (toll-free, 24x7):** 1800-2020-100\n\n"
    "✉️ **Admission Cell:** admission@auts.ac.in\n\n"
    "✉️ **Dr. Sukhjeet Singh:** sukhjeet_phy@auts.ac.in"
)

BLOG_URL = "https://blog.auts.ac.in/category/campus-life/clubs/"

EVENTS = [
    "🎤 Independence Day Declamation Competition — 170 students spoke on what freedom means to young voices",
    "🎨 Rashtriya Sadbhavna Diwas — students turned ideology into art",
    "🌟 TMTW October — a month of wellness, innovation & compassion",
    "👁️ Free Eye Awareness & Check-Up Camp by the Red Ribbon Club",
    "📚 Dr. Manmohan Singh Chair in Development Economics inaugurated",
    "💻 Empowering Rural Youth Through Innovative Tech Clubs",
    "🏆 Center for Competitive Examination — giving wings to fly high!",
    "🩸 Red Ribbon Club — significant participation in aiding humanity",
    "🤝 NSS — more activities for the betterment of society",
    "⚛️ Physics Association, Psychology Society, Mathrix Society & more active clubs on campus",
]


def render_events_ticker():
    items_html = "".join(
        f'<span class="ticker-item">{e}</span><span class="ticker-sep">&nbsp;&nbsp;★&nbsp;&nbsp;</span>'
        for e in EVENTS
    )
    components.html(
        f"""
        <style>
          .ticker-wrap {{
            width: 100%;
            overflow: hidden;
            background: linear-gradient(90deg, #0b3d91, #1565c0);
            padding: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
          }}
          .ticker-move {{
            display: inline-block;
            white-space: nowrap;
            padding-left: 100%;
            animation: ticker-scroll 45s linear infinite;
          }}
          .ticker-move:hover {{ animation-play-state: paused; }}
          .ticker-item {{
            color: #ffffff;
            font-size: 15px;
            font-weight: 600;
            font-family: 'Segoe UI', Arial, sans-serif;
          }}
          .ticker-sep {{ color: #ffd54f; }}
          @keyframes ticker-scroll {{
            0%   {{ transform: translate3d(0, 0, 0); }}
            100% {{ transform: translate3d(-100%, 0, 0); }}
          }}
          .ticker-label {{
            color: #ffd54f;
            font-weight: 800;
            margin-right: 10px;
          }}
        </style>
        <div class="ticker-wrap">
          <div class="ticker-move">
            <span class="ticker-label">📢 WHAT'S HAPPENING AT AKAL UNIVERSITY:</span>
            {items_html}
          </div>
        </div>
        """,
        height=50,
    )


LANGUAGES = {
    "English": {
        "label": "🇬🇧 English",
        "title": f"{UNIVERSITY_NAME} Admissions Assistant",
        "welcome": "Hey, I am Dr. Sukhjeet Singh. Ask me anything — if I can't answer, then no one can.",
        "caption": "Answers are based on the university's official information. Please confirm fees with the admission office.",
        "input_placeholder": "Ask anything about admissions, fees, hostel, scholarships...",
        "quick_label": "Quick questions",
        "apply_label": "Apply / Register Now",
        "callback_label": "📞 Request a callback",
        "lang_switch_label": "🌐 Language",
        "quick_questions": [
            ("Programs offered?", "What programs do you offer?"),
            ("Fee structure?", "What is the fee structure for all courses?"),
            ("Hostel fees?", "What are the hostel fees for boys and girls?"),
            ("Scholarships?", "What scholarships and fee concessions are available?"),
            ("Eligibility criteria?", "What is the eligibility criteria for admission?"),
            ("Transport fees?", "Is university transport available, and what are the fees?"),
        ],
    },
    "Hindi": {
        "label": "🇮🇳 हिंदी",
        "title": f"{UNIVERSITY_NAME} एडमिशन असिस्टेंट",
        "welcome": "हे, मैं डॉ. सुखजीत सिंह हूं। मुझसे कुछ भी पूछो — अगर मैं जवाब नहीं दे पाया, तो कोई नहीं दे पाएगा।",
        "caption": "जवाब यूनिवर्सिटी की आधिकारिक जानकारी पर आधारित हैं। कृपया फीस एडमिशन ऑफिस से कन्फर्म करें।",
        "input_placeholder": "एडमिशन, फीस, हॉस्टल, स्कॉलरशिप के बारे में कुछ भी पूछें...",
        "quick_label": "क्विक सवाल",
        "apply_label": "अभी अप्लाई करें",
        "callback_label": "📞 कॉलबैक रिक्वेस्ट करें",
        "lang_switch_label": "🌐 भाषा",
        "quick_questions": [
            ("कौन से प्रोग्राम हैं?", "आप कौन से प्रोग्राम ऑफर करते हैं?"),
            ("फीस स्ट्रक्चर?", "सभी कोर्सेज़ की फीस स्ट्रक्चर क्या है?"),
            ("हॉस्टल फीस?", "लड़कों और लड़कियों के लिए हॉस्टल फीस क्या है?"),
            ("स्कॉलरशिप?", "कौन-कौन सी स्कॉलरशिप और फीस छूट उपलब्ध है?"),
            ("एलिजिबिलिटी क्राइटेरिया?", "एडमिशन के लिए एलिजिबिलिटी क्राइटेरिया क्या है?"),
            ("ट्रांसपोर्ट फीस?", "यूनिवर्सिटी ट्रांसपोर्ट उपलब्ध है क्या, और फीस कितनी है?"),
        ],
    },
    "Punjabi": {
        "label": "☬ ਪੰਜਾਬੀ",
        "title": f"{UNIVERSITY_NAME} ਦਾਖਲਾ ਸਹਾਇਕ",
        "welcome": "ਹੇ, ਮੈਂ ਡਾ. ਸੁਖਜੀਤ ਸਿੰਘ ਹਾਂ। ਮੈਨੂੰ ਕੁਝ ਵੀ ਪੁੱਛੋ — ਜੇ ਮੈਂ ਜਵਾਬ ਨਾ ਦੇ ਸਕਿਆ, ਤਾਂ ਕੋਈ ਨਹੀਂ ਦੇ ਸਕਦਾ।",
        "caption": "ਜਵਾਬ ਯੂਨੀਵਰਸਿਟੀ ਦੀ ਸਰਕਾਰੀ ਜਾਣਕਾਰੀ 'ਤੇ ਆਧਾਰਿਤ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਫੀਸ ਦਾਖਲਾ ਦਫ਼ਤਰ ਤੋਂ ਪੱਕੀ ਕਰੋ।",
        "input_placeholder": "ਦਾਖਲਾ, ਫੀਸ, ਹੋਸਟਲ, ਸਕਾਲਰਸ਼ਿਪ ਬਾਰੇ ਕੁਝ ਵੀ ਪੁੱਛੋ...",
        "quick_label": "ਤੁਰੰਤ ਸਵਾਲ",
        "apply_label": "ਹੁਣੇ ਅਪਲਾਈ ਕਰੋ",
        "callback_label": "📞 ਕਾਲਬੈਕ ਦੀ ਬੇਨਤੀ ਕਰੋ",
        "lang_switch_label": "🌐 ਭਾਸ਼ਾ",
        "quick_questions": [
            ("ਕਿਹੜੇ ਪ੍ਰੋਗਰਾਮ ਹਨ?", "ਤੁਸੀਂ ਕਿਹੜੇ ਪ੍ਰੋਗਰਾਮ ਪੇਸ਼ ਕਰਦੇ ਹੋ?"),
            ("ਫੀਸ ਢਾਂਚਾ?", "ਸਾਰੇ ਕੋਰਸਾਂ ਦਾ ਫੀਸ ਢਾਂਚਾ ਕੀ ਹੈ?"),
            ("ਹੋਸਟਲ ਫੀਸ?", "ਮੁੰਡਿਆਂ ਅਤੇ ਕੁੜੀਆਂ ਲਈ ਹੋਸਟਲ ਫੀਸ ਕੀ ਹੈ?"),
            ("ਸਕਾਲਰਸ਼ਿਪ?", "ਕਿਹੜੀਆਂ ਸਕਾਲਰਸ਼ਿਪਾਂ ਅਤੇ ਫੀਸ ਛੋਟ ਉਪਲਬਧ ਹਨ?"),
            ("ਯੋਗਤਾ ਮਾਪਦੰਡ?", "ਦਾਖਲੇ ਲਈ ਯੋਗਤਾ ਮਾਪਦੰਡ ਕੀ ਹੈ?"),
            ("ਟਰਾਂਸਪੋਰਟ ਫੀਸ?", "ਕੀ ਯੂਨੀਵਰਸਿਟੀ ਟਰਾਂਸਪੋਰਟ ਉਪਲਬਧ ਹੈ, ਅਤੇ ਫੀਸ ਕਿੰਨੀ ਹੈ?"),
        ],
    },
}


@st.cache_resource(show_spinner="Loading knowledge base... (first run can take a few minutes)")
def load_models():
    warm_up()
    return True


def ensure_index():
    """Runs on every page load, so edited data files are picked up automatically."""
    if not index_exists() or index_is_stale():
        with st.spinner("Data files changed. Rebuilding knowledge base..."):
            build_index()
        load_models.clear()
    load_models()


try:
    ensure_index()
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------- Events ticker — always shown at the very top ----------------
render_events_ticker()

# ---------------- Step 1: Language selection (shown only once, before chat starts) ----------------
if "language" not in st.session_state:
    st.markdown(f"<h1 style='text-align:center;'>🎓 {UNIVERSITY_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center;'>Please choose your language / कृपया अपनी भाषा चुनें / ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਭਾਸ਼ਾ ਚੁਣੋ</h3>",
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🇬🇧 English", use_container_width=True):
            st.session_state.language = "English"
            st.rerun()
    with col2:
        if st.button("🇮🇳 हिंदी", use_container_width=True):
            st.session_state.language = "Hindi"
            st.rerun()
    with col3:
        if st.button("☬ ਪੰਜਾਬੀ", use_container_width=True):
            st.session_state.language = "Punjabi"
            st.rerun()
    st.stop()

lang_code = st.session_state.language
L = LANGUAGES[lang_code]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": L["welcome"]}]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title(f"🎓 {UNIVERSITY_NAME}")

    new_lang = st.selectbox(
        L["lang_switch_label"],
        options=list(LANGUAGES.keys()),
        format_func=lambda k: LANGUAGES[k]["label"],
        index=list(LANGUAGES.keys()).index(lang_code),
    )
    if new_lang != lang_code:
        st.session_state.language = new_lang
        st.session_state.messages.append(
            {"role": "assistant", "content": LANGUAGES[new_lang]["welcome"]}
        )
        st.rerun()

    st.link_button(L["apply_label"], APPLY_URL, use_container_width=True)
    st.link_button("📰 Campus Events & Clubs Blog", BLOG_URL, use_container_width=True)

    st.markdown(f"**{L['quick_label']}**")
    quick = None
    for label, full_question in L["quick_questions"]:
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            quick = full_question

    with st.expander(L["callback_label"]):
        with st.form("lead_form", clear_on_submit=True):
            name = st.text_input("Name")
            phone = st.text_input("Mobile number")
            email = st.text_input("Email (optional)")
            program = st.text_input("Program you are interested in")
            consent = st.checkbox("I agree to be contacted by the admission team.")
            submitted = st.form_submit_button("Submit")
        if submitted:
            clean_phone = normalize_phone(phone)
            if not name.strip():
                st.error("Please enter your name.")
            elif not clean_phone:
                st.error("Please enter a valid 10-digit mobile number.")
            elif email.strip() and not valid_email(email):
                st.error("Please enter a valid email address.")
            elif not consent:
                st.error("Please tick the consent box.")
            else:
                save_lead(name, clean_phone, email, program)
                st.success("Thank you! Our admission team will contact you soon.")

    st.divider()
    if st.button("🔄 Rebuild knowledge base"):
        with st.spinner("Rebuilding..."):
            n = build_index()
        st.success(f"Done: {n} chunks indexed.")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role": "assistant", "content": L["welcome"]}]
        st.rerun()

# ---------------- Chat ----------------
st.title(L["title"])
st.caption(L["caption"])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input(L["input_placeholder"]) or quick

if prompt:
    history = st.session_state.messages[1:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        messages, sources = prepare(prompt, history, language=lang_code)
        reply = st.write_stream(stream_reply(messages))
        st.markdown(CONTACT_FOOTER)
        if sources:
            st.caption("Sources: " + ", ".join(sources))
    full_reply = reply + CONTACT_FOOTER
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

    components.html(
        """
        <script>
        var doc = window.parent.document;
        var mainSection = doc.querySelector('section[data-testid="stMain"]') || doc.querySelector('section.main');
        if (mainSection) { mainSection.scrollTop = mainSection.scrollHeight; }
        </script>
        """,
        height=0,
    )
