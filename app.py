import time

import streamlit as st
import streamlit.components.v1 as components

# CHANGED: import from rag_pipeline_v4 (the new file), plus OFFICIAL_WEBSITE_URL and render_social_bar
from rag_pipeline import(
    APPLY_URL,
    OFFICIAL_WEBSITE_URL,
    UNIVERSITY_NAME,
    build_index,
    index_exists,
    index_is_stale,
    prepare,
    render_social_bar,
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


def render_voice_input(lang_code):
    """A floating 🎤 mic button placed just left of st.chat_input's send arrow. On click it listens with
    the browser's built-in speech recognition (language matched to the current UI language), writes the
    recognized text into the chat input box, and lets the student review/edit it before pressing send."""
    speech_lang = {"English": "en-IN", "Hindi": "hi-IN", "Punjabi": "pa-IN"}.get(lang_code, "en-IN")
    components.html(
        f"""
        <button id="akal-mic-btn" style="display:none;">🎤</button>
        <script>
        (function() {{
            var doc = window.parent.document;
            var speechLang = "{speech_lang}";

            function trySetup(retriesLeft) {{
                var chatInput = doc.querySelector('[data-testid="stChatInput"]');
                if (!chatInput) {{
                    if (retriesLeft > 0) setTimeout(function() {{ trySetup(retriesLeft - 1); }}, 300);
                    return;
                }}

                var btn = doc.getElementById('akal-mic-btn-clone');
                if (!btn) {{
                    btn = document.getElementById('akal-mic-btn').cloneNode(true);
                    btn.id = 'akal-mic-btn-clone';
                    btn.style.display = 'flex';
                    btn.style.position = 'fixed';
                    btn.style.width = '36px';
                    btn.style.height = '36px';
                    btn.style.borderRadius = '50%';
                    btn.style.border = 'none';
                    btn.style.background = '#e63946';
                    btn.style.color = '#fff';
                    btn.style.fontSize = '16px';
                    btn.style.alignItems = 'center';
                    btn.style.justifyContent = 'center';
                    btn.style.cursor = 'pointer';
                    btn.style.zIndex = '999999';
                    btn.style.boxShadow = '0 1px 4px rgba(0,0,0,0.3)';
                    doc.body.appendChild(btn);
                }}

                function positionBtn() {{
                    var rect = chatInput.getBoundingClientRect();
                    btn.style.top = (rect.top + rect.height / 2 - 18) + 'px';
                    btn.style.left = (rect.right - 100) + 'px';
                }}
                positionBtn();
                if (!window.__akalMicResizeBound) {{
                    window.parent.addEventListener('resize', positionBtn);
                    window.__akalMicResizeBound = true;
                }}
                if (window.__akalMicPositionInterval) clearInterval(window.__akalMicPositionInterval);
                window.__akalMicPositionInterval = setInterval(positionBtn, 1000);

                var Recognition = window.webkitSpeechRecognition || window.SpeechRecognition;
                if (!Recognition) {{
                    btn.title = "Voice input is not supported in this browser";
                    btn.style.opacity = '0.4';
                    return;
                }}
                var recognition = new Recognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                var listening = false;

                btn.onclick = function() {{
                    if (listening) {{ recognition.stop(); return; }}
                    recognition.lang = speechLang;
                    try {{ recognition.start(); }} catch (e) {{}}
                }};
                recognition.onstart = function() {{
                    listening = true;
                    btn.style.background = '#2a9d8f';
                    btn.innerHTML = '●';
                }};
                recognition.onend = function() {{
                    listening = false;
                    btn.style.background = '#e63946';
                    btn.innerHTML = '🎤';
                }};
                recognition.onerror = function() {{
                    listening = false;
                    btn.style.background = '#e63946';
                    btn.innerHTML = '🎤';
                }};
                recognition.onresult = function(event) {{
                    var text = event.results[0][0].transcript;
                    var textarea = chatInput.querySelector('textarea');
                    if (!textarea) return;
                    var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    nativeSetter.call(textarea, text);
                    textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    textarea.focus();
                }};
            }}
            trySetup(20);
        }})();
        </script>
        """,
        height=0,
    )


def scroll_chat_to_bottom(follow):
    """Scrolls the chat down to the newest message.

    follow=True  -> call it as soon as a question is sent: jumps to the bottom at once and keeps following
                    the answer while it is being written. It stops following the moment the student
                    scrolls, clicks or presses a key, so someone reading can still read in peace.
    follow=False -> call it when the answer is complete: one last jump to the bottom (skipped if the
                    student scrolled away while the answer was being written).

    Why scrolling failed before: the page only scrolled by itself while the student was already at the
    very bottom. This helper finds whichever container really scrolls (it differs between Streamlit
    versions) and the HTML below contains a number that is different on EVERY run, so the browser
    always reloads and re-runs the script instead of silently reusing the previous run's copy."""
    components.html(
        f"""
        <!-- run {time.time_ns()} -->
        <script>
        (function() {{
            var win = window.parent, doc = win.document;
            var FOLLOW = {"true" if follow else "false"};

            function scrollers() {{
                var found = [];
                var msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
                var node = msgs.length ? msgs[msgs.length - 1] : null;
                while (node && node !== doc.body) {{
                    var style = win.getComputedStyle(node);
                    if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {{
                        found.push(node);
                    }}
                    node = node.parentElement;
                }}
                var main = doc.querySelector('[data-testid="stMain"]') || doc.querySelector('section.main');
                if (main && found.indexOf(main) === -1) found.push(main);
                if (doc.scrollingElement && found.indexOf(doc.scrollingElement) === -1) found.push(doc.scrollingElement);
                return found;
            }}

            function toBottom() {{
                try {{
                    var msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
                    if (msgs.length) msgs[msgs.length - 1].scrollIntoView({{block: "end"}});
                }} catch (e) {{}}
                try {{
                    scrollers().forEach(function(el) {{ el.scrollTop = el.scrollHeight; }});
                }} catch (e) {{}}
            }}

            if (FOLLOW) {{
                win.__akalFollow = true;
                var stopFollowing = function() {{ win.__akalFollow = false; }};
                ['wheel', 'touchmove', 'keydown', 'mousedown'].forEach(function(name) {{
                    doc.addEventListener(name, stopFollowing, {{passive: true, once: true}});
                }});
                toBottom();
                var ticks = 0;
                var timer = setInterval(function() {{
                    ticks += 1;
                    if (win.__akalFollow === false || ticks > 300) {{ clearInterval(timer); return; }}
                    toBottom();
                }}, 250);
            }} else {{
                if (win.__akalFollow === false) return;
                toBottom();
                [150, 400, 800, 1500].forEach(function(ms) {{
                    setTimeout(function() {{ if (win.__akalFollow !== false) toBottom(); }}, ms);
                }});
            }}
        }})();
        </script>
        """,
        height=0,
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

# ---------------- Social media bar (Facebook / Twitter / YouTube / Instagram) — very top, top-left ----------------
render_social_bar()

# ---------------- Events ticker — always shown at the top ----------------
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

# Floating mic button (voice typing), positioned just left of the chat input's send arrow.
render_voice_input(lang_code)

if prompt:
    scroll_chat_to_bottom(follow=True)  # jump down at once, even if the student had scrolled up
    history = st.session_state.messages[1:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        messages, sources = prepare(prompt, history, language=lang_code)
        reply = st.write_stream(stream_reply(messages))
        st.markdown(CONTACT_FOOTER)
        if sources:  # CHANGED: the ONLY source shown is the official Akal University website
            st.caption(f"Source: [Akal University Official Website]({OFFICIAL_WEBSITE_URL})")
    full_reply = reply + CONTACT_FOOTER
    st.session_state.messages.append({"role": "assistant", "content": full_reply})

    scroll_chat_to_bottom(follow=False)
