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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=f"{UNIVERSITY_NAME} Admissions Assistant",
    page_icon="🎓",
    layout="wide",
)

# Make sidebar buttons consistent: same height, left-aligned text, full width
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        min-height: 44px;
        white-space: normal;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Shown at the end of every answer
CONTACT_FOOTER = (
    "📞 **Admission Helpline (toll-free, 24x7):** 1800-2020-100\n\n"
    "✉️ **Email:** admission@auts.ac.in"
)

QUICK_QUESTIONS = [
    ("Which programs do you offer?",
     "Which programs do you offer?"),
    ("Fee structure & installments?",
     "Explain the complete fee structure and installments."),
    ("Hostel fees: AC vs non-AC?",
     "What are the hostel fees for AC and non-AC rooms, and how are they split into installments?"),
    ("Scholarships and concessions?",
     "Tell me about scholarships and fee concessions and who is eligible."),
    ("What is the eligibility criteria?",
     "What is the eligibility criteria for admission?"),
    ("How many seats are available?",
     "How many seats are available in each program?"),
]

# =========================
# INIT CHAT
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR (LEFT PANEL)
# =========================
with st.sidebar:
    st.markdown(f"## 🎓 {UNIVERSITY_NAME}")

    st.link_button("Apply / Register Now", APPLY_URL, use_container_width=True)

    st.markdown("### Quick questions")
    for i, (label, question) in enumerate(QUICK_QUESTIONS):
        if st.button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state.prompt = question

    st.markdown("---")
    st.markdown("### 📞 Request a callback")
    st.caption("Leave your details and our admission team will call you.")

    with st.form("lead_form", clear_on_submit=True):
        name = st.text_input("Name")
        phone = st.text_input("Mobile number")
        email = st.text_input("Email (optional)")
        program = st.text_input("Program you are interested in")
        consent = st.checkbox("I agree to be contacted by the admission team.")
        submitted = st.form_submit_button("Submit", use_container_width=True)

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
            try:
                save_lead(name.strip(), clean_phone, email.strip(), program.strip())
            except Exception as e:  # noqa: BLE001
                st.error(f"Sorry, something went wrong while saving your details: {e}")
            else:
                st.success("✅ Thank you! Our admission team will contact you soon.")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if st.button("🔄 Rebuild knowledge base", use_container_width=True):
        with st.spinner("Rebuilding..."):
            n = build_index()
        st.success(f"Done: {n} chunks indexed.")

# =========================
# TITLE (shown immediately)
# =========================
st.title(f"{UNIVERSITY_NAME} Admissions Assistant")
st.caption(
    "Answers come from the university's official information. "
    "Always confirm fees with the admission office."
)


# =========================
# LOAD KNOWLEDGE BASE (once per server start)
# =========================
@st.cache_resource(show_spinner="Getting the assistant ready... (only the first start takes long)")
def init():
    if not index_exists() or index_is_stale():
        build_index()
    warm_up()
    return True


try:
    init()
except ValueError as e:
    st.error(str(e))
    st.stop()

# =========================
# WELCOME MESSAGE
# =========================
st.markdown(f"""
👋 **Hello! 🙏 I'm the {UNIVERSITY_NAME} assistant.**

I can help you with:
- Programs & courses
- Seats availability
- Fees & installments
- Hostel (AC / Non-AC)
- Scholarships & eligibility
- Admission process
""")

# =========================
# CHAT DISPLAY
# =========================
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.divider()
            st.markdown(CONTACT_FOOTER)

# =========================
# INPUT BOX
# =========================
prompt = st.chat_input("Ask anything about admissions, fees, hostel, scholarships...")

# also handle sidebar quick-question clicks
if "prompt" in st.session_state:
    prompt = st.session_state.pop("prompt")

# =========================
# CHAT LOGIC
# =========================
if prompt:
    earlier = list(st.session_state.history)  # messages before this question
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            messages, sources = prepare(prompt, earlier)
        reply = st.write_stream(stream_reply(messages))
        st.divider()
        st.markdown(CONTACT_FOOTER)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.history.append({"role": "assistant", "content": reply})

    # Scroll down automatically so the newest answer is visible without manual scrolling
    components.html(
        "<script>"
        "window.parent.document.querySelector('section.main').scrollTo("
        "0, window.parent.document.querySelector('section.main').scrollHeight);"
        "</script>",
        height=0,
    )
