import streamlit as st
import requests

st.set_page_config(page_title="Company Matcher", page_icon="🏢", layout="centered")

st.title("🔍 Semantic Company Matcher")
st.markdown("Căutare inteligentă bazată pe intenție (LLM Baseline).")

BACKEND_URL = "http://localhost:5000/api/search"

# Input pentru query
query = st.text_input("Ce fel de companii cauți?", 
                      placeholder="ex: Vreau o companie din Germania cu peste 1000 de angajati")

if st.button("Caută Companii", type="primary"):
    if query:
        with st.spinner("Analizez fiecare companie din baza de date... "):
            try:
                response = requests.post(BACKEND_URL, json={"query": query})
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        st.warning("Nu am găsit nicio companie care să depășească scorul de 50.")
                    else:
                        st.success(f"Am găsit {len(results)} companii relevante!")
                        
                        for item in results:
                            score = item['score']
                            company = item['company']
                            
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.subheader(company.get('operational_name', 'N/A'))
                                    st.caption(f"📍 {company.get('address', 'N/A')} | 👥 Angajați: {company.get('employee_count', 'N/A')}")
                                    st.write(company.get('description', 'N/A'))
                                    
                                with col2:
                                    # Afișare vizuală a scorului
                                    st.metric(label="Match Score", value=f"{score}/100")
                                    st.progress(score / 100.0)
                                    
                                st.divider()
                else:
                    st.error("Eroare la comunicarea cu backend-ul.")
            except requests.exceptions.ConnectionError:
                st.error("Nu mă pot conecta la backend. Asigură-te că serverul Flask rulează la :5000.")
    else:
        st.warning("Te rog să introduci un query.")