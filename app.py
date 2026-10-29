#app_multiple_quiery
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
from threading import Lock
import pandas as pd

# Hlaðum Excel skjalinu inn við ræsingu appsins
EXCEL_PATH = "VIS01002_20260810-092754.xlsx"

cpi_df = pd.read_excel(EXCEL_PATH, sheet_name='VIS01002')
OLLAMA_URL = "http://localhost:11434/api/generate"
#MODEL_NAME = "hagstofa-ai:latest"
MODEL_NAME = "llama3"
ollama_lock = Lock()

CHROMA_DIR = "./chroma_db_master"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name="hagstofa_master")

try:
    embed_model = SentenceTransformer("BAAI/bge-m3", device="mps")
except:
    embed_model = SentenceTransformer("BAAI/bge-m3")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hingolfur.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Búum til fljótleitar-orðabók (dict) úr Excel skjalinu
cpi_dict = {}
for idx, row in cpi_df.iterrows():
    t_str = str(row.iloc[0]).strip()
    if 'M' in t_str and len(t_str) == 7:  # T.d. '2010M06'
        try:
            val = float(row.iloc[1])
            cpi_dict[t_str] = val
        except ValueError:
            pass

# Global breyta sem geymir spegilmyndina í minni
METADATA_MIRROR = {
    "visitala": {"ars_bil": [], "manudir": []},
    "buferla": {"ars_bil": [], "svaedi": set()}
}

@app.on_event("startup")
def load_metadata_mirror():
    print("⏳ Byrja að hlaða metadata spegilmynd í minni...")
    
    # 1. Spegilmynd fyrir Excel (Vísitölur)
    try:
        ars_listi = set()
        for t_str in cpi_dict.keys():
            ar = t_str[:4]
            ars_listi.add(ar)
        METADATA_MIRROR["visitala"]["ars_bil"] = sorted(list(ars_listi))
        print("✅ Vísitölu-spegilmynd tilbúin.")
    except Exception as e:
        print(f"Villa við vísitölu-spegil: {e}")

    # 2. Spegilmynd fyrir ChromaDB (Búferlaflutningar) - Sækja í bútum
    try:
        ars_set = set()
        svaeda_set = set()
        kynja_set = set()
        
        batch_size, offset = 2000, 0
        while True:
            results = collection.get(limit=batch_size, offset=offset, include=['metadatas'])
            batch_metadatas = results.get('metadatas', [])
            if not batch_metadatas:
                break
                
            for meta in batch_metadatas:
                if meta:
                    timabil = meta.get('timabil') or meta.get('svaedi')
                    if timabil:
                        svaeda_set.add(str(timabil).lower())
                    
                    maeling = str(meta.get('maeling', ''))
                    if len(maeling) >= 4 and maeling[:4].isdigit():
                        ars_set.add(maeling[:4])
                        
                    kyn = meta.get('kyn')
                    if kyn:
                        kynja_set.add(str(kyn).lower())
            
            offset += batch_size
            if len(batch_metadatas) < batch_size:
                break

        METADATA_MIRROR["buferla"] = {
            "ars_bil": sorted(list(ars_set)),
            "svaedi": list(svaeda_set),
            "kyn": list(kynja_set)
        }
        
        print(f"✅ ChromaDB spegilmynd tilbúin. Fann {len(svaeda_set)} svæði og {len(ars_set)} ár.")
    except Exception as e:
        print(f"❌ Villa við ChromaDB spegil: {e}")

    print("🚀 Allar spegilmyndir hlaðnar í RAM!")


def skrifa_svar_med_ai(spurning, founn_texti):
    system_prompt = (
        "Þú ert reynslumikill hagfræðingur og upplýsingafulltrúi hjá Hagstofunni. "
        "Verkefni þitt er að svara spurningum á skýran, mannlegan og faglegan máta út frá gefnum tölum. "
        "Þó að spurning gæti verið á ensku, SKALT þú ávallt svara á lýsandi og vandaðri íslensku. "
        "BREYTTU alltaf orðalagi og setningagerð milli svara svo þau hljómi eðlilega og fjölbreytt. "
        "Ekki nota alltaf sama sniðmátið. Gerðu svarið læsilegt og aldrei skrifa bara eina tölu."
    )
    full_prompt = f"{system_prompt}\n\nSamhengi úr gögnum:\n{founn_texti}\n\nSpurning: {spurning}\nSvar:"
    payload = {"model": MODEL_NAME, "prompt": full_prompt, "stream": False, "options": {"temperature": 0.2}}
    
    try:
        with ollama_lock:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            svar = response.json().get("response", "").strip()
            return svar if svar else founn_texti
        else:
            return founn_texti
    except Exception:
        return founn_texti


def spyrja_beint(spurning):
    spurning_lower = spurning.lower()
    svor = []
    
    # 1. Athugum hvort spurt sé um vísitölu neysluverðs
    if any(w in spurning_lower for w in ["vísitala", "neyslu", "cpi", "price index", "index"]):
        valinn_ar = None
        for ar in METADATA_MIRROR["visitala"]["ars_bil"]:
            if ar in spurning_lower:
                valinn_ar = ar
                break
        
        if valinn_ar:
            manudir_map = {
                "janúar": "01", "januar": "01", "jan": "01", "january": "01",
                "febrúar": "02", "februar": "02", "feb": "02", "february": "02",
                "mars": "03", "mar": "03", "march": "03",
                "apríl": "04", "april": "04", "apr": "04",
                "maí": "05", "mai": "05", "may": "05",
                "júní": "06", "juni": "06", "jun": "06", "june": "06",
                "júlí": "07", "juli": "07", "jul": "07", "july": "07",
                "ágúst": "08", "agust": "08", "aug": "08", "august": "08",
                "september": "09", "sep": "09",
                "október": "10", "oktober": "10", "oct": "10", "october": "10",
                "nóvember": "11", "nov": "11", "november": "11",
                "desember": "12", "dec": "12", "december": "12"
            }
            valinn_manudur = None
            for m_orð, m_nr in manudir_map.items():
                if m_orð in spurning_lower:
                    valinn_manudur = m_nr
                    break
            
            if valinn_manudur:
                lykill = f"{valinn_ar}M{valinn_manudur}"
                gildi = cpi_dict.get(lykill)
                if gildi is not None:
                    svor.append(f"Vísitala neysluverðs fyrir tímabil {lykill} var {gildi} stig.")

    # 2. Búferlaflutningar
    if any(w in spurning_lower for w in ["fluttu", "búferla", "flutning", "karlar", "konur", "male", "woman", "women", "males", "men", "female", "females", "people", "moved", "migration"]):
        valinn_ar = None
        for word in spurning_lower.split():
            clean = word.strip("?.,!")
            if clean.isdigit() and len(clean) == 4:
                valinn_ar = clean

        svæða_listi = ["suðurland", "suðurnes", "austurland", "vesturland", "vestfirðir", 
                        "norðurland", "höfuðborgarsvæði", "norðurland eystra", "norðurland vestra", 
                        "norðurland austur", "norðurland vestur"]

        fundin_svæði = [s for s in svæða_listi if s in spurning_lower]
        
        sudur = ["south", "southern"]
        nordur = ["north", "northern"]
        east = ["east", "eastern"]
        west = ["west", "western"]
        reykjavik_orð = ["reykjavík", "reykjavikur", "reykjavik", "reykjavíkk", "reykjavikk", "reykjarvík", "reykjavíkur", "reikjavík", "capital", "reykjavik"]
        
        if any(r in spurning_lower for r in sudur) and not fundin_svæði:
            fundin_svæði = ["suðurland"]
        if any(r in spurning_lower for r in east) and not fundin_svæði:
            fundin_svæði = ["austurland"]
        if any(r in spurning_lower for r in west) and not fundin_svæði:
            fundin_svæði = ["vesturland"]
        if any(r in spurning_lower for r in nordur) and not fundin_svæði:
            fundin_svæði = ["norðurland"]
        if any(r in spurning_lower for r in reykjavik_orð) and not fundin_svæði:
            fundin_svæði = ["höfuðborgarsvæði"]

        if valinn_ar and fundin_svæði:
            valin_svæði = fundin_svæði[0]
            niðurstöður = {"alls": 0.0, "karlar": 0.0, "konur": 0.0}
            documents, metadatas = [], []
            batch_size, offset = 2000, 0
        
            while True:
                results = collection.get(where={"gagnasett": "buferlaflutningar"}, limit=batch_size, offset=offset, include=['documents', 'metadatas'])
                batch_docs, batch_metas = results.get('documents', []), results.get('metadatas', [])
                if not batch_docs: break
                documents.extend(batch_docs)
                metadatas.extend(batch_metas)
                offset += batch_size
                if len(batch_docs) < batch_size: break

            for doc, meta in zip(documents, metadatas):
                doc_lower = doc.lower()
                if valinn_ar in doc_lower and valin_svæði in doc_lower:
                    try:
                        val = float(str(meta.get('gildi') or meta.get('fjöldi') or '0').replace(',', '.'))
                        if "aldur: alls" in doc_lower or "aldur:alls" in doc_lower:
                            if "alls" in doc_lower and "karl" not in doc_lower and "kon" not in doc_lower: 
                                niðurstöður["alls"] = val
                            elif "karl" in doc_lower and "kon" not in doc_lower: 
                                niðurstöður["karlar"] = val
                            elif ("konur" in doc_lower or "kona" in doc_lower) and "karl" not in doc_lower: 
                                niðurstöður["konur"] = val
                    except ValueError: 
                        pass
        
            if niðurstöður["alls"] == 0: 
                niðurstöður["alls"] = niðurstöður["karlar"] + niðurstöður["konur"]
        
            buferla_texti = f"Heildarfjöldi þeirra sem fluttu frá {valin_svæði.capitalize()} árið {valinn_ar} var {int(niðurstöður['alls'])} (þar af {int(niðurstöður['karlar'])} karlar og {int(niðurstöður['konur'])} konur)."
            svor.append(buferla_texti)

    if svor:
        lokatexti = "\n".join(svor)
        return skrifa_svar_med_ai(spurning, lokatexti)
    else:
        return "Gat ekki túlkað fyrirspurnina eða fundið gögn út frá tiltæku metadata."

 