"""
data_generation/generate_synthetic_dataset.py

Generates 150,000+ diverse synthetic clinical sentences with BIO-tagged PHI.
Key improvements over v1:
  - 100+ templates covering varied real-world clinical language
  - Large Indian name, hospital, city pools
  - Realistic Indian phone / ID / date formats
  - Diverse clinical contexts: OPD, IPD, emergency, lab, pharmacy, insurance
  - Negative sentences (no PHI) to reduce false positives
  - Context word variation to prevent pattern memorization
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faker import Faker
from config.config import SYNTHETIC_SENTENCES, DATA_DIR

fake = Faker("en_IN")
random.seed(42)

# ── Name pools ────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Ravi","Priya","Arun","Sneha","Vikram","Anita","Suresh","Kavya","Rahul","Meena",
    "Amit","Deepa","Sanjay","Pooja","Rajesh","Sunita","Manoj","Rekha","Ajay","Nisha",
    "Vijay","Divya","Naveen","Lakshmi","Kiran","Anjali","Ramesh","Geeta","Ganesh","Usha",
    "Harish","Radha","Sunil","Saroja","Mohan","Savitha","Prakash","Sumathi","Prasad","Latha",
    "Venkat","Padma","Srinivas","Vani","Bhaskar","Shanthi","Naresh","Kamala","Ranjit","Sudha",
    "Dinesh","Malathi","Girish","Bhavana","Santosh","Hema","Anil","Jyothi","Mahesh","Geetha",
    "Ashok","Vijayalakshmi","Sudhir","Revathi","Raghu","Bharati","Girija","Rohit","Nandini",
    "Manohar","Saraswathi","Krishnamurthy","Ambika","Shivaraj","Mangala","Balasubramanian",
    "Mohammed","Ayesha","Imran","Fathima","Arjun","Bindhu","Aravind","Chitra","Shiva","Rani",
    "Gopal","Saranya","Hemant","Nithya","Tushar","Archana","Sachin","Reshma","Nikhil","Swathi",
]
LAST_NAMES = [
    "Kumar","Sharma","Reddy","Singh","Nair","Pillai","Iyer","Rao","Patel","Gupta",
    "Joshi","Mishra","Verma","Shah","Mehta","Desai","Bhat","Hegde","Murthy","Naidu",
    "Shetty","Kamath","Menon","Krishnan","Rajan","Subramaniam","Chandrasekhar","Venkatesh",
    "Agarwal","Banerjee","Chatterjee","Mukherjee","Das","Ghosh","Bose","Sen","Dey","Roy",
    "Khan","Ansari","Siddiqui","Shaikh","Malik","Qureshi","Hussain","Ali","Ahmed","Mirza",
    "Chowdhury","Kapoor","Malhotra","Khanna","Chopra","Bhatia","Sethi","Arora","Anand",
    "Pandey","Tiwari","Dubey","Bajpai","Chaturvedi","Tripathi","Awasthi","Srivastava",
    "Patil","Kulkarni","Jadhav","Shinde","Pawar","Bhosale","Gaikwad","Deshpande","More",
    "Nambiar","Kurup","Varghese","Thomas","Mathew","George","Philip","Jacob","Abraham",
]
TITLES = ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Shri", "Smt.", ""]

DOCTOR_FIRST = [
    "Rajesh","Priya","Anand","Meera","Suresh","Kavitha","Vikram","Deepika","Arun","Sunita",
    "Ramesh","Usha","Ganesh","Saroja","Harish","Radha","Srinivas","Vani","Bhaskar","Shanthi",
    "Naresh","Kamala","Dinesh","Malathi","Girish","Bhavana","Santosh","Hema","Anil","Jyothi",
    "Mahesh","Geetha","Ashok","Sudhir","Revathi","Raghu","Bharati","Rohit","Nandini","Manohar",
    "Abdul","Faisal","Sameer","Zoya","Arvind","Poornima","Balaji","Vasantha","Chandran","Meenakshi",
]
DOCTOR_LAST = [
    "Kumar","Sharma","Reddy","Singh","Nair","Pillai","Iyer","Rao","Patel","Gupta",
    "Joshi","Mishra","Verma","Shah","Mehta","Desai","Bhat","Hegde","Murthy","Naidu",
    "Krishnan","Rajan","Subramaniam","Venkatesh","Banerjee","Chatterjee","Mukherjee",
    "Khan","Ansari","Malik","Hussain","Ahmed","Chowdhury","Kapoor","Malhotra","Khanna",
    "Pandey","Tiwari","Dubey","Chaturvedi","Patil","Kulkarni","Jadhav","Shinde","Pawar",
]
DOCTOR_SUFFIXES = ["MD", "MBBS", "MS", "DM", "DNB", "MCh", "FRCS", "PhD", ""]

HOSPITALS = [
    "Apollo Hospital","Fortis Hospital","AIIMS","Manipal Hospital","Max Hospital",
    "Narayana Health","Medanta Hospital","Kokilaben Hospital","Lilavati Hospital","NIMHANS",
    "Christian Medical College","Tata Memorial Hospital","PGIMER","Amrita Hospital",
    "Sri Ramachandra Institute","Global Hospital","Aster CMI Hospital","Columbia Asia Hospital",
    "Ruby Hall Clinic","Breach Candy Hospital","Hinduja Hospital","Jaslok Hospital",
    "Wockhardt Hospital","Yashoda Hospital","Care Hospital","Sunshine Hospital",
    "Rainbow Hospital","Lotus Hospital","Sparsh Hospital","Sakra World Hospital",
    "Vikram Hospital","BGS Gleneagles Hospital","Sagar Hospital","Mallya Hospital",
    "Bangalore Baptist Hospital","St. John's Medical College Hospital","Victoria Hospital",
    "District Hospital","Government General Hospital","ESI Hospital","Railway Hospital",
    "Rajiv Gandhi Government Hospital","Bowring and Lady Curzon Hospital","KR Hospital",
    "St. Martha's Hospital","Kidwai Memorial Institute of Oncology","Nimhans Hospital",
    "SDM Hospital","KMC Hospital","Kasturba Hospital","Wenlock Hospital",
]

LOCATIONS = [
    "Bangalore","Mumbai","Delhi","Chennai","Hyderabad","Kolkata","Pune","Ahmedabad",
    "Jaipur","Lucknow","Chandigarh","Kochi","Coimbatore","Nagpur","Bhopal","Indore",
    "Mysore","Surat","Vadodara","Patna","Thiruvananthapuram","Visakhapatnam","Madurai",
    "Nashik","Aurangabad","Rajkot","Mangalore","Hubli","Dharwad","Bellary","Tumkur",
    "Shivamogga","Davanagere","Gulbarga","Bijapur","Hassan","Mandya","Udupi","Bagalkot",
    "North Bangalore","South Delhi","East Mumbai","West Chennai","Central Hyderabad",
    "Whitefield","Electronic City","Koramangala","Indiranagar","Jayanagar","Malleswaram",
    "Rajajinagar","Hebbal","Yelahanka","JP Nagar","BTM Layout","HSR Layout","Marathahalli",
]

DIAGNOSES = [
    "type 2 diabetes mellitus","hypertension","coronary artery disease","chest pain",
    "chronic kidney disease","hypothyroidism","asthma","COPD","rheumatoid arthritis",
    "osteoporosis","anemia","atrial fibrillation","deep vein thrombosis","pneumonia",
    "urinary tract infection","migraine","epilepsy","stroke","acute myocardial infarction",
    "heart failure","dengue fever","typhoid","malaria","tuberculosis","COVID-19",
    "appendicitis","gallstones","kidney stones","herniated disc","fracture of femur",
    "diabetic neuropathy","diabetic retinopathy","chronic liver disease","cirrhosis",
    "peptic ulcer disease","gastroesophageal reflux disease","irritable bowel syndrome",
    "hypothyroidism","hyperthyroidism","polycystic ovary syndrome","endometriosis",
    "breast cancer","lung cancer","prostate cancer","cervical cancer","oral cancer",
    "depression","anxiety disorder","schizophrenia","bipolar disorder","dementia",
]

DEPARTMENTS = [
    "Cardiology","Neurology","Orthopedics","Oncology","Dermatology","Endocrinology",
    "Gastroenterology","Nephrology","Pulmonology","Pediatrics","Gynecology","Urology",
    "Ophthalmology","ENT","Psychiatry","Radiology","Pathology","Anesthesiology",
    "General Surgery","Plastic Surgery","Neurosurgery","Vascular Surgery","Thoracic Surgery",
    "Emergency Medicine","Internal Medicine","Family Medicine","Geriatrics","Rheumatology",
    "Hematology","Immunology","Infectious Disease","Palliative Care","Rehabilitation",
]

WARD_TYPES = [
    "general ward","ICU","NICU","PICU","CCU","HDU","emergency ward",
    "surgical ward","medical ward","maternity ward","pediatric ward","isolation ward",
]

PROCEDURES = [
    "ECG","echocardiogram","MRI scan","CT scan","X-ray","ultrasound","endoscopy",
    "colonoscopy","bronchoscopy","biopsy","blood transfusion","dialysis","chemotherapy",
    "radiotherapy","angiography","angioplasty","bypass surgery","appendectomy",
    "cholecystectomy","hysterectomy","cesarean section","laparoscopy","cataract surgery",
    "knee replacement","hip replacement","spinal fusion","coronary artery bypass graft",
]

MEDICINES = [
    "Metformin 500mg","Amlodipine 5mg","Atorvastatin 40mg","Losartan 50mg",
    "Aspirin 75mg","Pantoprazole 40mg","Levothyroxine 50mcg","Insulin Glargine 10 units",
    "Azithromycin 500mg","Amoxicillin 500mg","Ciprofloxacin 500mg","Paracetamol 500mg",
    "Ibuprofen 400mg","Omeprazole 20mg","Atenolol 50mg","Ramipril 5mg","Furosemide 40mg",
    "Spironolactone 25mg","Clopidogrel 75mg","Warfarin 5mg","Prednisolone 10mg",
]

CONTEXT_PHRASES = [
    "as per doctor advice","for further examination","according to hospital records",
    "urgently","for consultation","for treatment","immediately","as per protocol",
    "for follow-up","after review","post-surgery","for monitoring","as prescribed",
    "on referral","under observation","for second opinion","after examination",
    "following the procedure","as directed","per medical advice",
]

# ── Generator helpers ─────────────────────────────────────────────────────────

def rname():
    title = random.choice(TITLES)
    name  = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    return (f"{title} {name}".strip() if title else name)

def rdoctor():
    suf = random.choice(DOCTOR_SUFFIXES)
    name = f"{random.choice(DOCTOR_FIRST)} {random.choice(DOCTOR_LAST)}"
    return (f"{name}, {suf}" if suf else name)

def rhospital(): return random.choice(HOSPITALS)
def rlocation(): return random.choice(LOCATIONS)
def rdept():     return random.choice(DEPARTMENTS)
def rdiag():     return random.choice(DIAGNOSES)
def rmed():      return random.choice(MEDICINES)
def rproc():     return random.choice(PROCEDURES)
def rward():     return random.choice(WARD_TYPES)
def rctx():      return random.choice(CONTEXT_PHRASES)

def rdate():
    d = fake.date_between(start_date="-10y", end_date="today")
    fmt = random.choice([
        "%d %B %Y", "%d/%m/%Y", "%B %d, %Y", "%d-%b-%Y",
        "%d.%m.%Y", "%d %b %Y", "%Y-%m-%d",
    ])
    return d.strftime(fmt)

def rphone():
    digits = "".join([str(random.randint(0,9)) for _ in range(10)])
    # Ensure starts with 6-9 (valid Indian mobile)
    digits = str(random.randint(6,9)) + digits[1:]
    return random.choice([
        f"+91 {digits[:5]} {digits[5:]}",
        f"+91-{digits[:5]}-{digits[5:]}",
        f"0{digits[:4]}-{digits[4:]}",
        f"+91 {digits}",
        digits,
    ])

def remail():
    fn = random.choice(FIRST_NAMES).lower()
    ln = random.choice(LAST_NAMES).lower()
    num = random.randint(1, 999)
    domain = random.choice(["gmail.com","yahoo.com","hotmail.com","outlook.com","rediffmail.com"])
    pattern = random.choice([
        f"{fn}.{ln}@{domain}",
        f"{fn}{ln}{num}@{domain}",
        f"{fn}_{ln}@{domain}",
        f"{fn}{num}@{domain}",
        f"{ln}.{fn}@{domain}",
    ])
    return pattern

def rid():
    prefix = random.choice(["MRN","PID","UHID","OPD","IPD","REG","OP","IP","HN","CR"])
    num    = random.randint(10000, 9999999)
    return f"{prefix}{num}"


# ── Tokenizer + BIO tagger ────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r"\S+", text)

def bio_tag(tokens, entities):
    """entities = list of (entity_type, span_text)"""
    sentence = " ".join(tokens)
    char_tags = ["O"] * len(sentence)

    for etype, span in entities:
        span = span.strip()
        start = 0
        while True:
            idx = sentence.find(span, start)
            if idx == -1: break
            char_tags[idx] = f"B-{etype}"
            for ci in range(idx+1, idx+len(span)):
                if sentence[ci] != " ":
                    char_tags[ci] = f"I-{etype}"
            start = idx + len(span)

    tags = []
    pos  = 0
    for tok in tokens:
        i = sentence.find(tok, pos)
        if i == -1: tags.append("O"); pos += len(tok)+1; continue
        tags.append(char_tags[i])
        pos = i + len(tok)
    return tags


# ── Templates (100+ diverse clinical sentence patterns) ──────────────────────

def make_sentence():
    """Randomly pick a template and fill it. Returns (sentence, entities)."""
    tid = random.randint(1, 110)

    # ── OPD / Visit templates ─────────────────────────────────────────────
    if tid == 1:
        n,h,c = rname(),rhospital(),rlocation()
        return f"Patient {n} visited {h} in {c}.", [("NAME",n),("HOSPITAL",h),("LOCATION",c)]
    if tid == 2:
        n,h,d = rname(),rhospital(),rdate()
        return f"{n} was admitted to {h} on {d}.", [("NAME",n),("HOSPITAL",h),("DATE",d)]
    if tid == 3:
        n,h,c,d = rname(),rhospital(),rlocation(),rdate()
        return f"Patient {n} reported to {h}, {c} on {d}.", [("NAME",n),("HOSPITAL",h),("LOCATION",c),("DATE",d)]
    if tid == 4:
        n,h = rname(),rhospital()
        ctx = rctx()
        return f"{n} was brought to {h} {ctx}.", [("NAME",n),("HOSPITAL",h)]
    if tid == 5:
        n,h,c = rname(),rhospital(),rlocation()
        return f"The patient {n} presented at {h} located in {c}.", [("NAME",n),("HOSPITAL",h),("LOCATION",c)]
    if tid == 6:
        n,h,d = rname(),rhospital(),rdate()
        diag  = rdiag()
        return f"{n} was admitted to {h} on {d} with {diag}.", [("NAME",n),("HOSPITAL",h),("DATE",d)]
    if tid == 7:
        n,h,d = rname(),rhospital(),rdate()
        return f"Discharge of patient {n} from {h} on {d} was completed.", [("NAME",n),("HOSPITAL",h),("DATE",d)]
    if tid == 8:
        n,c = rname(),rlocation()
        return f"{n} from {c} was referred for specialist care.", [("NAME",n),("LOCATION",c)]
    if tid == 9:
        n,c,h = rname(),rlocation(),rhospital()
        return f"{n} traveled from {c} to {h} for treatment.", [("NAME",n),("LOCATION",c),("HOSPITAL",h)]
    if tid == 10:
        n,h,w = rname(),rhospital(),rward()
        return f"Patient {n} has been shifted to the {w} at {h}.", [("NAME",n),("HOSPITAL",h)]

    # ── Doctor templates ──────────────────────────────────────────────────
    if tid == 11:
        doc,n,d = rdoctor(),rname(),rdate()
        return f"Dr {doc} examined {n} on {d}.", [("DOCTOR",doc),("NAME",n),("DATE",d)]
    if tid == 12:
        doc,n,h = rdoctor(),rname(),rhospital()
        return f"Dr {doc} is the treating physician for {n} at {h}.", [("DOCTOR",doc),("NAME",n),("HOSPITAL",h)]
    if tid == 13:
        doc,dept,h = rdoctor(),rdept(),rhospital()
        return f"Dr {doc} heads the {dept} department at {h}.", [("DOCTOR",doc),("HOSPITAL",h)]
    if tid == 14:
        doc,n,proc = rdoctor(),rname(),rproc()
        return f"Dr {doc} performed {proc} on patient {n}.", [("DOCTOR",doc),("NAME",n)]
    if tid == 15:
        doc,n,d,h = rdoctor(),rname(),rdate(),rhospital()
        return f"Consultation note by Dr {doc} for {n} at {h} on {d}.", [("DOCTOR",doc),("NAME",n),("HOSPITAL",h),("DATE",d)]
    if tid == 16:
        doc,n,med = rdoctor(),rname(),rmed()
        return f"Dr {doc} prescribed {med} to {n}.", [("DOCTOR",doc),("NAME",n)]
    if tid == 17:
        doc,n,diag = rdoctor(),rname(),rdiag()
        return f"Dr {doc} diagnosed {n} with {diag}.", [("DOCTOR",doc),("NAME",n)]
    if tid == 18:
        doc,n,d = rdoctor(),rname(),rdate()
        return f"Follow-up with Dr {doc} for patient {n} is scheduled on {d}.", [("DOCTOR",doc),("NAME",n),("DATE",d)]
    if tid == 19:
        doc,h,c = rdoctor(),rhospital(),rlocation()
        return f"Referral from Dr {doc} at {h} in {c}.", [("DOCTOR",doc),("HOSPITAL",h),("LOCATION",c)]
    if tid == 20:
        doc,n = rdoctor(),rname()
        return f"The attending doctor Dr {doc} reviewed {n}'s case.", [("DOCTOR",doc),("NAME",n)]
    if tid == 21:
        doc,n,d = rdoctor(),rname(),rdate()
        return f"Dr {doc} cleared {n} for discharge on {d}.", [("DOCTOR",doc),("NAME",n),("DATE",d)]
    if tid == 22:
        doc1,doc2,n = rdoctor(),rdoctor(),rname()
        return f"Dr {doc1} referred {n} to Dr {doc2} for further evaluation.", [("DOCTOR",doc1),("NAME",n),("DOCTOR",doc2)]
    if tid == 23:
        doc,dept = rdoctor(),rdept()
        return f"Dr {doc} is a consultant in {dept}.", [("DOCTOR",doc)]
    if tid == 24:
        doc,n,proc,d = rdoctor(),rname(),rproc(),rdate()
        return f"Dr {doc} scheduled {proc} for {n} on {d}.", [("DOCTOR",doc),("NAME",n),("DATE",d)]
    if tid == 25:
        doc,n,h,d = rdoctor(),rname(),rhospital(),rdate()
        return f"Case summary prepared by Dr {doc} for {n} admitted at {h} on {d}.", [("DOCTOR",doc),("NAME",n),("HOSPITAL",h),("DATE",d)]

    # ── Date templates ────────────────────────────────────────────────────
    if tid == 26:
        n,d = rname(),rdate()
        return f"Follow-up appointment for {n} on {d}.", [("NAME",n),("DATE",d)]
    if tid == 27:
        n,d = rname(),rdate()
        return f"Date of birth of patient {n} is {d}.", [("NAME",n),("DATE",d)]
    if tid == 28:
        n,d1,d2 = rname(),rdate(),rdate()
        return f"{n} was hospitalized from {d1} to {d2}.", [("NAME",n),("DATE",d1),("DATE",d2)]
    if tid == 29:
        d,n = rdate(),rname()
        return f"On {d}, {n} underwent emergency surgery.", [("DATE",d),("NAME",n)]
    if tid == 30:
        n,d = rname(),rdate()
        return f"Test results for {n} collected on {d} are enclosed.", [("NAME",n),("DATE",d)]
    if tid == 31:
        pid,d = rid(),rdate()
        return f"Patient ID {pid} was registered on {d}.", [("ID",pid),("DATE",d)]
    if tid == 32:
        n,d,h = rname(),rdate(),rhospital()
        return f"Appointment scheduled for {n} at {h} on {d}.", [("NAME",n),("HOSPITAL",h),("DATE",d)]
    if tid == 33:
        d,doc,n = rdate(),rdoctor(),rname()
        return f"On {d}, Dr {doc} reviewed {n}'s reports.", [("DATE",d),("DOCTOR",doc),("NAME",n)]

    # ── Phone templates ───────────────────────────────────────────────────
    if tid == 34:
        n,p = rname(),rphone()
        return f"Contact {n} at {p} for follow-up.", [("NAME",n),("PHONE",p)]
    if tid == 35:
        n,p = rname(),rphone()
        return f"Emergency contact number for {n} is {p}.", [("NAME",n),("PHONE",p)]
    if tid == 36:
        h,p = rhospital(),rphone()
        return f"For appointments at {h}, call {p}.", [("HOSPITAL",h),("PHONE",p)]
    if tid == 37:
        n,p,e = rname(),rphone(),remail()
        return f"Patient {n} can be reached at {p} or {e}.", [("NAME",n),("PHONE",p),("EMAIL",e)]
    if tid == 38:
        p = rphone()
        return f"For medical emergencies call {p}.", [("PHONE",p)]
    if tid == 39:
        n,p,c = rname(),rphone(),rlocation()
        return f"{n} from {c} provided contact number {p}.", [("NAME",n),("LOCATION",c),("PHONE",p)]
    if tid == 40:
        h,c,p = rhospital(),rlocation(),rphone()
        return f"{h} in {c} helpline: {p}.", [("HOSPITAL",h),("LOCATION",c),("PHONE",p)]
    if tid == 41:
        n,p = rname(),rphone()
        return f"Mobile number registered for {n}: {p}.", [("NAME",n),("PHONE",p)]

    # ── Email templates ───────────────────────────────────────────────────
    if tid == 42:
        e = remail()
        return f"Send discharge summary to {e}.", [("EMAIL",e)]
    if tid == 43:
        n,e = rname(),remail()
        return f"Lab report for {n} sent to {e}.", [("NAME",n),("EMAIL",e)]
    if tid == 44:
        doc,e = rdoctor(),remail()
        return f"Dr {doc}'s email address is {e}.", [("DOCTOR",doc),("EMAIL",e)]
    if tid == 45:
        n,e = rname(),remail()
        return f"Insurance documents for {n} forwarded to {e}.", [("NAME",n),("EMAIL",e)]
    if tid == 46:
        n,e,p = rname(),remail(),rphone()
        return f"Patient {n}: email {e}, phone {p}.", [("NAME",n),("EMAIL",e),("PHONE",p)]
    if tid == 47:
        e,n,d = remail(),rname(),rdate()
        return f"Prescription for {n} dated {d} emailed to {e}.", [("EMAIL",e),("NAME",n),("DATE",d)]

    # ── ID templates ──────────────────────────────────────────────────────
    if tid == 48:
        pid,n = rid(),rname()
        return f"Patient ID {pid} belongs to {n}.", [("ID",pid),("NAME",n)]
    if tid == 49:
        pid,n,h = rid(),rname(),rhospital()
        return f"Record {pid} for {n} at {h} is updated.", [("ID",pid),("NAME",n),("HOSPITAL",h)]
    if tid == 50:
        n,pid = rname(),rid()
        return f"{n} was assigned UHID {pid} at registration.", [("NAME",n),("ID",pid)]
    if tid == 51:
        pid,n,d = rid(),rname(),rdate()
        return f"MRN {pid} for patient {n} created on {d}.", [("ID",pid),("NAME",n),("DATE",d)]
    if tid == 52:
        pid,h = rid(),rhospital()
        return f"File number {pid} maintained at {h}.", [("ID",pid),("HOSPITAL",h)]
    if tid == 53:
        n,pid,e = rname(),rid(),remail()
        return f"Billing ID {pid} for {n} sent to {e}.", [("NAME",n),("ID",pid),("EMAIL",e)]
    if tid == 54:
        pid = rid()
        return f"OPD number {pid} issued at the registration counter.", [("ID",pid)]

    # ── Multi-entity rich templates ───────────────────────────────────────
    if tid == 55:
        doc,n,h,d = rdoctor(),rname(),rhospital(),rdate()
        return f"Dr {doc} reviewed {n} at {h} on {d} and ordered blood tests.", [("DOCTOR",doc),("NAME",n),("HOSPITAL",h),("DATE",d)]
    if tid == 56:
        n,h,c,pid,d = rname(),rhospital(),rlocation(),rid(),rdate()
        return f"Patient {n} from {c}, UHID {pid}, admitted to {h} on {d}.", [("NAME",n),("LOCATION",c),("ID",pid),("HOSPITAL",h),("DATE",d)]
    if tid == 57:
        n,p,e,pid = rname(),rphone(),remail(),rid()
        return f"Contact info for {n} — Phone: {p}, Email: {e}, MRN: {pid}.", [("NAME",n),("PHONE",p),("EMAIL",e),("ID",pid)]
    if tid == 58:
        doc,n,diag,d,h = rdoctor(),rname(),rdiag(),rdate(),rhospital()
        return f"Dr {doc} confirmed {diag} for {n} on {d} at {h}.", [("DOCTOR",doc),("NAME",n),("DATE",d),("HOSPITAL",h)]
    if tid == 59:
        n,c,h,d = rname(),rlocation(),rhospital(),rdate()
        return f"{n} residing in {c} was transferred to {h} on {d}.", [("NAME",n),("LOCATION",c),("HOSPITAL",h),("DATE",d)]
    if tid == 60:
        doc,n,med,d = rdoctor(),rname(),rmed(),rdate()
        return f"Dr {doc} prescribed {med} for {n} on {d}.", [("DOCTOR",doc),("NAME",n),("DATE",d)]

    # ── Clinical report header templates ──────────────────────────────────
    if tid == 61:
        n,pid,d = rname(),rid(),rdate()
        return f"Patient Name: {n}  |  UHID: {pid}  |  Date: {d}", [("NAME",n),("ID",pid),("DATE",d)]
    if tid == 62:
        n,doc,h = rname(),rdoctor(),rhospital()
        return f"Patient: {n}  Referring Doctor: Dr {doc}  Hospital: {h}", [("NAME",n),("DOCTOR",doc),("HOSPITAL",h)]
    if tid == 63:
        n,c,p,e = rname(),rlocation(),rphone(),remail()
        return f"Name: {n}  Address: {c}  Phone: {p}  Email: {e}", [("NAME",n),("LOCATION",c),("PHONE",p),("EMAIL",e)]
    if tid == 64:
        n,pid,d,h = rname(),rid(),rdate(),rhospital()
        return f"Discharge Summary — {n} ({pid}) from {h} on {d}.", [("NAME",n),("ID",pid),("HOSPITAL",h),("DATE",d)]
    if tid == 65:
        n,doc,d,pid = rname(),rdoctor(),rdate(),rid()
        return f"Prescription — Patient: {n}  Doctor: Dr {doc}  Date: {d}  Ref: {pid}", [("NAME",n),("DOCTOR",doc),("DATE",d),("ID",pid)]

    # ── Insurance / admin templates ───────────────────────────────────────
    if tid == 66:
        n,pid,d = rname(),rid(),rdate()
        return f"Insurance claim {pid} filed for {n} on {d}.", [("ID",pid),("NAME",n),("DATE",d)]
    if tid == 67:
        n,h,pid = rname(),rhospital(),rid()
        return f"{n} is registered at {h} with policy number {pid}.", [("NAME",n),("HOSPITAL",h),("ID",pid)]
    if tid == 68:
        n,e,pid = rname(),remail(),rid()
        return f"Bill {pid} for patient {n} emailed to {e}.", [("ID",pid),("NAME",n),("EMAIL",e)]
    if tid == 69:
        n,p,d = rname(),rphone(),rdate()
        return f"Reminder sent to {n} at {p} for appointment on {d}.", [("NAME",n),("PHONE",p),("DATE",d)]

    # ── Lab / pharmacy templates ──────────────────────────────────────────
    if tid == 70:
        n,d,pid = rname(),rdate(),rid()
        return f"Lab sample collected from {n} on {d}. Reference: {pid}.", [("NAME",n),("DATE",d),("ID",pid)]
    if tid == 71:
        n,proc,d = rname(),rproc(),rdate()
        return f"{proc} report for {n} dated {d} is ready.", [("NAME",n),("DATE",d)]
    if tid == 72:
        n,med,doc = rname(),rmed(),rdoctor()
        return f"Pharmacy dispensed {med} to {n} as prescribed by Dr {doc}.", [("NAME",n),("DOCTOR",doc)]
    if tid == 73:
        n,pid,proc = rname(),rid(),rproc()
        return f"Order {pid}: {proc} requested for patient {n}.", [("ID",pid),("NAME",n)]

    # ── Negative / O-only sentences (reduce false positives) ─────────────
    if tid in range(74, 85):
        sentences_o = [
            "The patient was stable overnight and vitals were normal.",
            "Blood pressure recorded as 120/80 mmHg.",
            "Hemoglobin levels were within normal range.",
            "The surgery was completed without complications.",
            "Follow-up blood work is recommended in two weeks.",
            "The ward is well-equipped with modern facilities.",
            "All investigations were normal.",
            "The patient tolerated the procedure well.",
            "No significant findings on the CT scan.",
            "Discharge instructions were explained to the family.",
            "Vital signs are stable. Continue current medications.",
        ]
        return random.choice(sentences_o), []

    # ── Extra variety templates ───────────────────────────────────────────
    if tid == 85:
        n,h,c,d = rname(),rhospital(),rlocation(),rdate()
        return f"Inpatient record: {n} admitted at {h}, {c} on {d}.", [("NAME",n),("HOSPITAL",h),("LOCATION",c),("DATE",d)]
    if tid == 86:
        n,doc,h = rname(),rdoctor(),rhospital()
        return f"Case of {n} under care of Dr {doc} at {h}.", [("NAME",n),("DOCTOR",doc),("HOSPITAL",h)]
    if tid == 87:
        n,c,p = rname(),rlocation(),rphone()
        return f"Home address of {n}: {c}. Contact: {p}.", [("NAME",n),("LOCATION",c),("PHONE",p)]
    if tid == 88:
        doc,n,d,diag = rdoctor(),rname(),rdate(),rdiag()
        return f"Clinical note — Dr {doc} assessed {n} on {d} for {diag}.", [("DOCTOR",doc),("NAME",n),("DATE",d)]
    if tid == 89:
        n,pid,h,e = rname(),rid(),rhospital(),remail()
        return f"Medical records of {n} (ID: {pid}) from {h} sent to {e}.", [("NAME",n),("ID",pid),("HOSPITAL",h),("EMAIL",e)]
    if tid == 90:
        n,d = rname(),rdate()
        return f"{n} checked in on {d} for routine examination.", [("NAME",n),("DATE",d)]
    if tid == 91:
        n,pid = rname(),rid()
        return f"Patient {n} holds registration number {pid}.", [("NAME",n),("ID",pid)]
    if tid == 92:
        h,c = rhospital(),rlocation()
        return f"{h} is located in {c} and provides 24x7 emergency services.", [("HOSPITAL",h),("LOCATION",c)]
    if tid == 93:
        doc,dept,h,c = rdoctor(),rdept(),rhospital(),rlocation()
        return f"Dr {doc} is a senior consultant in {dept} at {h}, {c}.", [("DOCTOR",doc),("HOSPITAL",h),("LOCATION",c)]
    if tid == 94:
        n,d,c = rname(),rdate(),rlocation()
        return f"Patient {n} from {c} last visited on {d}.", [("NAME",n),("LOCATION",c),("DATE",d)]
    if tid == 95:
        n,p,h,d = rname(),rphone(),rhospital(),rdate()
        return f"Reminder: {n} has an appointment at {h} on {d}. Confirm at {p}.", [("NAME",n),("HOSPITAL",h),("DATE",d),("PHONE",p)]
    if tid == 96:
        n,e = rname(),remail()
        return f"Health report for {n} has been dispatched to {e}.", [("NAME",n),("EMAIL",e)]
    if tid == 97:
        pid,n,d,h = rid(),rname(),rdate(),rhospital()
        return f"OPD slip {pid} issued to {n} on {d} at {h}.", [("ID",pid),("NAME",n),("DATE",d),("HOSPITAL",h)]
    if tid == 98:
        doc,n,proc,h = rdoctor(),rname(),rproc(),rhospital()
        return f"Dr {doc} has ordered {proc} for {n} at {h}.", [("DOCTOR",doc),("NAME",n),("HOSPITAL",h)]
    if tid == 99:
        n,doc,diag,med = rname(),rdoctor(),rdiag(),rmed()
        return f"{n} diagnosed with {diag} by Dr {doc}. Prescribed: {med}.", [("NAME",n),("DOCTOR",doc)]
    if tid == 100:
        n,h,pid,d,e = rname(),rhospital(),rid(),rdate(),remail()
        return f"Patient: {n}, Hospital: {h}, MRN: {pid}, Admission: {d}, Contact: {e}.", [("NAME",n),("HOSPITAL",h),("ID",pid),("DATE",d),("EMAIL",e)]

    # ── Fallback ──────────────────────────────────────────────────────────
    n,h,d = rname(),rhospital(),rdate()
    return f"Patient {n} attended {h} on {d}.", [("NAME",n),("HOSPITAL",h),("DATE",d)]


def make_tagged_sentence():
    sentence, entities = make_sentence()
    tokens = tokenize(sentence)
    if not tokens: return None
    tags = bio_tag(tokens, entities)
    return list(zip(tokens, tags))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n=SYNTHETIC_SENTENCES):
    print(f"[SyntheticGen] Generating {n:,} sentences...")
    sentences, attempts = [], 0
    while len(sentences) < n:
        r = make_tagged_sentence()
        if r: sentences.append(r)
        attempts += 1
        if attempts > n * 4: break
        if len(sentences) % 10000 == 0 and len(sentences) > 0:
            print(f"  {len(sentences):,} / {n:,}")
    print(f"[SyntheticGen] Done — {len(sentences):,} sentences generated.")
    return sentences

def write_conll(sentences, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, tag in sent:
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")
    print(f"[SyntheticGen] Written {len(sentences):,} → {filepath}")

if __name__ == "__main__":
    sents = generate_synthetic_dataset(SYNTHETIC_SENTENCES)
    write_conll(sents, DATA_DIR / "synthetic_raw.txt")


def split_and_save(sentences, train_file, dev_file, test_file,
                   train_ratio=0.70, dev_ratio=0.15):
    import random
    random.shuffle(sentences)
    n       = len(sentences)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)
    train   = sentences[:n_train]
    dev     = sentences[n_train:n_train + n_dev]
    test    = sentences[n_train + n_dev:]
    write_conll(train, train_file)
    write_conll(dev,   dev_file)
    write_conll(test,  test_file)
    print(f"[Split] Train: {len(train):,} | Dev: {len(dev):,} | Test: {len(test):,}")


if __name__ == "__main__":
    from config.config import TRAIN_FILE, DEV_FILE, TEST_FILE, TRAIN_RATIO, DEV_RATIO
    sents = generate_synthetic_dataset(SYNTHETIC_SENTENCES)
    split_and_save(sents, TRAIN_FILE, DEV_FILE, TEST_FILE, TRAIN_RATIO, DEV_RATIO)
    print("Dataset generation complete.")