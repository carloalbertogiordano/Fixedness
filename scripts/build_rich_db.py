"""
build_rich_db.py
=================
Genera rich_test.db (SQLite) + 8 CSV in tests/fixedness_test/database/

DESIGN:
  patients contiene valori ESATTI (non anonimizzati).
  L'anonimizzazione avviene A VOLO dall'anonymizer (Mondrian/k-anon).
  patients.person_id è il link nascosto verso persons — ground truth per valutazione.

  Lato MEDICO (dati privati, target dell'attacco):
    patients          - record clinici con valori esatti, person_id FK nascosto
    medical_visits    - visite granulari FK→patients
    prescriptions     - prescrizioni FK→patients, FD: diagnosis→drug_class
    lab_results       - esami FK→patients, FD: test_type→(unit,ref), value→flag

  Lato ORACLE (dati pubblici, noti all'attaccante):
    persons           - anagrafica pubblica (5000 match + 2500 rumore)
    insurance_policies - FK→persons, FD: policy_type→coverage→deductible
    consumer_profiles  - FK→persons, seed da Mall_Customers.csv
    social_links       - FK→persons, grafo FB+Twitter

Dipendenze Funzionali deterministiche:
  patients: diagnosis → treatment
  patients: diagnosis → drug_class
  patients: zip → (city, province, region)          [FD inter-tabella con persons]
  prescriptions: drug_class → is_chronic
  persons: zip → (city, province, region)
  insurance_policies: policy_type → coverage_level → deductible → copay_pct
  consumer_profiles: loyalty_tier → loyalty_discount
  lab_results: test_type → (unit, reference_low, reference_high)
  lab_results: (value < ref_low OR value > ref_high) → abnormal_flag = 1

Quasi-Identificatori (QI) in patients — valori esatti, generalizzati a runtime:
  age, sex, city, zip, region, job, bmi

Attributi Sensibili (SA) in patients:
  diagnosis, treatment, drug_class, insurance_tier,
  num_annual_visits, hba1c, ldl, glicemia_digiuno,
  pressione_sistolica, risk_score
"""

import pandas as pd
import numpy as np
import random
import sqlite3
import os
from datetime import date

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_PATIENTS      = 5_000
N_NOISE_PERSONS = 2_500

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database')
os.makedirs(OUT_DIR, exist_ok=True)
DB_PATH = os.path.join(OUT_DIR, 'rich_test.db')

# ─────────────────────────────────────────────────────────────
# LOOKUP TABLES
# ─────────────────────────────────────────────────────────────

# FD deterministica: zip → (city, province, region)
ZIP_TABLE = {
    '20121': ('Milano',  'MI', 'Lombardia'),  '20122': ('Milano',  'MI', 'Lombardia'),
    '20123': ('Milano',  'MI', 'Lombardia'),  '20124': ('Milano',  'MI', 'Lombardia'),
    '20125': ('Milano',  'MI', 'Lombardia'),
    '00100': ('Roma',    'RM', 'Lazio'),      '00118': ('Roma',    'RM', 'Lazio'),
    '00121': ('Roma',    'RM', 'Lazio'),      '00136': ('Roma',    'RM', 'Lazio'),
    '00141': ('Roma',    'RM', 'Lazio'),
    '80121': ('Napoli',  'NA', 'Campania'),   '80122': ('Napoli',  'NA', 'Campania'),
    '80123': ('Napoli',  'NA', 'Campania'),   '80131': ('Napoli',  'NA', 'Campania'),
    '80132': ('Napoli',  'NA', 'Campania'),
    '84121': ('Salerno', 'SA', 'Campania'),   '84122': ('Salerno', 'SA', 'Campania'),
    '84123': ('Salerno', 'SA', 'Campania'),   '84124': ('Salerno', 'SA', 'Campania'),
    '84125': ('Salerno', 'SA', 'Campania'),
    '10121': ('Torino',  'TO', 'Piemonte'),   '10122': ('Torino',  'TO', 'Piemonte'),
    '10123': ('Torino',  'TO', 'Piemonte'),   '10124': ('Torino',  'TO', 'Piemonte'),
    '10125': ('Torino',  'TO', 'Piemonte'),
}
CITY_ZIPS   = {}
for _z, (_c, _p, _r) in ZIP_TABLE.items():
    CITY_ZIPS.setdefault(_c, []).append(_z)
CITY_CODES  = {'Milano':'F205','Roma':'H501','Napoli':'F839','Salerno':'H703','Torino':'L219'}
CITY_W      = {'Milano':25,'Roma':30,'Napoli':20,'Salerno':10,'Torino':15}
CITIES_LIST = list(CITY_W.keys())
CITIES_WGTS = list(CITY_W.values())

JOBS   = ['Studente','Operaio','Pensionato','Dirigente','Disoccupato']
JOB_W  = [20, 22, 28, 15, 15]
DIAGS  = ['Sana Costituzione','Influenza','Ipertensione','Cardiopatia','Diabete']

# ── Dipendenze Funzionali deterministiche ───────────────────

# FD: diagnosis → treatment
TREATMENT_FD = {
    'Sana Costituzione': 'Nessuno',
    'Influenza':         'Antivirali_FANS',
    'Ipertensione':      'Beta-bloccanti',
    'Cardiopatia':       'Intervento_Chirurgico',
    'Diabete':           'Insulina',
}
# FD: diagnosis → drug_class
DRUG_CLASS_FD = {
    'Sana Costituzione': 'Nessuno',
    'Influenza':         'Antivirali_FANS',
    'Ipertensione':      'Antiipertensivi',
    'Cardiopatia':       'Anticoagulanti',
    'Diabete':           'Ipoglicemizzanti',
}
# FD: drug_class → insurance_tier (deterministic — crea ciclo con tier→drug_class)
DRUG_CLASS_TO_TIER = {
    'Nessuno':          'Base',
    'Antivirali_FANS':  'Base',
    'Antiipertensivi':  'Standard',
    'Anticoagulanti':   'Premium',
    'Ipoglicemizzanti': 'Premium',
}
# Restriction FD: insurance_tier → allowed drug_classes (reverse del ciclo)
TIER_TO_DRUG_CLASSES = {
    'Base':     {'Nessuno', 'Antivirali_FANS'},
    'Standard': {'Nessuno', 'Antivirali_FANS', 'Antiipertensivi'},
    'Premium':  {'Nessuno', 'Antivirali_FANS', 'Antiipertensivi',
                 'Anticoagulanti', 'Ipoglicemizzanti'},
}

# Soglie cliniche deterministiche per FD bidirezionale HbA1c ↔ Diagnosis
HBA1C_MIN = {'Sana Costituzione': 4.0, 'Influenza': 4.0,
             'Ipertensione': 4.0, 'Cardiopatia': 4.0, 'Diabete': 6.5}
HBA1C_MAX = {'Sana Costituzione': 6.4, 'Influenza': 6.4,
             'Ipertensione': 6.4, 'Cardiopatia': 6.4, 'Diabete': 15.0}

# Soglie cliniche deterministiche per FD Pressione ↔ {Ipertensione, Cardiopatia}
PRESS_MIN = {'Sana Costituzione': 90,  'Influenza': 90,
             'Ipertensione': 140, 'Cardiopatia': 130, 'Diabete': 90}
PRESS_MAX = {'Sana Costituzione': 129, 'Influenza': 129,
             'Ipertensione': 200, 'Cardiopatia': 200, 'Diabete': 129}

# FD: drug_class → drug_names (1-to-many)
DRUG_NAMES_FD = {
    'Nessuno':          ['—'],
    'Antivirali_FANS':  ['Ibuprofene','Paracetamolo','Oseltamivir'],
    'Antiipertensivi':  ['Ramipril','Amlodipina','Losartan','Bisoprololo'],
    'Anticoagulanti':   ['Warfarin','Aspirina_100mg','Clopidogrel','Atorvastatina'],
    'Ipoglicemizzanti': ['Metformina','Insulina_Glargine','Sitagliptin','Empagliflozin'],
}
DOSAGE_BASE = {
    'Ibuprofene':400,'Paracetamolo':500,'Oseltamivir':75,
    'Ramipril':5,'Amlodipina':5,'Losartan':50,'Bisoprololo':5,
    'Warfarin':5,'Aspirina_100mg':100,'Clopidogrel':75,'Atorvastatina':20,
    'Metformina':850,'Insulina_Glargine':10,'Sitagliptin':100,'Empagliflozin':10,
    '—':0,
}
# FD: diagnosis → is_chronic
IS_CHRONIC_FD = {
    'Sana Costituzione':False,'Influenza':False,
    'Ipertensione':True,'Cardiopatia':True,'Diabete':True,
}
# FD probabilistica forte: diagnosis → department
DEPT_FD = {
    'Sana Costituzione': ['Medicina_Generale'],
    'Influenza':         ['Medicina_Generale','Pronto_Soccorso'],
    'Ipertensione':      ['Cardiologia','Medicina_Interna'],
    'Cardiopatia':       ['Cardiologia','Cardiochirurgia','UTI'],
    'Diabete':           ['Endocrinologia','Medicina_Interna'],
}
# FD: policy_type → coverage_level
COVERAGE_FD   = {'Base':'Ricovero','Standard':'Ricovero_Specialistica','Premium':'Completa'}
# FD: coverage_level → deductible
DEDUCTIBLE_FD = {'Ricovero':500.0,'Ricovero_Specialistica':200.0,'Completa':0.0}
# FD: coverage_level → copay_pct
COPAY_FD      = {'Ricovero':0.30,'Ricovero_Specialistica':0.15,'Completa':0.05}
# FD: loyalty_tier → loyalty_discount
LOYALTY_DISC  = {'Bronze':0.05,'Silver':0.10,'Gold':0.15,'Platinum':0.25}
# FD: test_type → (unit, reference_low, reference_high)
LAB_REF_FD = {
    'HbA1c':              ('%',       4.0,  5.6),
    'Glicemia_Digiuno':   ('mg/dL',  70.0, 99.0),
    'LDL':                ('mg/dL',   0.0, 99.0),
    'Pressione_Sistolica':('mmHg',   90.0,120.0),
    'Colesterolo_Totale': ('mg/dL', 100.0,200.0),
    'Trigliceridi':       ('mg/dL',  50.0,150.0),
}
DIAG_TESTS = {
    'Sana Costituzione': ['HbA1c','Glicemia_Digiuno','LDL'],
    'Influenza':         ['Glicemia_Digiuno','Colesterolo_Totale'],
    'Ipertensione':      ['Pressione_Sistolica','Colesterolo_Totale','Trigliceridi'],
    'Cardiopatia':       ['Pressione_Sistolica','LDL','Colesterolo_Totale','Trigliceridi'],
    'Diabete':           ['HbA1c','Glicemia_Digiuno','LDL','Trigliceridi'],
}

# ── Nomi, cognomi, indirizzi ─────────────────────────────────

COGNOMI = [
    'Rossi','Ferrari','Russo','Bianchi','Romano','Gallo','Costa','Fontana',
    'Conti','Esposito','Ricci','Bruno','De Luca','Moretti','Marino','Greco',
    'Barbieri','Lombardi','Giordano','Colombo','Mancini','Longo','Leone',
    'Martinelli','Marini','Ferrara','Vitale','Orlando','Serra','Coppola',
    'Caruso','Amato','Pellegrini','Santoro','Silvestri','Martini','Palumbo',
    'Messina','Villa','Napolitano','Sartori','Cattaneo','Fabbri','Pagano',
]
NOMI_M = [
    'Marco','Luca','Andrea','Matteo','Francesco','Alessandro','Davide',
    'Simone','Fabio','Roberto','Paolo','Stefano','Riccardo','Daniele',
    'Emilio','Giuseppe','Antonio','Gianluca','Lorenzo','Giorgio','Mario',
    'Carlo','Vincenzo','Angelo','Nicola','Salvatore','Sergio','Bruno',
    'Claudio','Federico','Massimo','Enrico','Filippo','Alberto','Pietro',
]
NOMI_F = [
    'Giulia','Sofia','Martina','Sara','Laura','Elena','Chiara','Valentina',
    'Federica','Alessandra','Paola','Francesca','Maria','Anna','Roberta',
    'Silvia','Monica','Cristina','Claudia','Giovanna','Rosa','Carmela',
    'Michela','Beatrice','Lucia','Teresa','Angela','Concetta','Patrizia',
    'Lorenza','Simona','Elisa','Serena','Ilaria','Cinzia',
]
VIE = [
    'Via Roma','Via Milano','Via Napoli','Corso Italia','Via Garibaldi',
    'Via Mazzini','Via Dante','Via Verdi','Piazza del Comune','Via della Repubblica',
    'Corso Vittorio Emanuele','Via Cavour','Via Leopardi','Via Marconi',
    'Via Matteotti','Corso Umberto I','Via XX Settembre','Via Nazionale',
    'Via della Libertà','Viale Europa','Via Tasso',
]
INSURERS = [
    'Generali Assicurazioni','Allianz Italia','UnipolSai','Cattolica Assicurazioni',
    'Reale Mutua','HDI Assicurazioni','BNP Paribas Cardif','Poste Vita',
]
MESI_CF = 'ABCDEHLMPRST'

EDUCATION_W = {
    'Dirigente':   (['Laurea_Magistrale','Laurea_Triennale','Diploma'],    [60,30,10]),
    'Studente':    (['Diploma','Laurea_Triennale','Laurea_Magistrale'],     [50,45,5]),
    'Operaio':     (['Licenza_Media','Diploma','Laurea_Triennale'],         [50,45,5]),
    'Disoccupato': (['Licenza_Media','Diploma','Laurea_Triennale'],         [40,45,15]),
    'Pensionato':  (['Licenza_Media','Diploma','Laurea_Triennale'],         [50,40,10]),
}
INCOME_W = {
    'Studente':   [80,20,0,0], 'Operaio':  [10,60,25,5], 'Pensionato': [5,55,30,10],
    'Dirigente':  [2,8,30,60], 'Disoccupato':[85,15,0,0],
}
INCOME_BRACKETS = ['<15k','15k-30k','30k-50k','>50k']
INCOME_K_RANGES = {'<15k':(5,14),'15k-30k':(15,29),'30k-50k':(30,49),'>50k':(50,120)}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _consonants(s): return [c for c in s.upper() if c in 'BCDFGHJKLMNPQRSTVWXYZ']
def _vowels(s):     return [c for c in s.upper() if c in 'AEIOU']

def _cf_part(word, n=3):
    parts = _consonants(word) + _vowels(word) + ['X','X','X']
    return ''.join(parts[:n])

def codice_fiscale(cognome, nome, year, month, day, sesso, city_code='H501'):
    cog = _cf_part(cognome, 3)
    cn  = _consonants(nome)
    nom = (cn[0]+cn[2]+cn[3]) if len(cn) >= 4 else _cf_part(nome, 3)
    yr  = str(year)[-2:]
    mo  = MESI_CF[month - 1]
    dd  = f"{(day + 40 if sesso == 'F' else day):02d}"
    return f"{cog}{nom}{yr}{mo}{dd}{city_code}".upper()

def age_for_job(job):
    ranges = {'Studente':(18,32),'Operaio':(22,65),'Dirigente':(30,65),
              'Disoccupato':(20,65),'Pensionato':(55,90)}
    lo, hi = ranges[job]
    return random.randint(lo, hi)

# ─────────────────────────────────────────────────────────────
# EPIDEMIOLOGICAL MODEL
# ─────────────────────────────────────────────────────────────

def score_diagnosis(age, job, sesso, bmi, fumatore, attivita):
    s = {d: 0.01 for d in DIAGS}
    if age < 30:
        s['Sana Costituzione'] += 5.0; s['Influenza'] += 1.5
    elif age < 50:
        s['Sana Costituzione'] += 2.0; s['Ipertensione'] += 1.0; s['Diabete'] += 0.8
    elif age < 65:
        s['Ipertensione'] += 2.5; s['Diabete'] += 1.8; s['Cardiopatia'] += 1.2
    else:
        s['Ipertensione'] += 4.0; s['Cardiopatia'] += 3.5; s['Diabete'] += 2.0

    if bmi < 18.5:
        s['Influenza'] += 1.0
    elif bmi < 25:
        s['Sana Costituzione'] += 2.5
    elif bmi < 30:
        s['Ipertensione'] += 1.5; s['Diabete'] += 1.0
    elif bmi < 35:
        s['Ipertensione'] += 2.5; s['Diabete'] += 2.5; s['Cardiopatia'] += 1.5
    else:
        s['Ipertensione'] += 3.0; s['Diabete'] += 4.0; s['Cardiopatia'] += 3.0

    if fumatore == 'Sì':
        s['Cardiopatia'] += 3.0; s['Ipertensione'] += 2.0
    if attivita == 'alta':
        s['Sana Costituzione'] += 2.0
        s['Ipertensione'] = max(0.01, s['Ipertensione'] - 1.5)
        s['Diabete']      = max(0.01, s['Diabete']      - 1.5)
    elif attivita == 'bassa':
        s['Ipertensione'] += 1.5; s['Diabete'] += 1.5; s['Cardiopatia'] += 1.0

    if job == 'Dirigente':     s['Ipertensione'] += 2.0; s['Diabete'] += 0.8
    elif job == 'Operaio':     s['Cardiopatia']  += 0.8; s['Ipertensione'] += 0.5
    elif job == 'Disoccupato': s['Ipertensione'] += 1.5

    if sesso == 'M' and age > 45:   s['Cardiopatia'] += 1.5
    elif sesso == 'F' and age > 55: s['Cardiopatia'] += 0.8; s['Ipertensione'] += 0.5
    return {k: max(v, 0.01) for k, v in s.items()}

def sample_bmi(diag):
    mu, sd = {'Sana Costituzione':(22,1.8),'Influenza':(23,2.5),'Ipertensione':(28.5,4),
              'Cardiopatia':(29.5,4.5),'Diabete':(31.5,5)}[diag]
    return round(float(np.clip(np.random.normal(mu, sd), 15.0, 50.0)), 1)

def sample_hba1c(diag):
    mu, sd = {'Sana Costituzione':(5.0,.25),'Influenza':(5.1,.3),'Ipertensione':(5.6,.4),
              'Cardiopatia':(5.8,.5),'Diabete':(8.8,1.8)}[diag]
    return round(float(np.clip(np.random.normal(mu, sd), 4.0, 15.0)), 1)

def sample_ldl(diag, fumatore):
    mu, sd = {'Sana Costituzione':(100,18),'Influenza':(105,20),'Ipertensione':(135,25),
              'Cardiopatia':(150,30),'Diabete':(130,25)}[diag]
    if fumatore == 'Sì': mu += 15
    return int(np.clip(np.random.normal(mu, sd), 40, 250))

def sample_glicemia(diag):
    mu, sd = {'Sana Costituzione':(85,8),'Influenza':(88,10),'Ipertensione':(100,12),
              'Cardiopatia':(105,15),'Diabete':(185,40)}[diag]
    return int(np.clip(np.random.normal(mu, sd), 50, 400))

def sample_pressione(diag, age, fumatore):
    mu, sd = {'Sana Costituzione':(112,8),'Influenza':(116,9),'Ipertensione':(158,14),
              'Cardiopatia':(148,16),'Diabete':(138,11)}[diag]
    if age > 65: mu += 10
    if fumatore == 'Sì': mu += 8
    return int(np.clip(np.random.normal(mu, sd), 90, 200))

def compute_risk_score(diag, age, bmi, fumatore):
    base = {'Sana Costituzione':1,'Influenza':2,'Ipertensione':5,'Cardiopatia':8,'Diabete':6}[diag]
    if age > 65: base += 1
    if bmi > 30: base += 1
    if fumatore == 'Sì': base += 1
    return min(10, max(1, base + random.randint(-1, 1)))

# ─────────────────────────────────────────────────────────────
# STEP 1: GENERA PERSONS prima (oracle pubblico)
# Poi genera PATIENTS con person_id FK che punta a persons.
# ─────────────────────────────────────────────────────────────

# Prima raccoglie i parametri demografici per i 5000 pazienti,
# poi costruisce persons da quelli (+ 2500 rumore).
# Così patients.person_id → persons.person_id è un join valido.

print("Generating demographic parameters for patients...")
pat_demos = []   # lista di dict con dati demografici grezzi per ogni paziente

SMOKER_P   = {'Sana Costituzione':10,'Influenza':20,'Ipertensione':40,'Cardiopatia':60,'Diabete':30}
ACTIVITY_W = {
    'Sana Costituzione':[8,38,54],'Influenza':[30,45,25],
    'Ipertensione':[65,28,7],'Cardiopatia':[65,28,7],'Diabete':[58,33,9],
}
TIER_W = {
    'Sana Costituzione':[50,35,15],'Influenza':[55,35,10],
    'Ipertensione':[30,45,25],'Cardiopatia':[20,35,45],'Diabete':[25,45,30],
}
VISIT_RANGE = {
    'Sana Costituzione':(0,1),'Influenza':(0,3),'Ipertensione':(2,6),
    'Cardiopatia':(4,10),'Diabete':(2,7),
}

for pid in range(N_PATIENTS):
    job   = random.choices(JOBS, weights=JOB_W)[0]
    city  = random.choices(CITIES_LIST, weights=CITIES_WGTS)[0]
    age   = age_for_job(job)
    sesso = random.choices(['M','F'], weights=[55,45])[0]
    zip_  = random.choice(CITY_ZIPS[city])
    _, prov, region = ZIP_TABLE[zip_]

    # Step 1: campiona diagnosi con valori neutri
    probe_bmi = float(np.random.normal(25, 5))
    probe_fum = random.choices(['Sì','No'], weights=[22,78])[0]
    probe_att = random.choices(['bassa','media','alta'], weights=[30,45,25])[0]
    scores    = score_diagnosis(age, job, sesso, probe_bmi, probe_fum, probe_att)
    diag      = random.choices(list(scores.keys()), weights=list(scores.values()))[0]

    # Step 2: campiona parametri clinici coerenti con la diagnosi
    bmi_val  = sample_bmi(diag)
    sp       = SMOKER_P[diag]
    fumatore = random.choices(['Sì','No'], weights=[sp, 100-sp])[0]
    attivita = random.choices(['bassa','media','alta'], weights=ACTIVITY_W[diag])[0]
    hba1c    = float(np.clip(sample_hba1c(diag), HBA1C_MIN[diag], HBA1C_MAX[diag]))
    ldl      = sample_ldl(diag, fumatore)
    glicemia = sample_glicemia(diag)
    press    = int(np.clip(sample_pressione(diag, age, fumatore),
                           PRESS_MIN[diag], PRESS_MAX[diag]))
    rs       = compute_risk_score(diag, age, bmi_val, fumatore)

    # FD deterministica: drug_class → insurance_tier (crea ciclo con tier→drug_class)
    drug_class = DRUG_CLASS_FD[diag]
    tier       = DRUG_CLASS_TO_TIER[drug_class]
    lo, hi   = VISIT_RANGE[diag]
    n_visits = random.randint(lo, hi)

    pat_demos.append({
        'patient_id':           pid,
        # QI — valori ESATTI (non generalizzati)
        'age':                  age,
        'sex':                  sesso,
        'city':                 city,
        'zip':                  zip_,
        'province':             prov,
        'region':               region,
        'job':                  job,
        'bmi':                  bmi_val,
        'smoking_status':       fumatore,
        'physical_activity':    attivita,
        # SA — attributi sensibili, valori esatti
        'diagnosis':            diag,
        'treatment':            TREATMENT_FD[diag],
        'drug_class':           drug_class,
        'insurance_tier':       tier,
        'num_annual_visits':    n_visits,
        'hba1c':                hba1c,
        'ldl':                  ldl,
        'glicemia_digiuno':     glicemia,
        'pressione_sistolica':  press,
        'risk_score':           rs,
    })
    if pid % 1000 == 0: print(f"  {pid}/{N_PATIENTS}")

print(f"  demographic params: {len(pat_demos)} done\n")

# ─────────────────────────────────────────────────────────────
# STEP 2: PERSONS (oracle pubblico)
# ─────────────────────────────────────────────────────────────
print("Generating persons (oracle)...")

def make_person(pid, sesso, age, city, job):
    nomi    = NOMI_M if sesso == 'M' else NOMI_F
    nome    = random.choice(nomi)
    cogn    = random.choice(COGNOMI)
    by      = date.today().year - age
    bm      = random.randint(1, 12)
    bd      = random.randint(1, 28)
    zip_    = random.choice(CITY_ZIPS[city])
    _, prov, region = ZIP_TABLE[zip_]
    cf      = codice_fiscale(cogn, nome, by, bm, bd, sesso, CITY_CODES[city])
    inc_b   = random.choices(INCOME_BRACKETS, weights=INCOME_W[job])[0]
    lo, hi  = INCOME_K_RANGES[inc_b]
    inc_k   = random.randint(lo, hi)
    edu_v, edu_w = EDUCATION_W[job]
    edu     = random.choices(edu_v, weights=edu_w)[0]
    if age < 25:
        marital = random.choices(['Celibe_Nubile','Coniugato_a'], weights=[92,8])[0]
    elif age < 40:
        marital = random.choices(['Celibe_Nubile','Coniugato_a','Separato_a'], weights=[28,62,10])[0]
    elif age < 65:
        marital = random.choices(['Celibe_Nubile','Coniugato_a','Separato_a','Divorziato_a'],
                                  weights=[8,65,12,15])[0]
    else:
        marital = random.choices(['Coniugato_a','Vedovo_a','Divorziato_a'], weights=[50,40,10])[0]
    phone = f"0{random.randint(10,99)}{random.randint(1000000,9999999)}"
    email = (f"{nome.lower()}.{cogn.lower().replace(' ','')}"
             f"{random.randint(1,99)}@{'gmail.com' if random.random()>.5 else 'libero.it'}")
    return {
        'person_id':      pid,      'tax_code':       cf,
        'first_name':     nome,     'last_name':       cogn,
        'birth_year':     by,       'birth_month':     bm,    'birth_day': bd,
        'sex':            sesso,    'city':            city,
        'zip':            zip_,     'province':        prov,  'region':    region,
        'job':            job,      'education':       edu,
        'marital_status': marital,  'phone':           phone, 'email':     email,
        'annual_income_k':inc_k,    'income_bracket':  inc_b,
    }

pers_rows = []
# 5000 persone che corrispondono ai pazienti (stessi demografici)
for d in pat_demos:
    pers_rows.append(make_person(d['patient_id'], d['sex'], d['age'], d['city'], d['job']))
    if d['patient_id'] % 1000 == 0: print(f"  matched {d['patient_id']}/{N_PATIENTS}")
# 2500 persone rumore (non pazienti)
for i in range(N_NOISE_PERSONS):
    pid_  = N_PATIENTS + i
    job   = random.choices(JOBS, weights=JOB_W)[0]
    city  = random.choices(CITIES_LIST, weights=CITIES_WGTS)[0]
    sesso = random.choice(['M','F'])
    age   = age_for_job(job)
    pers_rows.append(make_person(pid_, sesso, age, city, job))

persons_df = pd.DataFrame(pers_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"persons: {len(persons_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: PATIENTS (lato medico, valori esatti, non anonimizzati)
# person_id è il link nascosto verso persons — ground truth
# ─────────────────────────────────────────────────────────────
print("Building patients table...")
patients_df = pd.DataFrame(pat_demos)
# Aggiunge person_id = patient_id (stessa posizione nell'oracle matched)
patients_df['person_id'] = patients_df['patient_id']
# Riordina colonne: person_id subito dopo patient_id
cols = ['patient_id','person_id','age','sex','city','zip','province','region',
        'job','bmi','smoking_status','physical_activity',
        'diagnosis','treatment','drug_class','insurance_tier',
        'num_annual_visits','hba1c','ldl','glicemia_digiuno','pressione_sistolica','risk_score']
patients_df = patients_df[cols]
print(f"patients: {len(patients_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 4: INSURANCE POLICIES (lato oracle)
# ─────────────────────────────────────────────────────────────
print("Generating insurance_policies...")
POLICY_W_BY_INC = {
    '<15k':[70,25,5],'15k-30k':[40,45,15],'30k-50k':[20,45,35],'>50k':[10,30,60],
}
BASE_PREMIUM = {'Base':800,'Standard':1400,'Premium':2400}
REGION_MULT  = {'Lombardia':1.10,'Lazio':1.05,'Campania':0.92,'Piemonte':1.02}

ins_rows = []
pol_id   = 0
for _, p in persons_df.iterrows():
    is_matched = int(p['person_id']) < N_PATIENTS
    prob = 0.85 if is_matched else 0.60
    if random.random() > prob: continue
    inc_b  = p['income_bracket']
    region = p['region']
    rmult  = REGION_MULT.get(region, 1.0)
    pt     = random.choices(['Base','Standard','Premium'], weights=POLICY_W_BY_INC[inc_b])[0]
    cov    = COVERAGE_FD[pt]
    issue  = random.randint(2015, 2024)
    expiry = issue + random.choices([1,2,3,5], weights=[20,30,30,20])[0]
    ins_rows.append({
        'policy_id':      pol_id,
        'person_id':      p['person_id'],
        'insurer':        random.choice(INSURERS),
        'policy_type':    pt,
        'annual_premium': round(BASE_PREMIUM[pt] * rmult * random.uniform(0.9,1.1), 2),
        'coverage_level': cov,             # FD: policy_type → coverage_level
        'deductible':     DEDUCTIBLE_FD[cov],  # FD: coverage_level → deductible
        'copay_pct':      COPAY_FD[cov],       # FD: coverage_level → copay_pct
        'issue_year':     issue,
        'expiry_year':    expiry,
        'active':         int(expiry >= 2025),
    })
    pol_id += 1
    # 10% polizza doppia per redditi alti
    if random.random() < 0.10 and inc_b in ('30k-50k','>50k'):
        pt2  = random.choices(['Standard','Premium'], weights=[40,60])[0]
        cov2 = COVERAGE_FD[pt2]
        ins_rows.append({
            'policy_id': pol_id, 'person_id': p['person_id'],
            'insurer': random.choice(INSURERS), 'policy_type': pt2,
            'annual_premium': round(BASE_PREMIUM[pt2]*rmult*random.uniform(0.9,1.1),2),
            'coverage_level': cov2, 'deductible': DEDUCTIBLE_FD[cov2],
            'copay_pct': COPAY_FD[cov2],
            'issue_year': random.randint(2018,2024), 'expiry_year': 2027, 'active': 1,
        })
        pol_id += 1

insurance_df = pd.DataFrame(ins_rows)
print(f"insurance_policies: {len(insurance_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 5: CONSUMER PROFILES (lato oracle, seed da Mall_Customers.csv)
# ─────────────────────────────────────────────────────────────
print("Generating consumer_profiles...")
mall_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'db_data', 'Mall_Customers.csv')
try:
    mall_df  = pd.read_csv(mall_path)
    use_mall = True
    print(f"  Mall_Customers.csv loaded: {len(mall_df)} rows")
except Exception:
    mall_df  = None
    use_mall = False

PREFERRED_CATS = ['Elettronica','Abbigliamento','Alimentari','Casa','Sport','Viaggi','Lusso']
CAT_W_BY_SEG   = {
    'Premium':    [20,15,5,10,10,15,25], 'Value_Seeker':[10,20,30,20,10,10,0],
    'Budget':     [5,20,40,25,5,5,0],    'Moderate':    [15,20,20,20,10,10,5],
    'Enthusiast': [20,20,10,15,15,15,5],
}

def spending_segment(ss, inc_b):
    if inc_b in ('<15k','15k-30k') and ss > 60: return 'Value_Seeker'
    if inc_b == '>50k' and ss > 70:             return 'Premium'
    if ss < 30:                                  return 'Budget'
    if ss < 60:                                  return 'Moderate'
    return 'Enthusiast'

cons_rows = []
prof_id   = 0
for _, p in persons_df.iterrows():
    if random.random() > 0.60: continue
    inc_k = p['annual_income_k']
    inc_b = p['income_bracket']
    if use_mall:
        close = mall_df[abs(mall_df['Annual Income (k$)'] - inc_k) < 20]
        ss    = int(random.choice(close['Spending Score (1-100)'].values)) if len(close)>0 else random.randint(1,100)
    else:
        ss_mu = {'<15k':60,'15k-30k':50,'30k-50k':45,'>50k':55}[inc_b]
        ss    = int(np.clip(np.random.normal(ss_mu, 20), 1, 100))
    loyalty = random.choices(
        ['Bronze','Silver','Gold','Platinum'],
        weights=[40,30,20,10] if inc_b in ('<15k','15k-30k') else [20,30,30,20]
    )[0]
    seg = spending_segment(ss, inc_b)
    cons_rows.append({
        'profile_id':        prof_id,      'person_id':         p['person_id'],
        'annual_income_k':   inc_k,        'income_bracket':    inc_b,
        'spending_score':    ss,           'preferred_segment': seg,
        'loyalty_tier':      loyalty,
        'loyalty_discount':  LOYALTY_DISC[loyalty],  # FD: loyalty_tier → discount
        'preferred_category': random.choices(PREFERRED_CATS, weights=CAT_W_BY_SEG[seg])[0],
        'num_purchases_yr':  random.randint(1, 50),
    })
    prof_id += 1

consumer_df = pd.DataFrame(cons_rows)
print(f"consumer_profiles: {len(consumer_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 6: SOCIAL LINKS (lato oracle)
# ─────────────────────────────────────────────────────────────
print("Generating social_links...")
person_ids_list  = persons_df['person_id'].tolist()
person_city_map  = dict(zip(persons_df['person_id'], persons_df['city']))

city_pids = {}
for pid_, city_ in person_city_map.items():
    city_pids.setdefault(city_, []).append(pid_)

# Precomputa liste non-città (evita O(n²) nel loop)
non_city_pids = {
    city_: [p for p in person_ids_list if p not in set(ids)]
    for city_, ids in city_pids.items()
}

social_rows = []
link_id     = 0
seen_fb     = set()
seen_tw     = set()

for pid_ in person_ids_list:
    city_      = person_city_map[pid_]
    same_city  = [x for x in city_pids[city_] if x != pid_]
    other_city = non_city_pids[city_]

    # Facebook: 2-8 amici, non diretti, 70% stessa città
    for _ in range(random.randint(2, 8)):
        t = (random.choice(same_city) if (random.random() < 0.70 and same_city)
             else (random.choice(other_city) if other_city else None))
        if t is None or t == pid_: continue
        key = (min(pid_,t), max(pid_,t))
        if key in seen_fb: continue
        seen_fb.add(key)
        social_rows.append({
            'link_id': link_id, 'source_id': pid_, 'target_id': t,
            'platform': 'facebook', 'link_type': 'friend',
            'strength': random.choices(['weak','medium','strong'], weights=[40,40,20])[0],
            'created_year': random.randint(2008, 2023),
        })
        link_id += 1

    # Twitter: 3-12 follows, diretti, 50% stessa città
    for _ in range(random.randint(3, 12)):
        t = (random.choice(same_city) if (random.random() < 0.50 and same_city)
             else (random.choice(other_city) if other_city else None))
        if t is None or t == pid_: continue
        key = (pid_, t)
        if key in seen_tw: continue
        seen_tw.add(key)
        social_rows.append({
            'link_id': link_id, 'source_id': pid_, 'target_id': t,
            'platform': 'twitter', 'link_type': 'follows',
            'strength': 'N/A', 'created_year': random.randint(2006, 2023),
        })
        link_id += 1

social_df = pd.DataFrame(social_rows)
print(f"social_links: {len(social_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 7: MEDICAL VISITS (lato medico)
# ─────────────────────────────────────────────────────────────
print("Generating medical_visits...")
VISIT_TYPES = ['Emergenza','Controllo_Routine','Ricovero','Specialistica']
VISIT_TYPE_W = {
    'Sana Costituzione':[5,60,5,30],'Influenza':[30,20,15,35],
    'Ipertensione':[10,50,20,20],'Cardiopatia':[20,20,40,20],'Diabete':[10,40,20,30],
}
DUR_RANGE   = {'Emergenza':(0,2),'Controllo_Routine':(0,0),'Ricovero':(2,14),'Specialistica':(0,1)}
OUTCOME_W   = {'Emergenza':[60,20,5,15],'Ricovero':[50,30,5,15],
               'Controllo_Routine':[90,5,1,4],'Specialistica':[85,10,1,4]}
OUTCOMES    = ['Dimesso','Ricoverato','Deceduto','Trasferito']

vis_rows = []
vis_id   = 0
for _, pat in patients_df.iterrows():
    diag     = pat['diagnosis']
    n        = pat['num_annual_visits']
    dept_opt = DEPT_FD[diag]
    for _ in range(n):
        vtype      = random.choices(VISIT_TYPES, weights=VISIT_TYPE_W[diag])[0]
        dur_lo, dur_hi = DUR_RANGE[vtype]
        dur        = random.randint(dur_lo, dur_hi)
        if vtype == 'Ricovero' and dur > 7:  cost = 'Alto'
        elif vtype == 'Ricovero':            cost = 'Medio_Alto'
        elif vtype == 'Emergenza':           cost = 'Medio'
        elif vtype == 'Specialistica':       cost = 'Basso_Medio'
        else:                                cost = 'Basso'
        vis_rows.append({
            'visit_id':      vis_id,
            'patient_id':    pat['patient_id'],
            'visit_year':    random.randint(2018, 2025),
            'visit_month':   random.randint(1, 12),
            'visit_type':    vtype,
            'department':    random.choice(dept_opt),  # FD prob.: diagnosis → department
            'hospital_tier': random.choices(['Comunale','Regionale','Universitario'],
                                             weights=[50,35,15])[0],
            'duration_days': dur,
            'outcome':       random.choices(OUTCOMES, weights=OUTCOME_W[vtype])[0],
            'cost_bracket':  cost,  # FD: visit_type × duration → cost_bracket
        })
        vis_id += 1

visits_df = pd.DataFrame(vis_rows)
print(f"medical_visits: {len(visits_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 8: PRESCRIPTIONS (lato medico)
# ─────────────────────────────────────────────────────────────
print("Generating prescriptions...")
rx_rows = []
rx_id   = 0
for _, pat in patients_df.iterrows():
    diag    = pat['diagnosis']
    dc      = DRUG_CLASS_FD[diag]
    chronic = IS_CHRONIC_FD[diag]
    n_rx    = 0 if diag == 'Sana Costituzione' else random.randint(1, 4 if chronic else 2)
    for _ in range(n_rx):
        dname = random.choice(DRUG_NAMES_FD[dc])
        base  = DOSAGE_BASE.get(dname, 100)
        dose  = base + random.choice([-1,0,1]) * max(1, base//5)
        rx_rows.append({
            'rx_id':             rx_id,
            'patient_id':        pat['patient_id'],
            'drug_class':        dc,         # FD: diagnosis → drug_class
            'drug_name':         dname,      # FD: drug_class → drug_name (1-to-many)
            'dosage_mg':         dose,
            'duration_days':     7 if not chronic else random.randint(30, 365),
            'prescribing_year':  random.randint(2018, 2025),
            'refills':           0 if not chronic else random.randint(1, 6),
            'is_chronic':        int(chronic),  # FD: drug_class → is_chronic
        })
        rx_id += 1

rx_df = pd.DataFrame(rx_rows)
print(f"prescriptions: {len(rx_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# STEP 9: LAB RESULTS (lato medico)
# ─────────────────────────────────────────────────────────────
print("Generating lab_results...")
lab_rows = []
lab_id   = 0
for idx, pat in patients_df.iterrows():
    diag   = pat['diagnosis']
    ttypes = DIAG_TESTS[diag]
    n_yrs  = random.randint(1, 3)
    for _ in range(n_yrs):
        for ttype in ttypes:
            unit, ref_lo, ref_hi = LAB_REF_FD[ttype]
            if ttype == 'HbA1c':               v = pat['hba1c']
            elif ttype == 'Glicemia_Digiuno':  v = float(pat['glicemia_digiuno'])
            elif ttype == 'LDL':               v = float(pat['ldl'])
            elif ttype == 'Pressione_Sistolica': v = float(pat['pressione_sistolica'])
            elif ttype == 'Colesterolo_Totale':  v = float(pat['ldl']) + random.randint(40,80)
            else:  # Trigliceridi
                base = {'Sana Costituzione':90,'Influenza':95,'Ipertensione':140,
                        'Cardiopatia':160,'Diabete':180}[diag]
                v = float(int(np.clip(np.random.normal(base, 30), 30, 400)))
            v        = round(float(v), 1)
            abnormal = int(v < ref_lo or v > ref_hi)
            if not abnormal:                              sev = 'Normal'
            elif abs(v - ref_hi) < (ref_hi-ref_lo)*0.2:  sev = 'Borderline'
            elif v > ref_hi * 1.5 or v < ref_lo * 0.5:   sev = 'Critical'
            else:                                          sev = 'Abnormal'
            lab_rows.append({
                'result_id':      lab_id,
                'patient_id':     pat['patient_id'],
                'test_type':      ttype,
                'value':          v,
                'unit':           unit,      # FD: test_type → unit
                'reference_low':  ref_lo,    # FD: test_type → reference_low
                'reference_high': ref_hi,    # FD: test_type → reference_high
                'abnormal_flag':  abnormal,  # FD: (value < ref_lo OR value > ref_hi) → 1
                'severity':       sev,
                'result_year':    random.randint(2018, 2025),
            })
            lab_id += 1

lab_df = pd.DataFrame(lab_rows)
print(f"lab_results: {len(lab_df)} rows\n")

# ─────────────────────────────────────────────────────────────
# SCHEMA SQLite
# ─────────────────────────────────────────────────────────────
SCHEMA = """
DROP TABLE IF EXISTS lab_results;
DROP TABLE IF EXISTS prescriptions;
DROP TABLE IF EXISTS medical_visits;
DROP TABLE IF EXISTS social_links;
DROP TABLE IF EXISTS consumer_profiles;
DROP TABLE IF EXISTS insurance_policies;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS persons;

-- ── ORACLE (lato pubblico, noto all'attaccante) ───────────────

CREATE TABLE persons (
    person_id         INTEGER PRIMARY KEY,
    tax_code          TEXT    UNIQUE NOT NULL,
    first_name        TEXT    NOT NULL,
    last_name         TEXT    NOT NULL,
    birth_year        INTEGER NOT NULL,
    birth_month       INTEGER NOT NULL CHECK(birth_month BETWEEN 1 AND 12),
    birth_day         INTEGER NOT NULL CHECK(birth_day   BETWEEN 1 AND 31),
    sex               TEXT    NOT NULL CHECK(sex IN ('M','F')),
    city              TEXT    NOT NULL,
    zip               TEXT    NOT NULL,
    province          TEXT    NOT NULL,  -- FD: zip → province
    region            TEXT    NOT NULL,  -- FD: zip → region
    job               TEXT    NOT NULL,
    education         TEXT    NOT NULL,
    marital_status    TEXT    NOT NULL,
    phone             TEXT,
    email             TEXT,
    annual_income_k   INTEGER NOT NULL,
    income_bracket    TEXT    NOT NULL
);
CREATE INDEX idx_persons_zip    ON persons(zip);
CREATE INDEX idx_persons_region ON persons(region);
CREATE INDEX idx_persons_city   ON persons(city);

CREATE TABLE insurance_policies (
    policy_id       INTEGER PRIMARY KEY,
    person_id       INTEGER NOT NULL REFERENCES persons(person_id),
    insurer         TEXT    NOT NULL,
    policy_type     TEXT    NOT NULL CHECK(policy_type IN ('Base','Standard','Premium')),
    annual_premium  REAL    NOT NULL,
    coverage_level  TEXT    NOT NULL,   -- FD: policy_type → coverage_level
    deductible      REAL    NOT NULL,   -- FD: coverage_level → deductible
    copay_pct       REAL    NOT NULL,   -- FD: coverage_level → copay_pct
    issue_year      INTEGER NOT NULL,
    expiry_year     INTEGER NOT NULL,
    active          INTEGER NOT NULL CHECK(active IN (0,1))
);
CREATE INDEX idx_ins_person ON insurance_policies(person_id);

CREATE TABLE consumer_profiles (
    profile_id         INTEGER PRIMARY KEY,
    person_id          INTEGER NOT NULL REFERENCES persons(person_id),
    annual_income_k    INTEGER NOT NULL,
    income_bracket     TEXT    NOT NULL,
    spending_score     INTEGER NOT NULL CHECK(spending_score BETWEEN 1 AND 100),
    preferred_segment  TEXT    NOT NULL,   -- FD: spending_score × income_bracket → segment
    loyalty_tier       TEXT    NOT NULL CHECK(loyalty_tier IN ('Bronze','Silver','Gold','Platinum')),
    loyalty_discount   REAL    NOT NULL,   -- FD: loyalty_tier → loyalty_discount
    preferred_category TEXT    NOT NULL,
    num_purchases_yr   INTEGER NOT NULL
);
CREATE INDEX idx_cons_person ON consumer_profiles(person_id);

CREATE TABLE social_links (
    link_id      INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL REFERENCES persons(person_id),
    target_id    INTEGER NOT NULL REFERENCES persons(person_id),
    platform     TEXT    NOT NULL CHECK(platform IN ('facebook','twitter')),
    link_type    TEXT    NOT NULL CHECK(link_type IN ('friend','follows')),
    strength     TEXT    NOT NULL,
    created_year INTEGER NOT NULL
);
CREATE INDEX idx_sl_source ON social_links(source_id);
CREATE INDEX idx_sl_target ON social_links(target_id);

-- ── MEDICO (lato privato, target dell'attacco) ─────────────────

CREATE TABLE patients (
    patient_id           INTEGER PRIMARY KEY,
    person_id            INTEGER NOT NULL REFERENCES persons(person_id),
    -- QI: valori ESATTI — l'anonymizer li generalizzerà a runtime
    age                  INTEGER NOT NULL,
    sex                  TEXT    NOT NULL CHECK(sex IN ('M','F')),
    city                 TEXT    NOT NULL,
    zip                  TEXT    NOT NULL,
    province             TEXT    NOT NULL,   -- FD: zip → province
    region               TEXT    NOT NULL,   -- FD: zip → region
    job                  TEXT    NOT NULL,
    bmi                  REAL    NOT NULL,
    smoking_status       TEXT    NOT NULL CHECK(smoking_status IN ('Sì','No')),
    physical_activity    TEXT    NOT NULL,
    -- SA: attributi sensibili, valori esatti
    diagnosis            TEXT    NOT NULL,
    treatment            TEXT    NOT NULL,   -- FD: diagnosis → treatment
    drug_class           TEXT    NOT NULL,   -- FD: diagnosis → drug_class
    insurance_tier       TEXT    NOT NULL CHECK(insurance_tier IN ('Base','Standard','Premium')),
    num_annual_visits    INTEGER NOT NULL,
    hba1c                REAL    NOT NULL,
    ldl                  INTEGER NOT NULL,
    glicemia_digiuno     INTEGER NOT NULL,
    pressione_sistolica  INTEGER NOT NULL,
    risk_score           INTEGER NOT NULL CHECK(risk_score BETWEEN 1 AND 10)
);
CREATE INDEX idx_pat_person ON patients(person_id);
CREATE INDEX idx_pat_city   ON patients(city);
CREATE INDEX idx_pat_zip    ON patients(zip);

CREATE TABLE medical_visits (
    visit_id      INTEGER PRIMARY KEY,
    patient_id    INTEGER NOT NULL REFERENCES patients(patient_id),
    visit_year    INTEGER NOT NULL,
    visit_month   INTEGER NOT NULL CHECK(visit_month BETWEEN 1 AND 12),
    visit_type    TEXT    NOT NULL,
    department    TEXT    NOT NULL,   -- FD prob.: diagnosis → department
    hospital_tier TEXT    NOT NULL,
    duration_days INTEGER NOT NULL,
    outcome       TEXT    NOT NULL,
    cost_bracket  TEXT    NOT NULL    -- FD: visit_type × duration → cost_bracket
);
CREATE INDEX idx_mv_patient ON medical_visits(patient_id);

CREATE TABLE prescriptions (
    rx_id            INTEGER PRIMARY KEY,
    patient_id       INTEGER NOT NULL REFERENCES patients(patient_id),
    drug_class       TEXT    NOT NULL,   -- FD: diagnosis → drug_class
    drug_name        TEXT    NOT NULL,   -- FD: drug_class → drug_name (1-to-many)
    dosage_mg        INTEGER NOT NULL,
    duration_days    INTEGER NOT NULL,
    prescribing_year INTEGER NOT NULL,
    refills          INTEGER NOT NULL,
    is_chronic       INTEGER NOT NULL CHECK(is_chronic IN (0,1))  -- FD: drug_class → is_chronic
);
CREATE INDEX idx_rx_patient ON prescriptions(patient_id);

CREATE TABLE lab_results (
    result_id       INTEGER PRIMARY KEY,
    patient_id      INTEGER NOT NULL REFERENCES patients(patient_id),
    test_type       TEXT    NOT NULL,
    value           REAL    NOT NULL,
    unit            TEXT    NOT NULL,    -- FD: test_type → unit
    reference_low   REAL    NOT NULL,    -- FD: test_type → reference_low
    reference_high  REAL    NOT NULL,    -- FD: test_type → reference_high
    abnormal_flag   INTEGER NOT NULL CHECK(abnormal_flag IN (0,1)),
    severity        TEXT    NOT NULL,
    result_year     INTEGER NOT NULL
);
CREATE INDEX idx_lab_patient ON lab_results(patient_id);
CREATE INDEX idx_lab_type    ON lab_results(test_type);
"""

# ─────────────────────────────────────────────────────────────
# WRITE SQLite + CSV
# ─────────────────────────────────────────────────────────────
print(f"Writing SQLite → {DB_PATH}")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
conn = sqlite3.connect(DB_PATH)
conn.executescript(SCHEMA)
conn.commit()

tables = {
    'persons':            persons_df,
    'insurance_policies': insurance_df,
    'consumer_profiles':  consumer_df,
    'social_links':       social_df,
    'patients':           patients_df,
    'medical_visits':     visits_df,
    'prescriptions':      rx_df,
    'lab_results':        lab_df,
}
for name, df in tables.items():
    df.to_sql(name, conn, if_exists='append', index=False)
    csv_path = os.path.join(OUT_DIR, f'{name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  {name:25s}  {len(df):7d} rows  →  {os.path.basename(csv_path)}")
conn.close()

# ─────────────────────────────────────────────────────────────
# VERIFICA FD
# ─────────────────────────────────────────────────────────────
print("\n── FD violations (tutti devono essere 0) ──")
checks = [
    ("diagnosis → treatment",       patients_df,   'diagnosis',     'treatment'),
    ("diagnosis → drug_class",      patients_df,   'diagnosis',     'drug_class'),
    ("zip → city (patients)",       patients_df,   'zip',           'city'),
    ("zip → province (patients)",   patients_df,   'zip',           'province'),
    ("zip → region (patients)",     patients_df,   'zip',           'region'),
    ("zip → city (persons)",        persons_df,    'zip',           'city'),
    ("zip → province (persons)",    persons_df,    'zip',           'province'),
    ("zip → region (persons)",      persons_df,    'zip',           'region'),
    ("policy_type → coverage",      insurance_df,  'policy_type',   'coverage_level'),
    ("coverage → deductible",       insurance_df,  'coverage_level','deductible'),
    ("loyalty_tier → discount",     consumer_df,   'loyalty_tier',  'loyalty_discount'),
    ("test_type → unit",            lab_df,        'test_type',     'unit'),
]
for label, df, det, dep in checks:
    n = (df.groupby(det)[dep].nunique() > 1).sum()
    print(f"  {label:40s} {'OK' if n==0 else f'FAIL ({n})'}")

flag_err = lab_df[
    ((lab_df['value'] < lab_df['reference_low']) | (lab_df['value'] > lab_df['reference_high']))
    != lab_df['abnormal_flag'].astype(bool)
]
print(f"  {'abnormal_flag consistency':40s} {'OK' if len(flag_err)==0 else f'FAIL ({len(flag_err)})'}")

# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────
print("\n── Stats ──")
print("Diagnosi (patients):")
print(patients_df['diagnosis'].value_counts().to_string())
print("\nEtà media per diagnosi:")
print(patients_df.groupby('diagnosis')['age'].mean().round(1).to_string())
print("\nBMI medio per diagnosi:")
print(patients_df.groupby('diagnosis')['bmi'].mean().round(1).to_string())
print("\nHbA1c medio per diagnosi:")
print(patients_df.groupby('diagnosis')['hba1c'].mean().round(2).to_string())
print("\nPolizze per tipo (insurance):")
print(insurance_df['policy_type'].value_counts().to_string())
print("\nLink social per platform:")
print(social_df['platform'].value_counts().to_string())
print(f"\nDone. DB: {DB_PATH}")

# ─────────────────────────────────────────────────────────────
# REGISTRY FD (per il loader SMT)
# ─────────────────────────────────────────────────────────────
print("\n── FD Registry (input per il solver SMT) ──")
FD_REGISTRY = [
    # Tabella patients (lato medico)
    ("patients",      "diagnosis",                "treatment",           "1-to-1"),
    ("patients",      "diagnosis",                "drug_class",          "1-to-1"),
    ("patients",      "zip",                      "city",                "1-to-1"),
    ("patients",      "zip",                      "province",            "1-to-1"),
    ("patients",      "zip",                      "region",              "1-to-1"),
    # Tabella prescriptions (lato medico)
    ("prescriptions", "drug_class",               "is_chronic",          "1-to-1"),
    # Tabella persons (lato oracle)
    ("persons",       "zip",                      "city",                "1-to-1"),
    ("persons",       "zip",                      "province",            "1-to-1"),
    ("persons",       "zip",                      "region",              "1-to-1"),
    # Tabella insurance_policies (lato oracle)
    ("insurance",     "policy_type",              "coverage_level",      "1-to-1"),
    ("insurance",     "coverage_level",           "deductible",          "1-to-1"),
    ("insurance",     "coverage_level",           "copay_pct",           "1-to-1"),
    # Tabella consumer_profiles (lato oracle)
    ("consumer",      "loyalty_tier",             "loyalty_discount",    "1-to-1"),
    # Tabella lab_results (lato medico)
    ("lab_results",   "test_type",                "unit",                "1-to-1"),
    ("lab_results",   "test_type",                "reference_low",       "1-to-1"),
    ("lab_results",   "test_type",                "reference_high",      "1-to-1"),
    ("lab_results",   "value vs ref_range",       "abnormal_flag",       "rule-based"),
]
print(f"  {'Table':20s} {'Determinant':28s} {'Dependent':20s} {'Type'}")
print(f"  {'-'*80}")
for t, det, dep, typ in FD_REGISTRY:
    print(f"  {t:20s} {det:28s} → {dep:20s} [{typ}]")
