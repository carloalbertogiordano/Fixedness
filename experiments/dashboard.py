"""
dashboard.py — Fixedness Privacy Audit Dashboard
Avvio: python experiments/dashboard.py  (da project root)
Apri:  http://localhost:8050
"""
import os, sys, glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

GF_URL = 'https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,700&family=Lora:ital@0;1&display=swap'

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, '..', 'results')


def _latest_csv(prefix):
    for d in reversed(sorted(glob.glob(os.path.join(RESULTS_DIR, f'{prefix}_*/')))):
        csvs = glob.glob(os.path.join(d, '*.csv'))
        if csvs:
            return pd.read_csv(csvs[0])
    return None


def _latest_audit_rho():
    """Read rho and effective_rho from the latest full_audit.csv (single run values)."""
    for d in reversed(sorted(glob.glob(os.path.join(RESULTS_DIR, '*/')))):
        csvs = glob.glob(os.path.join(d, 'full_audit.csv'))
        if csvs:
            df = pd.read_csv(csvs[0])
            if 'rho' in df.columns:
                return {
                    'rho':          float(df['rho'].iloc[0]),
                    'effective_rho': float(df['effective_rho'].iloc[0]) if 'effective_rho' in df.columns else None,
                    'n_records':    len(df),
                }
    return None


CP   = _latest_csv('crossproduct')
MS   = _latest_csv('multiseed')
QI   = _latest_csv('qi')
BKSA = _latest_csv('bksa')
TIM  = _latest_csv('timing')
ORA  = _latest_csv('oracle')
RHO       = _latest_csv('rho')
RHO_QI    = _latest_csv('rho_qi')
AUDIT_RHO = _latest_audit_rho()   # rho from latest full_audit.csv (actual run)

# Normalizza nomi colonne timing vecchio formato
if TIM is not None and 'mean_ms' in TIM.columns:
    TIM = TIM.rename(columns={
        'mean_ms':  'mean_solve_ms', 'median_ms': 'mean_median_ms',
        'p95_ms':   'mean_p95_ms',   'max_ms':    'mean_max_ms',
        'build_ms': 'mean_build_ms',
    })

# ── Tema ─────────────────────────────────────────────────────────────────────

PRIMARY = '#2C1810'
ACCENT  = '#B8753A'
ACCENT2 = '#7A4520'
LIGHT   = '#E8D5B7'
BG      = '#c8b49a'
CARD    = '#FDFAF5'
TEXT    = '#1A0E08'
MUTED   = '#4a3828'
GRID    = '#E4D0B4'

FONT_SERIF = 'Playfair Display, Georgia, serif'
FONT_BODY  = 'Lora, Segoe UI, Arial, sans-serif'

MC = {
    'no_anonymization':    '#B83232',  # cremisi
    'mondrian_k':          '#2C6FAC',  # blu acciaio
    'l_diversity':         '#2E7D52',  # verde bosco
    't_closeness':         '#C95F1A',  # arancio bruciato
    'microaggregation':    '#7B3FA2',  # viola
    'noise_addition':      '#0E7C7B',  # verde acqua
    'laplace_dp':          '#A0522D',  # sienna
    'local_dp':            '#5F6B6B',  # grigio ardesia
    'suppression':         '#1C3A6E',  # blu notte
    'randomized_response': '#3D7A4A',  # verde salvia scuro
}
METHODS = list(MC.keys())

METRICS = [
    {'label': 'Fixedness (F)',    'value': 'mean_fixedness'},
    {'label': 'Sponginess (S)',   'value': 'mean_sponginess'},
    {'label': 'Effective ρ',      'value': 'effective_rho'},
    {'label': 'Candidati',        'value': 'mean_candidates'},
    {'label': 'Conflitti Z3',     'value': 'mean_conflicts'},
    {'label': 'Decisioni Z3',     'value': 'mean_decisions'},
    {'label': 'Propagazioni Z3',  'value': 'mean_propagations'},
    {'label': 'Solve time (ms)',  'value': 'mean_solve_ms'},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fc(**kw):
    return dict(
        template='plotly_white',
        paper_bgcolor=CARD,
        plot_bgcolor='#FAF4EC',
        font=dict(color=TEXT, size=12, family=FONT_BODY),
        title_font=dict(family=FONT_SERIF, color=PRIMARY, size=14),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT)),
        margin=dict(l=60, r=20, t=54, b=50),
        xaxis=dict(gridcolor=GRID, linecolor=LIGHT, tickfont_color=MUTED, title_font_color=MUTED),
        yaxis=dict(gridcolor=GRID, linecolor=LIGHT, tickfont_color=MUTED, title_font_color=MUTED),
        **kw,
    )

def _empty(msg='Dati non disponibili — esegui prima gli sweep.'):
    fig = go.Figure()
    fig.update_layout(annotations=[dict(text=msg, xref='paper', yref='paper', x=0.5, y=0.5,
                                        showarrow=False, font=dict(color=MUTED, size=14,
                                                                    family=FONT_SERIF))], **_fc())
    return fig

def _err_bar(x, y, e, name, color):
    y_l = [float(v) if pd.notna(v) else None for v in y]
    e_l = [float(v) if pd.notna(v) else 0    for v in e]
    return go.Scatter(x=list(x), y=y_l, mode='lines+markers', name=name,
                      line=dict(color=color, width=2.5), marker=dict(size=7, line=dict(width=1, color='white')),
                      error_y=dict(type='data', array=e_l, visible=True,
                                   color=color, thickness=1.5, width=5))

def _metric_label(m):
    return next((o['label'] for o in METRICS if o['value'] == m), m)

def _dd(id_, opts, val, **kw):
    return dcc.Dropdown(id=id_, options=opts, value=val,
                        style={'backgroundColor': CARD, 'color': TEXT,
                               'border': f'1px solid {LIGHT}', 'minWidth': '160px',
                               'fontFamily': FONT_BODY, 'fontSize': '12px'}, **kw)

def _lbl(t):
    return html.Label(t, style={'color': ACCENT2, 'fontSize': '10px', 'fontWeight': '700',
                                 'textTransform': 'uppercase', 'letterSpacing': '0.08em',
                                 'display': 'block', 'marginBottom': '4px',
                                 'fontFamily': FONT_BODY})

def _card(*ch):
    return html.Div(list(ch), style={
        'background': CARD, 'borderRadius': '6px', 'padding': '20px 22px', 'margin': '8px 0',
        'boxShadow': '0 1px 4px rgba(44,24,16,0.10)', 'borderTop': f'3px solid {ACCENT}',
    })

def _desc(t):
    return html.P(t, style={
        'color': MUTED, 'fontSize': '13px', 'lineHeight': '1.75',
        'borderLeft': f'3px solid {ACCENT}', 'paddingLeft': '14px',
        'margin': '0 0 16px', 'fontStyle': 'italic', 'fontFamily': FONT_BODY,
    })

def _row(*ch):
    return html.Div(list(ch), style={'display': 'flex', 'gap': '14px',
                                      'flexWrap': 'wrap', 'marginBottom': '10px'})

def _col(*ch, flex='1', mn='180px'):
    return html.Div(list(ch), style={'flex': flex, 'minWidth': mn})

def _graph(id_, h=420):
    return dcc.Graph(id=id_, style={'height': f'{h}px'},
                     config={'displayModeBar': True, 'scrollZoom': True})

# ── Tab content (tutti pre-renderizzati nel DOM) ──────────────────────────────

def _overview():
    k_opts = [] if CP is None else [{'label': f'k={k}', 'value': k}
                                     for k in sorted(CP['k'].unique())]
    return _card(
        _desc("Questo grafico risponde alla domanda: a parità di k, quale metodo protegge di più? "
              "Le barre sono ordinate dal metodo più vulnerabile (sinistra) al più protetto (destra). "
              "La Fixedness F∈[0,1] è una garanzia formale: F=1.0 significa che il solver Z3 ha dimostrato "
              "con certezza logica che solo una persona nell'oracolo è compatibile con il record — "
              "re-identificazione deterministica, non probabilistica. F=0 significa che esistono almeno "
              "due identità compatibili: l'attaccante non può essere certo. "
              "La Sponginess S∈[0,1] misura quante identità alternative sopravvivono: S≈1 indica che "
              "il record è compatibile con molti candidati (anonimato forte), S=0 indica unicità assoluta. "
              "I metodi full-range (suppression, randomized_response, local_dp) saturano sempre S=1 "
              "perché generalizzano completamente i QI: il solver non riceve vincoli utili e tratta tutti "
              "i candidati come ugualmente possibili. Questo non è protezione intelligente — è cecità del "
              "sistema. Le barre di errore mostrano ±1 std tra i seed Monte Carlo."),
        _row(
            _col(_lbl('Metrica'), _dd('ov-metric', METRICS, 'mean_sponginess')),
            _col(_lbl('Valore di k'), _dd('ov-k', k_opts, 5 if k_opts else None)),
        ),
        _graph('ov-chart', 500),
    )


def _phase():
    k_vals = [] if CP is None else sorted(CP['k'].unique())
    k_opts = [{'label': f'k={k}', 'value': k} for k in k_vals]
    return _card(
        _desc("La transizione di fase è il fenomeno centrale: esiste un valore critico di k oltre il quale "
              "la fixedness crolla bruscamente a zero. Sotto quella soglia il solver Z3 riesce a eliminare "
              "tutti i candidati alternativi (regime UNSAT); sopra non ci riesce più (regime SAT). "
              "Il grafico a sinistra mostra questa curva per ogni metodo — l'occhio cerca il punto di "
              "ginocchio, dove la linea scende da un plateau positivo a zero. "
              "Mondrian k=2: F≈0.16 (16% dei record ha un'unica identità compatibile). A k=5 F≈0 perché "
              "le classi di equivalenza sono abbastanza grandi da garantire almeno due candidati. "
              "I metodi full-range (suppression, randomized_response, local_dp) non hanno transizione: "
              "F=0 per ogni k perché rimuovono completamente i QI — il solver parte già cieco. "
              "Il heatmap a destra mostra la stessa informazione su griglia (metodo × k): "
              "celle rosse = fixedness positiva = rischio reale, celle bianche/blu = protezione. "
              "Leggi il heatmap colonna per colonna: a k=2, quali metodi hanno ancora rischio?"),
        _row(
            _col(_lbl('Metrica'), _dd('ph-metric', METRICS[:4], 'mean_fixedness')),
            _col(_lbl('Metodi'), _dd('ph-methods', [{'label': m, 'value': m} for m in METHODS],
                                     METHODS, multi=True), flex='3', mn='300px'),
            _col(_lbl('Filtra k'), _dd('ph-k-filter', k_opts, k_vals, multi=True), flex='2', mn='200px'),
        ),
        _row(_col(_graph('ph-lines', 380)), _col(_graph('ph-heat', 380))),
    )


def _gap():
    k_vals    = [] if CP is None else sorted(CP['k'].unique())
    k_opts    = [{'label': f'k={k}', 'value': k} for k in k_vals]
    default_k = k_opts[2]['value'] if len(k_opts) > 2 else (k_opts[0]['value'] if k_opts else None)
    return _card(
        _desc("Questo tab rivela il meccanismo interno del solver: come cambia il suo lavoro al variare di k? "
              "I Conflitti Z3 (asse sinistra, linee piene) misurano quante volte il solver ha trovato una "
              "contraddizione e ha dovuto tornare indietro (backtracking). Conflitti alti = il solver sta "
              "cercando attivamente di eliminare candidati uno a uno = regime UNSAT = record probabilmente "
              "univoco = rischio reale. Le Decisioni Z3 (asse destra, linee tratteggiate) contano quante "
              "scelte arbitrarie ha fatto il solver prima di trovare una soluzione. Decisioni alte = il "
              "solver trova facilmente molte soluzioni = regime SAT = record ben protetto. "
              "Il punto di crossover — dove i conflitti scendono e le decisioni salgono — corrisponde "
              "esattamente alla soglia della transizione di fase vista nel tab precedente. "
              "È la firma computazionale della transizione: non si vede nei dati di privacy (F, S), "
              "ma si vede nel comportamento del solver. "
              "Il bar chart a destra mostra quale metodo genera più conflitti a k fisso: "
              "più conflitti = quel metodo, a quel k, espone record che il solver riesce a vincolare fortemente."),
        _row(
            _col(_lbl('Metodi (linee su k)'), _dd('gap-method', [{'label': m, 'value': m} for m in METHODS],
                                                    METHODS, multi=True), flex='3', mn='300px'),
            _col(_lbl('Filtra k (linee)'), _dd('gap-k-filter', k_opts, k_vals, multi=True), flex='2', mn='200px'),
            _col(_lbl('k fisso (bar per metodo)'), _dd('gap-k', k_opts, default_k)),
        ),
        _row(_col(_graph('gap-lines', 380)), _col(_graph('gap-bar', 380))),
    )


def _sens():
    return _card(
        _desc("Quanto è sensibile il rischio a ciascun parametro operativo? "
              "Rumore oracolo: è il fattore critico. Con oracle_noise=0 l'oracolo contiene solo pazienti reali "
              "→ ogni record tende a un'unica corrispondenza. All'aumentare del rumore l'oracolo si 'inquina' "
              "con persone non pazienti: il solver non riesce più a escluderle → fixedness crolla. "
              "Già oracle_noise=25 è sufficiente a portare F≈0 per la maggior parte dei metodi. "
              "BK fraction: quota del registro anagrafico in possesso dell'attaccante (0=nessuna, 1=tutto). "
              "Agisce come constraint-tightener: più background knowledge, più candidati esclusi — ma non "
              "forza il collasso deterministico finché le classi di equivalenza sono abbastanza grandi. "
              "N pazienti: effetto paradossale. Dataset più grandi producono intervalli di generalizzazione "
              "più stretti → vincoli più precisi → a volte più re-identificabilità, non meno. "
              "Questo contraddice l'intuizione che 'più dati = più privacy' ed è una vulnerabilità "
              "che k-anonymity standard non cattura. "
              "k — tutti i metodi: mostra una linea per ogni metodo che usa k come parametro "
              "(mondrian, l-diversity, t-closeness, suppression, microaggregation). "
              "Confronta la pendenza: metodi con transizione più ripida proteggono meglio già a k piccolo. "
              "I metodi full-range non appaiono perché il loro k non influenza la generalizzazione dei QI. "
              "Per oracle_noise, bk_frac, n_patients il grafico è dual-axis: metrica sinistra e destra "
              "su scale separate, permettendo di vedere due metriche correlate simultaneamente."),
        _row(
            _col(_lbl('Parametro'), _dd('sens-param', [
                {'label': 'Rumore oracolo',       'value': 'oracle_noise'},
                {'label': 'BK fraction',           'value': 'bk_frac'},
                {'label': 'N pazienti',            'value': 'n_patients'},
                {'label': 'k — tutti i metodi',   'value': 'k'},
            ], 'oracle_noise')),
            _col(_lbl('Metrica sinistra'), _dd('sens-m1', METRICS, 'mean_fixedness')),
            _col(_lbl('Metrica destra'),   _dd('sens-m2', METRICS, 'mean_sponginess')),
        ),
        _graph('sens-chart', 460),
    )


def _dp():
    return _card(
        _desc("Ogni metodo di questo tab ha un parametro che controlla il trade-off privacy/utilità. "
              "ε (epsilon) — budget di privacy differenziale: ε→0 = massimo rumore = massima protezione, "
              "ε→∞ = nessun rumore = nessuna protezione. La soglia standard in letteratura è ε=1.0. "
              "Laplace DP (centrale): il rumore viene aggiunto dopo l'aggregazione. All'aumentare di ε "
              "la sponginess scende monotonamente perché il record pubblicato si avvicina al valore reale "
              "→ il solver può escludere più candidati. "
              "Local DP: il rumore viene aggiunto prima, su ogni valore individuale con scala d/ε (d=dominio). "
              "Rimane al tetto S≈1 per tutti i valori testati: con domini piccoli (d≤25) il rumore locale "
              "è così denso che il solver non riesce a escludere nessun candidato. "
              "σ (sigma) — deviazione standard per noise_addition gaussiano: più alto = più rumore = meno vincoli. "
              "l — l-diversity: numero minimo di valori distinti del SA per gruppo. "
              "Valori più alti = gruppi più diversificati = meno correlazione SA/QI = meno informazione per il solver. "
              "t — t-closeness: soglia sulla distanza tra distribuzione SA nel gruppo e nel dataset globale. "
              "t piccolo = il gruppo deve rispecchiare fedelmente la distribuzione globale = meno leakage. "
              "La linea tratteggiata verticale indica il valore default usato negli altri sweep."),
        _row(
            _col(_lbl('Parametro di privacy'), _dd('dp-param', [
                {'label': 'ε — Laplace DP vs Local DP', 'value': 'epsilon'},
                {'label': 'σ — Noise Addition',          'value': 'sigma'},
                {'label': 'l — l-diversity',              'value': 'l'},
                {'label': 't — t-closeness',              'value': 't'},
            ], 'epsilon')),
            _col(_lbl('Metrica'), _dd('dp-metric', METRICS[:6], 'mean_sponginess')),
        ),
        _graph('dp-chart', 460),
    )


def _qi():
    return _card(
        _desc("I quasi-identificatori (QI) sono gli attributi che l'attaccante conosce da fonti pubbliche "
              "(registro anagrafico, social, telefonate). Ogni QI diventa un vincolo nel sistema SMT: "
              "il solver cerca persone nell'oracolo che abbiano esattamente quel valore (o range, dopo "
              "generalizzazione). Più QI = più vincoli = pool di candidati più ristretto. "
              "Esempio concreto: con solo il CAP (Zip) ci sono molte persone compatibili → S alta. "
              "Aggiungendo età, sesso, Zip simultaneamente il pool si riduce drasticamente. "
              "Con 5 QI (tutti disponibili) la protezione offerta da k=10 può degradare al livello "
              "di k=2 con 1 QI solo: l'attaccante con più informazione pubblica bypassa la k-anonymity. "
              "Questo è il risultato chiave del tab: k protegge contro un avversario con QI limitati; "
              "aumentare k non compensa un avversario con molti QI. "
              "Vista linee: una curva per QI set, asse x = k. Vista heatmap: vede l'intera griglia. "
              "Cerca celle anomale: QI set con pochi attributi ma alta fixedness a k piccolo."),
        _row(
            _col(_lbl('Metrica'),   _dd('qi-metric', METRICS[:5], 'mean_sponginess')),
            _col(_lbl('Vista'),     _dd('qi-view', [
                {'label': 'Linee (metric vs k per QI set)', 'value': 'lines'},
                {'label': 'Heatmap (QI set × k)',           'value': 'heat'},
            ], 'lines')),
        ),
        _graph('qi-chart', 480),
    )


def _bksa():
    return _card(
        _desc("Il Background Knowledge SA è il tipo di informazione sensibile che l'attaccante ricava "
              "dall'oracolo. Il dataset di test ha più attributi SA (Diagnosis, altri). Questo tab chiede: "
              "se l'attaccante sa che la persona che cerca ha una certa diagnosi, quanto cambia il rischio? "
              "Attributi SA con distribuzione sbilanciata (es. Diagnosis: 5 classi non equiprobabili, "
              "correlate all'età) danno vincoli più discriminativi: il solver può escludere candidati "
              "dell'oracolo che hanno SA incompatibili. Un SA 'raro' (pochi pazienti con quella diagnosi) "
              "è più pericoloso: riduce il pool a pochissimi candidati, potenzialmente uno solo. "
              "Attenzione alla copertura effettiva: le entry di rumore nell'oracolo (non-pazienti) hanno "
              "SA=UNKNOWN. Quindi il BK fraction nominale (es. 0.9) non corrisponde alla copertura reale: "
              "solo i veri pazienti portano informazione SA. La copertura effettiva è "
              "BK_frac × N_patients / (N_patients + oracle_noise), tipicamente molto più bassa. "
              "Vista linee: una curva per SA type, mostra se SA diversi danno profili di rischio diversi. "
              "Vista heatmap: confronto diretto su griglia SA × k."),
        _row(
            _col(_lbl('Metrica'), _dd('bksa-metric', METRICS[:5], 'mean_sponginess')),
            _col(_lbl('Vista'),   _dd('bksa-view', [
                {'label': 'Linee (metric vs k per SA)', 'value': 'lines'},
                {'label': 'Heatmap (SA × k)',           'value': 'heat'},
            ], 'lines')),
        ),
        _graph('bksa-chart', 480),
    )


def _oracle():
    grid_method = 'mondrian_k'
    if ORA is not None and 'method' in ORA.columns:
        grids = ORA[ORA['sweep_type'] == 'grid']['method'].unique()
        if len(grids):
            grid_method = grids[0]
    return _card(
        _desc("Questo tab analizza come il rischio di re-identificazione dipende dalla qualità e dalla "
              "dimensione dell'oracolo in possesso dell'attaccante. "
              "L'oracolo è il registro di riferimento: contiene i veri pazienti più oracle_noise righe "
              "civetta (persone non-pazienti estratte casualmente dalla popolazione). "
              "Rumore oracolo (noise × method): mostra come ogni metodo reagisce all'aggiunta di civette. "
              "I metodi full-range (suppression, local_dp, randomized_response) rimangono a F=0 "
              "indipendentemente dal rumore — il solver è già cieco per costruzione. "
              "I metodi sintattici (mondrian, l-diversity, t-closeness) mostrano un crollo di fixedness "
              "già a oracle_noise=25÷50: bastano poche decine di civette per rendere il solver incerto. "
              "BK fraction (bkfrac × method): mostra l'effetto della conoscenza dell'attributo sensibile. "
              "Con bk_frac=0 l'attaccante non sa nulla della diagnosi dei candidati → più candidati compatibili. "
              "Con bk_frac=1 conosce la diagnosi di tutti i pazienti → vincoli più stretti → più fixedness. "
              "L'effetto è asimmetrico: bk_frac bassa aiuta poco (le civette hanno SA=UNKNOWN comunque), "
              "ma bk_frac alta fa differenza solo nei metodi che lasciano vincoli QI utili. "
              f"Heatmap griglia ({grid_method}): joint effect di oracle_noise e bk_frac. "
              "Celle rosse in alto-sinistra = oracle piccolo + attaccante informato = massimo rischio. "
              "La diagonale mostra come i due effetti si compensano o si amplificano."),
        _row(
            _col(_lbl('Vista'), _dd('ora-view', [
                {'label': 'Rumore oracle × metodo',     'value': 'noise'},
                {'label': 'BK fraction × metodo',        'value': 'bkfrac'},
                {'label': f'Heatmap griglia ({grid_method})', 'value': 'grid'},
            ], 'noise')),
            _col(_lbl('Metrica'), _dd('ora-metric', METRICS[:4], 'mean_fixedness')),
            _col(_lbl('Metodi (linee)'), _dd('ora-methods',
                      [{'label': m, 'value': m} for m in METHODS],
                      METHODS, multi=True), flex='3', mn='300px'),
        ),
        _graph('ora-chart', 500),
    )


def _k_variance():
    return _card(
        _desc("Come varia una metrica al crescere di k, per ogni metodo? "
              "Ogni linea è un metodo; l'asse x è k (scala logaritmica); l'asse y è la metrica scelta. "
              "Le barre di errore mostrano ±1 std tra i seed Monte Carlo. "
              "I metodi che non rispondono a k (suppression, local_dp, randomized_response, noise_addition, laplace_dp) "
              "appaiono come linee piatte — la loro metrica è indipendente dal parametro k di anonimizzazione. "
              "Usa questo tab per identificare la soglia di k oltre la quale la metrica si stabilizza "
              "e per confrontare la pendenza di metodi diversi nella zona di transizione."),
        _row(
            _col(_lbl('Metrica'), _dd('kv-metric', METRICS, 'mean_fixedness')),
            _col(_lbl('Metodi'),  _dd('kv-methods', [{'label': m, 'value': m} for m in METHODS],
                                      METHODS, multi=True), flex='3', mn='300px'),
        ),
        _graph('kv-chart', 500),
    )


def _runtime():
    tim_params = [] if TIM is None else [{'label': p, 'value': p}
                                          for p in TIM['sweep_param'].unique()]
    default_p = tim_params[0]['value'] if tim_params else None
    return _card(
        _desc("Questo tab risponde a: è fattibile usare questo sistema su dati reali? "
              "Il tempo di audit ha due componenti. Build solver: costruisce il sistema SMT con tutti i "
              "vincoli oracle una volta per configurazione (O(oracle_size × n_FD_constraints)). "
              "Con oracle=100 → ~3ms; con oracle=7500 (registro anagrafico comunale) → ~80ms. "
              "Solve per record: per ogni record anonimizzato, Z3 cerca se esiste almeno un candidato "
              "compatibile (SAT) o dimostra che nessuno esiste (UNSAT). Scala quasi linearmente con "
              "n_patients ma rimane sotto 1ms/record nei range testati — il collo di bottiglia è il build. "
              "Le percentili p95 e max identificano i casi difficili: record in classi di equivalenza "
              "ambigue dove il solver deve esplorare più a lungo prima di decidere. "
              "Un gap grande tra mean e p95 segnala eterogeneità nei dati (alcuni record strutturalmente "
              "più ambigui di altri). "
              "Per un audit di 100k record con oracle=7500: build ≈ 80ms una volta, "
              "solve ≈ 100k × 1ms = 100s single-thread. Con 40 worker paralleli → ~2.5s wall-clock. "
              "La scalabilità orizzontale (più worker, stesso tempo per record) è il percorso corretto "
              "per la produzione: Z3 è process-safe ma non thread-safe, ogni worker usa la propria istanza."),
        _row(
            _col(_lbl('Parametro sweep'), _dd('tim-param', tim_params, default_p)),
            _col(_lbl('Metrica'), _dd('tim-metric', [
                {'label': 'Solve medio (ms)',  'value': 'mean_solve_ms'},
                {'label': 'Solve p95 (ms)',    'value': 'mean_p95_ms'},
                {'label': 'Solve max (ms)',    'value': 'mean_max_ms'},
                {'label': 'Build solver (ms)', 'value': 'mean_build_ms'},
            ], 'mean_solve_ms')),
        ),
        _graph('tim-chart', 460),
    )


def _rho():
    return _card(
        _desc("ρ = clausole_3SAT / variabili_totali: rapporto clausole-variabili dell'istanza 3-SAT. "
              "ρ > ρ* ≈ 4.27 → regime over-constrained (UNSAT tipico per random 3-SAT); ρ < ρ* → under-constrained. "
              "L'Effective ρ pesa le clausole At-Most-One per l'attività delle celle (knowledge map): "
              "w(r,a) = 1 - (range_conosciuto) / (dominio - 1). "
              "Il Retention Ratio (ρ_eff / ρ) indica quanta informazione vincolante sopravvive all'anonimizzazione."),
        _row(
            _col(_lbl('Metodo (per ρ_eff)'), _dd('rho-method', [{'label': m, 'value': m} for m in METHODS],
                                                'mondrian_k')),
            _col(_lbl('k (per ρ_eff)'), _dd('rho-k', [], None)),
        ),
        _row(
            _col(_graph('rho-chart', 460)),
            _col(_graph('rho-qi-chart', 460)),
        ),
    )


# ── Layout ────────────────────────────────────────────────────────────────────

_TAB_STYLE = {
    'fontFamily': FONT_BODY, 'fontSize': '12px', 'fontWeight': '700',
    'color': MUTED, 'textTransform': 'uppercase', 'letterSpacing': '0.06em',
    'padding': '10px 18px', 'borderBottom': f'2px solid {LIGHT}',
    'background': BG,
}
_TAB_SELECTED = {**_TAB_STYLE, 'color': PRIMARY, 'borderBottom': f'2px solid {ACCENT}',
                 'background': CARD}

app = Dash(__name__, title='Fixedness Dashboard',
           suppress_callback_exceptions=True,
           external_stylesheets=[GF_URL])

app.layout = html.Div([
    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1('Fixedness Privacy Audit',
                style={'fontFamily': FONT_SERIF, 'fontSize': '28px', 'fontWeight': '800',
                       'color': PRIMARY, 'margin': '0', 'letterSpacing': '-0.02em'}),
        html.P('Analisi interattiva degli sweep sperimentali · SMT/Z3 · k-anonymity · Differential Privacy',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '4px 0 0',
                      'fontFamily': FONT_BODY, 'fontStyle': 'italic'}),
    ], style={
        'background': CARD, 'borderBottom': f'3px solid {ACCENT}',
        'padding': '18px 28px 14px', 'boxShadow': '0 2px 8px rgba(44,24,16,0.12)',
    }),

    # ── Tabs ────────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(children=[
            dcc.Tab(label='Overview',         value='ov',   children=[_overview()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Fase Transizione', value='ph',   children=[_phase()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Gap Complessità',  value='gap',  children=[_gap()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Sensitività',      value='sens', children=[_sens()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Budget Privacy',   value='dp',   children=[_dp()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='QI Richness',      value='qi',   children=[_qi()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='BK SA',            value='bksa', children=[_bksa()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Oracle',           value='ora',  children=[_oracle()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='k Variance',        value='kv',   children=[_k_variance()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Runtime',          value='tim',  children=[_runtime()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label='Phase Transition ρ', value='rho', children=[_rho()],
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
        ], colors={'border': LIGHT, 'primary': ACCENT, 'background': BG}),
    ], style={'background': BG, 'padding': '0 28px 40px'}),

], style={'background': BG, 'minHeight': '100vh', 'fontFamily': FONT_BODY})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(Output('ov-chart', 'figure'),
              Input('ov-metric', 'value'), Input('ov-k', 'value'))
def cb_overview(metric, k):
    if CP is None or not metric or k is None:
        return _empty()
    df = CP[CP['k'] == k].copy()
    if df.empty:
        return _empty(f'Nessun dato per k={k}')
    std_col = metric.replace('mean_', 'std_')
    df = df.sort_values(metric)
    fig = go.Figure(go.Bar(
        x=df[metric], y=df['method'], orientation='h',
        marker_color=[MC.get(m, '#888') for m in df['method']],
        error_x=dict(type='data', array=df[std_col].tolist() if std_col in df else None, visible=True),
        text=df[metric].round(3), textposition='outside', textfont_color=TEXT,
    ))
    fig.update_layout(title=f'{_metric_label(metric)} per metodo  (k={k})',
                      xaxis_title=_metric_label(metric), yaxis_title='',
                      showlegend=False, **_fc(height=500))
    return fig


@app.callback(Output('ph-lines', 'figure'), Output('ph-heat', 'figure'),
              Input('ph-metric', 'value'),  Input('ph-methods', 'value'),
              Input('ph-k-filter', 'value'))
def cb_phase(metric, methods, k_filter):
    if CP is None:
        return _empty(), _empty()
    methods  = methods or METHODS
    k_filter = k_filter or sorted(CP['k'].unique())
    df  = CP[CP['method'].isin(methods) & CP['k'].isin(k_filter)]
    lbl = _metric_label(metric)
    std = metric.replace('mean_', 'std_')

    fig_l = go.Figure()
    for m in methods:
        sub = df[df['method'] == m].sort_values('k')
        fig_l.add_trace(_err_bar(sub['k'], sub[metric], sub[std] if std in sub else [0]*len(sub),
                                  m, MC.get(m, '#888')))
    fig_l.update_xaxes(title_text='k', type='log')
    fig_l.update_yaxes(title_text=lbl)
    fig_l.update_layout(title=f'{lbl} vs k', **_fc())

    hm_m   = metric
    pivot  = df.pivot_table(index='method', columns='k', values=hm_m, aggfunc='mean')
    is_fix = 'fixedness' in hm_m
    fig_h  = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale='RdBu_r' if is_fix else 'RdBu',
        text=pivot.round(3).values, texttemplate='%{text}', showscale=True,
    ))
    fig_h.update_layout(title=f'Heatmap {lbl}', xaxis_title='k', **_fc())
    return fig_l, fig_h


@app.callback(Output('gap-lines', 'figure'), Output('gap-bar', 'figure'),
              Input('gap-method', 'value'),  Input('gap-k-filter', 'value'),
              Input('gap-k', 'value'))
def cb_gap(methods, k_filter, k_val):
    if CP is None:
        return _empty(), _empty()
    methods  = methods or METHODS
    k_filter = k_filter or sorted(CP['k'].unique())
    df = CP[CP['method'].isin(methods) & CP['k'].isin(k_filter)]

    fig_l = make_subplots(specs=[[{'secondary_y': True}]])
    for m in methods:
        sub = df[df['method'] == m].sort_values('k')
        c   = MC.get(m, '#888')
        fig_l.add_trace(go.Scatter(x=sub['k'], y=sub['mean_conflicts'], mode='lines+markers',
                                    name=f'{m} — conflitti', line=dict(color=c, width=2)),
                         secondary_y=False)
        fig_l.add_trace(go.Scatter(x=sub['k'], y=sub['mean_decisions'], mode='lines',
                                    name=f'{m} — decisioni', line=dict(color=c, width=1, dash='dot'),
                                    showlegend=True), secondary_y=True)
    fig_l.update_xaxes(title_text='k', type='log')
    fig_l.update_yaxes(title_text='Conflitti Z3 (linea piena)', secondary_y=False)
    fig_l.update_yaxes(title_text='Decisioni Z3 (tratteggiata)', secondary_y=True)
    fig_l.update_layout(title='Gap Complessità: Conflitti e Decisioni vs k', **_fc())

    if k_val is not None:
        sub = CP[CP['k'] == k_val].sort_values('mean_conflicts', ascending=False)
        fig_b = go.Figure(go.Bar(
            x=sub['method'], y=sub['mean_conflicts'],
            marker_color=[MC.get(m, '#888') for m in sub['method']],
            error_y=dict(type='data', array=sub['std_conflicts'].tolist(), visible=True),
        ))
        fig_b.update_layout(title=f'Conflitti per metodo (k={k_val})',
                             yaxis_title='Conflitti medi', **_fc())
    else:
        fig_b = _empty()
    return fig_l, fig_b


@app.callback(Output('sens-chart', 'figure'),
              Input('sens-param', 'value'), Input('sens-m1', 'value'), Input('sens-m2', 'value'))
def cb_sens(param, m1, m2):
    lbl1 = _metric_label(m1)
    s1   = m1.replace('mean_', 'std_')

    if param == 'k':
        if CP is None: return _empty()
        fig = go.Figure()
        for method in sorted(CP['method'].unique()):
            sub = CP[CP['method'] == method].sort_values('k')
            if sub[m1].isna().all(): continue
            fig.add_trace(_err_bar(sub['k'], sub[m1],
                                   sub[s1] if s1 in sub else [0]*len(sub),
                                   method, MC.get(method, '#888')))
        fig.update_xaxes(title_text='k', type='log')
        fig.update_yaxes(title_text=lbl1)
        fig.update_layout(title=f'{lbl1} vs k — tutti i metodi', **_fc())
        return fig

    if MS is None: return _empty()
    sub = MS[MS['sweep_param'] == param].copy()
    if sub.empty: return _empty(f'Nessun dato: sweep_param={param}')

    x_col = 'param_value' if 'param_value' in sub.columns else param
    if x_col not in sub.columns:
        return _empty(f'Re-esegui sweep_multiseed.py — colonna {x_col} mancante.')
    sub = sub.sort_values(x_col)
    x   = sub[x_col].astype(float)

    lbl2 = _metric_label(m2)
    s2   = m2.replace('mean_', 'std_')
    log_x = param in ('oracle_noise', 'n_patients')

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(_err_bar(x, sub[m1], sub[s1] if s1 in sub else [0]*len(sub), lbl1, ACCENT),
                  secondary_y=False)
    fig.add_trace(_err_bar(x, sub[m2], sub[s2] if s2 in sub else [0]*len(sub), lbl2, '#f38ba8'),
                  secondary_y=True)
    fig.update_xaxes(title_text=param, type='log' if log_x else 'linear')
    fig.update_yaxes(title_text=lbl1, secondary_y=False)
    fig.update_yaxes(title_text=lbl2, secondary_y=True)
    fig.update_layout(title=f'Sensitività: {param}', **_fc())
    return fig


@app.callback(Output('dp-chart', 'figure'),
              Input('dp-param', 'value'), Input('dp-metric', 'value'))
def cb_dp(param, metric):
    if MS is None or not param: return _empty()
    sub = MS[MS['sweep_param'] == param].copy()
    if sub.empty: return _empty(f'Nessun dato per parametro: {param}')
    if 'param_value' not in sub.columns:
        return _empty('Re-esegui sweep_multiseed.py — CSV vecchio senza param_value.')
    sub = sub.sort_values('param_value')
    lbl = _metric_label(metric)
    std = metric.replace('mean_', 'std_')

    fig = go.Figure()
    for m in sub['method'].unique():
        ms_m = sub[sub['method'] == m]
        x    = ms_m['param_value'].astype(float)
        fig.add_trace(_err_bar(x, ms_m[metric], ms_m[std] if std in ms_m else [0]*len(ms_m),
                                m, MC.get(m, '#888')))

    defaults = {'epsilon': 1.0, 'sigma': 3.0, 'l': 2.0, 't': 0.2}
    if param in defaults:
        fig.add_vline(x=defaults[param], line_dash='dash', line_color=ACCENT,
                      annotation_text=f'default={defaults[param]}',
                      annotation_font=dict(color=ACCENT2, size=11, family=FONT_BODY))

    fig.update_xaxes(title_text=param, type='log' if param == 'epsilon' else 'linear')
    fig.update_yaxes(title_text=lbl)
    fig.update_layout(title=f'{lbl} vs {param}', **_fc())
    return fig


@app.callback(Output('qi-chart', 'figure'),
              Input('qi-metric', 'value'), Input('qi-view', 'value'))
def cb_qi(metric, view):
    if QI is None: return _empty()
    lbl    = _metric_label(metric)
    std    = metric.replace('mean_', 'std_')
    pal    = ['#89b4fa', '#a6e3a1', '#fab387', '#cba6f7', '#f38ba8']
    names  = list(QI['qi_name'].unique())

    if view == 'lines':
        fig = go.Figure()
        for i, name in enumerate(names):
            sub = QI[QI['qi_name'] == name].sort_values('k')
            fig.add_trace(_err_bar(sub['k'], sub[metric],
                                    sub[std] if std in sub else [0]*len(sub),
                                    name, pal[i % len(pal)]))
        fig.update_xaxes(title_text='k', type='log')
        fig.update_yaxes(title_text=lbl)
        fig.update_layout(title=f'{lbl} vs k — per QI set', **_fc())
    else:
        pivot = QI.pivot_table(index='qi_name', columns='k', values=metric, aggfunc='mean')
        cs    = 'RdBu' if 'spong' in metric or 'cand' in metric else 'RdBu_r'
        fig   = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
                                      y=pivot.index.tolist(), colorscale=cs,
                                      text=pivot.round(3).values, texttemplate='%{text}'))
        fig.update_layout(title=f'Heatmap {lbl} (QI set × k)', **_fc())
    return fig


@app.callback(Output('bksa-chart', 'figure'),
              Input('bksa-metric', 'value'), Input('bksa-view', 'value'))
def cb_bksa(metric, view):
    if BKSA is None: return _empty()
    lbl  = _metric_label(metric)
    std  = metric.replace('mean_', 'std_')
    pal  = ['#89b4fa', '#a6e3a1', '#fab387', '#f38ba8']

    if view == 'lines':
        fig = go.Figure()
        for i, sa in enumerate(BKSA['bk_sa'].unique()):
            sub = BKSA[BKSA['bk_sa'] == sa].sort_values('k')
            fig.add_trace(_err_bar(sub['k'], sub[metric],
                                    sub[std] if std in sub else [0]*len(sub),
                                    sa, pal[i % len(pal)]))
        fig.update_xaxes(title_text='k', type='log')
        fig.update_yaxes(title_text=lbl)
        fig.update_layout(title=f'{lbl} vs k — per SA type', **_fc())
    else:
        pivot = BKSA.pivot_table(index='bk_sa', columns='k', values=metric, aggfunc='mean')
        cs    = 'RdBu' if 'spong' in metric or 'cand' in metric else 'RdBu_r'
        fig   = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
                                      y=pivot.index.tolist(), colorscale=cs,
                                      text=pivot.round(3).values, texttemplate='%{text}'))
        fig.update_layout(title=f'Heatmap {lbl} (SA type × k)', **_fc())
    return fig


@app.callback(Output('ora-chart', 'figure'),
              Input('ora-view', 'value'), Input('ora-metric', 'value'),
              Input('ora-methods', 'value'))
def cb_oracle(view, metric, methods):
    if ORA is None:
        return _empty('Esegui sweep_oracle.py prima di visualizzare questo tab.')
    methods = methods or METHODS
    lbl = _metric_label(metric)
    std = metric.replace('mean_', 'std_')

    if view == 'noise':
        sub = ORA[ORA['sweep_type'] == 'noise_vs_method']
        if sub.empty: return _empty('Nessun dato noise_vs_method nel CSV.')
        fig = go.Figure()
        for m in methods:
            ms = sub[sub['method'] == m].sort_values('oracle_noise')
            if ms.empty or ms[metric].isna().all(): continue
            fig.add_trace(_err_bar(ms['oracle_noise'], ms[metric],
                                   ms[std] if std in ms else [0]*len(ms),
                                   m, MC.get(m, '#888')))
        fig.update_xaxes(title_text='oracle_noise (righe civetta)', type='log')
        fig.update_yaxes(title_text=lbl)
        fig.update_layout(title=f'{lbl} vs rumore oracle — per metodo', **_fc())
        return fig

    if view == 'bkfrac':
        sub = ORA[ORA['sweep_type'] == 'bkfrac_vs_method']
        if sub.empty: return _empty('Nessun dato bkfrac_vs_method nel CSV.')
        fig = go.Figure()
        for m in methods:
            ms = sub[sub['method'] == m].sort_values('bk_frac')
            if ms.empty or ms[metric].isna().all(): continue
            fig.add_trace(_err_bar(ms['bk_frac'], ms[metric],
                                   ms[std] if std in ms else [0]*len(ms),
                                   m, MC.get(m, '#888')))
        fig.update_xaxes(title_text='bk_frac (conoscenza SA attaccante)')
        fig.update_yaxes(title_text=lbl)
        fig.update_layout(title=f'{lbl} vs BK fraction — per metodo', **_fc())
        return fig

    # grid heatmap
    sub = ORA[ORA['sweep_type'] == 'grid']
    if sub.empty: return _empty('Nessun dato grid nel CSV.')
    grid_method = sub['method'].iloc[0]
    pivot = sub.pivot_table(index='bk_frac', columns='oracle_noise',
                            values=metric, aggfunc='mean')
    is_fix = 'fixedness' in metric
    cs = 'RdBu_r' if is_fix else 'RdBu'
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns.tolist()],
        y=[str(r) for r in pivot.index.tolist()],
        colorscale=cs,
        text=pivot.round(3).values, texttemplate='%{text}', showscale=True,
    ))
    fig.update_layout(
        title=f'Heatmap {lbl} — {grid_method} (bk_frac × oracle_noise)',
        xaxis_title='oracle_noise (righe civetta)',
        yaxis_title='bk_frac',
        **_fc(),
    )
    return fig


@app.callback(Output('kv-chart', 'figure'),
              Input('kv-metric', 'value'), Input('kv-methods', 'value'))
def cb_k_variance(metric, methods):
    if CP is None: return _empty()
    methods = methods or METHODS
    df  = CP[CP['method'].isin(methods)]
    lbl = _metric_label(metric)
    std = metric.replace('mean_', 'std_')
    fig = go.Figure()
    for m in methods:
        sub = df[df['method'] == m].sort_values('k')
        if sub.empty: continue
        fig.add_trace(_err_bar(sub['k'], sub[metric],
                               sub[std] if std in sub else [0] * len(sub),
                               m, MC.get(m, '#888')))
    fig.update_xaxes(title_text='k', type='log')
    fig.update_yaxes(title_text=lbl)
    fig.update_layout(title=f'{lbl} vs k — per metodo', **_fc(height=500))
    return fig


@app.callback(Output('tim-chart', 'figure'),
              Input('tim-param', 'value'), Input('tim-metric', 'value'))
def cb_runtime(param, metric):
    if TIM is None or not param: return _empty()
    sub = TIM[TIM['sweep_param'] == param].sort_values('param_value')
    if sub.empty: return _empty()
    log_x = param in ('oracle_noise', 'n_patients')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub['param_value'], y=sub[metric], mode='lines+markers',
                              name=metric, line=dict(color=ACCENT, width=2), marker_size=7))
    if metric == 'mean_solve_ms':
        for col, name, dash in [('mean_p95_ms', 'p95', 'dash'), ('mean_max_ms', 'max', 'dot')]:
            if col in sub.columns:
                fig.add_trace(go.Scatter(x=sub['param_value'], y=sub[col], mode='lines',
                                          name=name, line=dict(color='#f38ba8', dash=dash)))

    fig.update_xaxes(title_text=param, type='log' if log_x else 'linear')
    fig.update_yaxes(title_text='ms')
    fig.update_layout(title=f'{_metric_label(metric)} vs {param}', **_fc())
    return fig


_RHO_QI_COLORS = {
    'zip_only':         '#2C6FAC',
    'sex_zip':          '#2E7D52',
    'sex_zip_job':      '#B8753A',
    'sex_city_zip_job': '#7B3FA2',
    'all_qi':           '#B83232',
}


@app.callback(Output('rho-k', 'options'), Output('rho-k', 'value'), Input('rho-method', 'value'))
def cb_rho_k_opts(method):
    if RHO is not None and 'method' in RHO.columns and 'k' in RHO.columns:
        sub = RHO[RHO['method'] == method] if method in RHO['method'].unique() else RHO
        vals = sorted(sub['k'].dropna().unique().astype(int))
        opts = [{'label': f'k={v}', 'value': v} for v in vals]
        return opts, vals[0] if vals else None
    if CP is None or method not in CP['method'].unique():
        return [], None
    vals = sorted(CP[CP['method'] == method]['k'].unique())
    opts = [{'label': f'k={v}', 'value': v} for v in vals]
    return opts, vals[0] if vals else None


@app.callback(Output('rho-chart', 'figure'), Output('rho-qi-chart', 'figure'),
              Input('rho-method', 'value'), Input('rho-k', 'value'))
def cb_rho(method, k):
    # ── Left: ρ vs N — Analitico + Effective ρ sullo stesso asse ─────────────
    if RHO is None:
        fig_l = _empty('Esegui: python experiments/sweep_rho.py')
    else:
        df = RHO.sort_values('n_patients')
        fig_l = make_subplots(specs=[[{'secondary_y': True}]])

        # ρ Analitico (schema-level, indipendente dall'anonimizzazione)
        fig_l.add_trace(go.Scatter(
            x=df['n_patients'], y=df['rho'], mode='lines+markers',
            name='ρ analitico (schema)',
            line=dict(color=MUTED, width=2, dash='dash'),
            marker=dict(size=6, symbol='circle-open'),
        ), secondary_y=False)

        # Effective ρ — una linea per metodo, filtrate per k selezionato
        if 'effective_rho' in df.columns and 'method' in df.columns:
            df_k = df[df['k'] == k] if ('k' in df.columns and k is not None) else df
            for m in sorted(df_k['method'].unique()):
                sub = df_k[df_k['method'] == m].sort_values('n_patients')
                if sub.empty:
                    continue
                is_selected = (m == method)
                fig_l.add_trace(go.Scatter(
                    x=sub['n_patients'], y=sub['effective_rho'], mode='lines+markers',
                    name=f'Effective ρ — {m}',
                    line=dict(color=MC.get(m, ACCENT), width=3 if is_selected else 1.5,
                              dash='solid' if is_selected else 'dot'),
                    marker=dict(size=9 if is_selected else 5,
                                line=dict(width=1, color='white')),
                    visible=True if is_selected else 'legendonly',
                ), secondary_y=False)

            # Retention ratio del metodo selezionato — asse secondario
            sub_sel = df_k[df_k['method'] == method].sort_values('n_patients') if method else pd.DataFrame()
            if 'retention_ratio' in df.columns and not sub_sel.empty:
                fig_l.add_trace(go.Scatter(
                    x=sub_sel['n_patients'], y=sub_sel['retention_ratio'], mode='lines',
                    name=f'Retention ratio — {method}',
                    line=dict(color='#2E7D52', width=1.5, dash='dot'),
                    visible='legendonly',
                ), secondary_y=True)

        # ρ e effective ρ dall'ultimo audit reale (main.py) — punto singolo
        if AUDIT_RHO is not None:
            n_run = AUDIT_RHO['n_records']
            fig_l.add_trace(go.Scatter(
                x=[n_run], y=[AUDIT_RHO['rho']], mode='markers',
                name=f'ρ audit reale (N={n_run})',
                marker=dict(color='#B83232', size=14, symbol='star',
                            line=dict(width=1.5, color='white')),
            ), secondary_y=False)
            if AUDIT_RHO.get('effective_rho') is not None:
                fig_l.add_trace(go.Scatter(
                    x=[n_run], y=[AUDIT_RHO['effective_rho']], mode='markers',
                    name=f'Effective ρ audit reale (N={n_run})',
                    marker=dict(color=ACCENT, size=14, symbol='star',
                                line=dict(width=1.5, color='white')),
                ), secondary_y=False)

        fig_l.add_hline(y=4.27, line_dash='dash', line_color='#B83232', line_width=1.5,
                        annotation_text='ρ* ≈ 4.27',
                        annotation_position='top right',
                        annotation_font=dict(color='#B83232', size=11))

        for col, name, color in [
            ('total_clauses_3sat', 'Clausole 3-SAT',              '#2C6FAC'),
            ('total_vars',         'Var. totali',                  '#2E7D52'),
            ('base_vars',          'Var. base (N×M×D)',            '#7B3FA2'),
            ('aux_vars',           'Var. ausiliarie (Tseitin FD)', '#C95F1A'),
        ]:
            if col not in df.columns:
                continue
            fig_l.add_trace(go.Scatter(
                x=df['n_patients'], y=df[col], mode='lines', name=name,
                line=dict(color=color, width=1.5, dash='dot'),
                visible='legendonly',
            ), secondary_y=True)

        m_label = method or ''
        fig_l.update_xaxes(title_text='N pazienti', type='log')
        fig_l.update_yaxes(title_text='ρ / Effective ρ', secondary_y=False, rangemode='tozero')
        fig_l.update_yaxes(title_text='Conteggio / Retention', secondary_y=True, type='log')
        fig_l.update_layout(title=f'ρ analitico vs Effective ρ ({m_label})', **_fc())

    # ── Right: ρ vs N per QI set ─────────────────────────────────────────────
    if RHO_QI is None:
        fig_r = _empty('Esegui: python experiments/sweep_rho_qi.py')
    else:
        df_qi = RHO_QI.sort_values('n_patients')
        fig_r = go.Figure()

        fig_r.add_hline(y=4.27, line_dash='dash', line_color='#B83232', line_width=1.5,
                        annotation_text='ρ* ≈ 4.27',
                        annotation_position='top right',
                        annotation_font=dict(color='#B83232', size=11))

        for qi_name in df_qi['qi_name'].unique():
            sub = df_qi[df_qi['qi_name'] == qi_name].sort_values('n_patients')
            m_val = sub['n_attributes'].iloc[0]
            fig_r.add_trace(go.Scatter(
                x=sub['n_patients'], y=sub['rho'], mode='lines+markers',
                name=f"{qi_name} (M={m_val})",
                line=dict(color=_RHO_QI_COLORS.get(qi_name, '#888'), width=2.5),
                marker=dict(size=7, line=dict(width=1, color='white')),
            ))

        fig_r.update_xaxes(title_text='N pazienti', type='log')
        fig_r.update_yaxes(title_text='ρ', rangemode='tozero')
        fig_r.update_layout(title='ρ vs N — confronto QI set', **_fc())

    return fig_l, fig_r


if __name__ == '__main__':
    print(f"\nDashboard avviata → http://localhost:8050\n")
    app.run(debug=False, host='0.0.0.0', port=8050)
