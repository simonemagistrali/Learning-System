# 🧠 Learning System - Laboratori e Progetti

Benvenuto nella repository del corso **Learning System**. Questo repository raccoglie lo sviluppo, la sperimentazione e l'analisi di cinque progetti incentrati su algoritmi di machine learning, ottimizzazione euristica, analisi non supervisionata e reinforcement learning (RL), sviluppati in **Python** e **MATLAB**.

---

## 📑 Indice dei Progetti

1. [Lab 1: Stima del Decadimento di Turbine a Gas (Multi-Layer Perceptron)](#-lab-1-stima-del-decadimento-di-turbine-a-gas-mlp-from-scratch)
2. [Lab 2: Ottimizzazione del Travelling Salesperson Problem (Algoritmo Genetico)](#-lab-2-ottimizzazione-del-commesso-viaggiatore-algoritmo-genetico)
3. [Lab 3: Riduzione della Dimensionalità e Clustering (PCA & K-Means)](#-lab-3-riduzione-dimensionale-e-clustering-pca--k-means)
4. [Lab 4: Ricerca del Cammino Minimo (Value Iteration MDP)](#-lab-4-ricerca-del-cammino-minimo-value-iteration-mdp)
5. [Lab 5: Controllo del Pendolo Inverso su Carrello (Q-Learning)](#-lab-5-controllo-del-pendolo-inverso-su-carrello-q-learning)
6. [Tabella Riassuntiva](#-tabella-riassuntiva)
7. [Come Eseguire i Progetti](#-come-eseguire-i-progetti)

---

## 🔬 Lab 1: Stima del Decadimento di Turbine a Gas (MLP from scratch)

### 📌 Nome
**Stima del Decadimento Prestazionale di Impianti a Turbina a Gas tramite Rete Neurale Artificiale Multistrato**

### 📝 Descrizione
Il progetto affronta il problema della manutenzione predittiva per la propulsione navale a turbina a gas. L'obiettivo è stimare con elevata accuratezza lo stato di degrado di due componenti critici: il **compressore** e la **turbina**, partendo da parametri operativi misurati dai sensori dell'impianto durante il funzionamento. La rete neurale è implementata interamente da zero (*from scratch*) senza librerie ad alto livello come PyTorch o TensorFlow.

### 📊 Dati Usati
- **File:** `lab 1/maintenance.txt` (e copia in `lab 1/data/maintenance.txt`)
- **Dimensioni:** 11.934 campioni numerici generati da un simulatore navale realistico.
- **Input (16 feature):** Velocità della nave, regime di rotazione della turbina, coppia erogata, temperature e pressioni nei vari stadi, portata del combustibile.
- **Output (2 target continui):**
  1. *Compressor Decay State Coefficient* ($k_{Mc}$)
  2. *Turbine Decay State Coefficient* ($k_{Mt}$)
- **Suddivisione:** 50% Training set, 25% Validation set, 25% Test set (random state fissato a 45 per riproducibilità).

### ⚙️ Algoritmo Usato
- **Rete Neurale Multistrato (Multi-Layer Perceptron - MLP)**:
  - Architettura a 4 strati: `[16 input -> 30 neuroni -> 30 neuroni -> 2 output]`.
  - Discesa del gradiente con retropropagazione dell'errore (**Backpropagation**).
  - Funzione di costo: Mean Squared Error (**MSE**).

### 🛠️ Come l'Algoritmo è Stato Modificato / Adattato
- **Implementazione custom completa:** Algoritmi di forward pass e backward pass derivati ed espressi puramente in algebra matriciale NumPy.
- **Normalizzazione Z-Score bidirezionale:** Standardizzazione sia delle 16 feature di input che dei 2 target di output basata esclusivamente su media e deviazione standard del Training set. Questo passaggio si è rivelato cruciale per stabilizzare i gradienti e permettere la convergenza verso valori dell'ordine di $10^{-8}$.
- **Attivazione ibrida:** Funzione Sigmoide sui layer nascosti e attivazione **lineare pura** (*identity*) sul layer di output per consentire regressione continua senza saturazione.
- **Inizializzazione controllata:** Inizializzazione gaussiana scalata dei pesi con deviazione standard ridotta (`np.random.randn * 0.01`) e bias nulli per evitare la saturazione iniziale dei neuroni sigmoidei.
- **Monitoraggio epoch-by-epoch:** Tracciamento della curva di apprendimento su scala logaritmica per monitorare costantemente il validation error ed escludere l'insorgenza di overfitting.

### 🎯 Precisione / Goal Raggiunta
- **Obiettivo di specifica:**
  - MSE Test Set (Compressore) $\le 5.0 \times 10^{-8}$
  - MSE Test Set (Turbina) $\le 1.5 \times 10^{-8}$
- **Risultato finale ottenuto:**
  - ✅ **MSE Compressore:** **$1.3482 \times 10^{-8}$** (ben al di sotto della soglia $5.0 \times 10^{-8}$)
  - ✅ **MSE Turbina:** **$1.0734 \times 10^{-8}$** (ben al di sotto della soglia $1.5 \times 10^{-8}$)
  - Entrambi i requisiti sono stati pienamente superati. Grafico delle curve di convergenza disponibile in `lab 1/grafico_soluzione_positiva.png`.

---

## 🧬 Lab 2: Ottimizzazione del Commesso Viaggiatore (Algoritmo Genetico)

### 📌 Nome
**Risoluzione del Travelling Salesperson Problem (TSP) su Berlin52 tramite Algoritmo Genetico**

### 📝 Descrizione
Ottimizzazione euristica del celebre problema combinatorio NP-hard del commesso viaggiatore (TSP): trovare il circuito hamiltoniano di lunghezza minima che visita tutte le 52 città una sola volta tornando alla città iniziale.

### 📊 Dati Usati
- **File:** `lab 2/data/berlin52.tsp`
- **Descrizione:** Benchmark internazionale TSPLIB contenente 52 coordinate bidimensionali $(x, y)$ di nodi distribuiti nella città di Berlino.
- **Ottimo teorico noto globale:** $\approx 7542$.

### ⚙️ Algoritmo Usato
- **Algoritmo Genetico (Genetic Algorithm - GA)** orientato a problemi di permutazione.

### 🛠️ Come l'Algoritmo è Stato Modificato / Adattato
- **Codifica genetica:** Ogni individuo rappresenta una permutazione valida degli indici delle 51 città da visitare (nodo 0 fissato stabilmente come punto di partenza e arrivo).
- **Crossover Ordinato (Ordered Crossover - OX):** Probabilità del 90%. Preserva le sottosequenze contigue del primo genitore e ordina le restanti città secondo l'apparizione nel secondo genitore, garantendo la validità della permutazione senza duplicati né omissioni.
- **Strategia di Mutazione Dinamica Doppia:** Probabilità di mutazione al 50% con scelta casuale tra due operatori complementari:
  1. *Swap Mutation:* Scambio casuale di due geni (ottimizzazione locale).
  2. *Inversion Mutation:* Inversione dell'ordine di un segmento intermedio (rottura efficace di intersezioni geometriche tra percorsi).
- **Selezione a Torneo (Tournament Selection):** Dimensione del torneo fissata a $k=3$.
- **Elitismo:** Preservazione automatica dei migliori 2 individui (`num_elite = 2`) alla generazione successiva per evitare il degrado della fitness.
- **Controllo della diversità genetica:** Metrica integrata di diversità di popolazione calcolata generation-by-generation per prevenire la convergenza prematura su minimi locali.

### 🎯 Precisione / Goal Raggiunta
- **Obiettivo richiesto:** Lunghezza totale del percorso inferiore alla soglia di **8000 unità**.
- **Risultato raggiunto:**
  - ✅ **Soglia infranta (< 8000):** L'algoritmo converge sistematicamente a una soluzione valida con lunghezza $< 8000$, approssimando strettamente l'ottimo globale noto di Berlin52 (~7542) con un errore percentuale trascurabile.
  - Visualizzazione del percorso ottimale generata con plot grafico interattivo delle connessioni.

---

## 🍷 Lab 3: Riduzione Dimensionale e Clustering (PCA & K-Means)

### 📌 Nome
**Analisi Chimica del Vino tramite Principal Component Analysis (PCA) e Clustering K-Means**

### 📝 Descrizione
Studio esplorativo non supervisionato applicato al dataset enologico UCI Wine. Il progetto si articola in due fasi:
1. Riduzione della dimensionalità tramite PCA per individuare le direzioni di massima varianza e proiettare lo spazio multidimensionale in 2D.
2. Segmentazione non supervisionata con K-Means testato su diversi sottospazi di componenti principali per verificare la separabilità dei cluster.

### 📊 Dati Usati
- **File:** `lab 3/WINE.txt`
- **Dimensioni:** 178 campioni di vino provenienti da tre cultivar differenti della stessa regione in Italia.
- **Attributi (13 feature chimiche):** Alcol, acido malico, ceneri, alcalinità delle ceneri, magnesio, fenoli totali, flavonoidi, fenoli non flavonoidi, proantocianine, intensità colore, tonalità, OD280/OD315 dei vini diluiti, prolina.
- **Target reale (Ground Truth per validazione):** 3 cultivar (Classi 1, 2, 3).

### ⚙️ Algoritmi Usati
1. **Principal Component Analysis (PCA)** implementata da zero tramite decomposizione spettrale.
2. **K-Means Clustering ($K=3$)** implementato da zero con metrica di distanza euclidea.

### 🛠️ Come gli Algoritmi sono Stati Modificati / Adattati
- **Standardizzazione Z-Score pre-PCA:** Trasformazione a media nulla e varianza unitaria per evitare che attributi con magnitudo elevate (come prolina o magnesio) dominassero artificialmente la matrice di covarianza.
- **Decomposizione Spettrale custom:** Calcolo della matrice di covarianza $Q = \text{Cov}(X_{\text{scaled}})$, estrazione di autovalori e autovettori con `np.linalg.eigh`, e ordinamento rigorosamente decrescente in base alla varianza spiegata.
- **K-Means a convergenza robusta:**
  - Assegnamento euclideo iterativo con ricalcolo vettorizzato dei centroidi.
  - Criterio di arresto duplice: limite di 100 iterazioni o convergenza assoluta verificata con `np.allclose(old_centroids, centroids)`.
- **Analisi Comparativa a Dimensionalità Scalare:** Valutazione comparativa del clustering $K=3$ su tre diversi scenari di feature ridotte:
  - **Esperimento 1:** Prime 2 componenti principali.
  - **Esperimento 2:** Prime 5 componenti principali.
  - **Esperimento 3:** Prime 8 componenti principali.

### 🎯 Precisione / Goal Raggiunta
- **Visualizzazione 2D:** Generazione del plot `lab 3/pca_top2_visualization.png` in cui le cultivar risultano visibilmente raggruppate e separate nello spazio PC1-PC2.
- **Clustering:** Il modello ha dimostrato che le prime 2 componenti principali catturano già gran parte della struttura geometrica delle cultivar originali, con assegnazioni coerenti e centroidi stabili anche aumentando la dimensionalità a 5 e 8 componenti.

---

## 🗺️ Lab 4: Ricerca del Cammino Minimo (Value Iteration MDP)

### 📌 Nome
**Pianificazione del Percorso Ottimo tramite Value Iteration su Processo Decisionale di Markov (MDP)**

### 📝 Descrizione
Risoluzione del problema del cammino minimo su un grafo topologico di città collegate mediante tecniche di Reinforcement Learning e Programmazione Dinamica. L'agente deve calcolare la funzione di valore ottima $V^*(s)$ per ciascun nodo e ricostruire la traiettoria più breve da ogni città di partenza verso una città obiettivo designata.

### 📊 Dati Usati
- **File:** `lab 4/city 1.txt`
- **Struttura del Grafo:** 23 città/nodi (da A a Z) collegati da 35 archi bidirezionali pesati con distanze chilometriche intere.
- **Obiettivo (Goal State):** Città **F** (`GOAL_CITY = "F"`).

### ⚙️ Algoritmo Usato
- **Value Iteration** basato sull'Equazione di Ottimalità di Bellman per MDP deterministici.

### 🛠️ Come l'Algoritmo è Stato Modificato / Adattato
- **Formulazione MDP del Cammino Minimo:**
  - Ogni città è uno stato $s \in S$.
  - Il costo di percorrenza dell'arco tra $s$ e $s'$ (distanza $d$) è modellato come una ricompensa negativa (penalità): $R(s, a, s') = -d$.
  - Il goal "F" è definito come stato terminale assorbente con $V^*(F) = 0.0$.
  - Tutti gli altri stati sono inizializzati a $-\infty$ ($V_0(s) = -\infty$).
  - Fattore di sconto $\gamma = 1.0$ (orizzonte finito con terminazione garantita).
- **Aggiornamento di Bellman:**
  $$V_{k+1}(s) = \max_{s' \in \text{Vicini}(s)} \left( -d(s, s') + \gamma V_k(s') \right)$$
- **Ricostruzione della Policy Ottima:** Implementata la funzione `find_shortest_path` che estrae la policy greedy $\pi^*(s) = \arg\max_{s'} [-d(s, s') + V^*(s')]$ per restituire la sequenza esatta di tappe minime da qualsiasi nodo di partenza.

### 🎯 Precisione / Goal Raggiunta
- **Convergenza:** Raggiunta la convergenza assoluta e deterministica in poche iterazioni senza oscillazioni.
- **Accuratezza:** Calcolo esatto dei valori ottimi $V^*(s)$ e delle distanze reali minime per tutti i 23 nodi verso il goal F (es. $A \to B \to F$ distanza 5, $W \to A \to B \to F$ distanza 6, ecc.), determinando univocamente il percorso ottimale per ogni punto della mappa.

---

## 🕹️ Lab 5: Controllo del Pendolo Inverso su Carrello (Q-Learning)

### 📌 Nome
**Bilanciamento di un Pendolo Inverso (Cart-Pole Balancing) con Tabular Q-Learning in MATLAB**

### 📝 Descrizione
Controllo dinamico di un sistema non lineare instabile costituito da un pendolo inverso montato su un carrello mobile (Cart-Pole system). L'agente deve imparare da zero ad applicare forze orizzontali correttive per mantenere il pendolo in posizione verticale ed evitare che il carrello oltrepassi i limiti del binario.

### 📊 Dati Usati
- **Ambiente:** Simulatore fisico `lab 5/SimulatePendel.m` con integrazione numerica ad ogni intervallo di tempo $\Delta t = 0.02$ s.
- **Spazio di Stato Continuo 4D:**
  1. Posizione del carrello $x$ (vincolo limite: $|x| \le 2.4$ m)
  2. Velocità del carrello $\dot{x}$
  3. Angolo del pendolo $\theta$ (vincolo limite: $|\theta| \le 12^\circ$)
  4. Velocità angolare del pendolo $\dot{\theta}$
- **Spazio di Azione Discreto:** 2 azioni possibili:
  - Spinta verso sinistra: $-10\text{ N}$
  - Spinta verso destra: $+10\text{ N}$

### ⚙️ Algoritmo Usato
- **Reinforcement Learning - Q-Learning Tabulare (Model-Free, Off-Policy)**.

### 🛠️ Come l'Algoritmo è Stato Modificato / Adattato
- **Discretizzazione Spaziale non uniforme tramite Binning:** Lo spazio continuo a 4 dimensioni è stato mappato in una tabella Q discreta di **5.000 stati complessivi** tramite partizioni mirate:
  - Posizione $x$: 10 bin in $[-2.4, 2.4]$
  - Velocità $\dot{x}$: 5 bin in $[-2, 2]$
  - Angolo $\theta$: 20 bin in $[-12^\circ, 12^\circ]$ (maggiore risoluzione sull'angolo critico)
  - Velocità angolare $\dot{\theta}$: 5 bin in $[-2, 2]$
  - Compressione in indice monodimensionale con `sub2ind([10, 5, 20, 5], s1, s2, s3, s4)`.
- **Politica di Esplorazione Softmax (Boltzmann-like) con Annealing Adattivo:**
  - Probabilità di selezione dell'azione: $P(a) \propto \kappa^{Q(s, a) - \max_a Q(s, a)}$.
  - La sottrazione del massimo garantisce stabilità numerica evitando overflow esponenziali.
  - Il parametro $\kappa$ cresce progressivamente ad ogni episodio ($\kappa \leftarrow \min(20, \kappa \times 1.0005)$), bilanciando l'esplorazione iniziale con l'exploitation raffinata a lungo termine.
- **Reward Shaping:**
  - $+1$ punto per ogni step sopravvissuto entro i limiti fisici.
  - Forte penalità di $-100$ in caso di fallimento ($|x| > 2.4$ m oppure $|\theta| > 12^\circ$), con arresto immediato dell'episodio.
  - Fattore di sconto $\gamma = 0.99$.

### 🎯 Precisione / Goal Raggiunta
- **Obiettivo:** Sopravvivere a **3.000 step consecutivi** di simulazione continua (corrispondenti a 60 secondi ininterrotti a $50\text{ Hz}$) senza caduta del pendolo o collisione con i bordi.
- **Risultato raggiunto:**
  - ✅ **Successo totale nel Test:** Come verificabile eseguendo `lab 5/testAgent.m`, la tabella pre-addestrata `lab 5/Q_Table_Addestrata.mat` stabilizza il pendolo completando con successo **tutti i 3.000 step di test**.
  - L'angolo $\theta$ oscilla stabilmente entro un ristretto intorno di $0^\circ$ e la posizione $x$ rimane perfettamente contenuta entro il binario di sicurezza.

---

## 📊 Tabella Riassuntiva

| Lab | Tema & Paradigma | Dati & Ambiente | Algoritmo | Adattamento Chiave | Goal / Metrica Raggiunta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lab 1** | Apprendimento Supervisionato (Regressione) | `maintenance.txt` (11.934 campioni, 16 input, 2 target) | MLP 4 strati (16-30-30-2) con Backprop from scratch | Z-score su input e target, attivazione lineare sull'output | MSE Compressore: **$1.35 \times 10^{-8}$**<br>MSE Turbina: **$1.07 \times 10^{-8}$** (entrambi sotto le soglie richieste) |
| **Lab 2** | Ottimizzazione Combinatoria Euristica | `berlin52.tsp` (52 città coordinate 2D) | Algoritmo Genetico (GA) | Crossover OX (90%), doppia mutazione Swap/Inversion (50%), elitismo e monitoraggio diversità | Lunghezza circuito **$< 8000$** (vicino all'ottimo globale noto di ~7542) |
| **Lab 3** | Apprendimento Non Supervisionato | `WINE.txt` (178 campioni, 13 feature chimiche, 3 cultivar) | PCA + K-Means ($K=3$) | Autovalori/autovettori di $Q_{\text{cov}}$, clustering a dimensionalità variabile (Top-2, Top-5, Top-8) | Chiara separazione visiva 2D e convergenza coerente dei 3 cluster naturali |
| **Lab 4** | Reinforcement Learning / Programmazione Dinamica | `city 1.txt` (23 città, 35 archi pesati) | Value Iteration MDP su grafo | Ricompensa negativa $r = -d$, $\gamma = 1.0$, estrazione policy greedy | Convergenza esatta a $V^*(s)$ e calcolo deterministico di tutti i cammini minimi verso F |
| **Lab 5** | Reinforcement Learning per Controllo Continuo | Simulazione fisica Cart-Pole ($\Delta t=0.02$s) | Q-Learning Tabulare (MATLAB) | Discretizzazione in 5000 stati, Softmax numericamente stabile con annealing di $\kappa$ | **3.000/3.000 step sopravvissuti** (60s di stabilità perfetta) |

---

## 🚀 Come Eseguire i Progetti

### Prerequisiti
- **Python 3.8+** con i moduli: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (usato per lo split del dataset nel Lab 1).
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```
- **MATLAB** (R2019b o superiore) per il Lab 5.

### Esecuzione dei Laboratori Python
```bash
# Lab 1 - Rete Neurale MLP
cd "lab 1"
python main.py

# Lab 2 - Algoritmo Genetico TSP
cd "../lab 2"
python main.py

# Lab 3 - PCA e K-Means
cd "../lab 3"
python main.py

# Lab 4 - Value Iteration MDP
cd "../lab 4"
python main.py
```

### Esecuzione del Laboratorio MATLAB (Lab 5)
1. Aprire MATLAB e posizionarsi nella cartella `lab 5/`.
2. Per verificare l'agente addestrato: eseguire lo script `testAgent.m`.
3. Per avviare un nuovo ciclo di training: eseguire lo script `main.m`.

---

## 👤 Autore
- **Simone Magistrali** - *Learning System Course*