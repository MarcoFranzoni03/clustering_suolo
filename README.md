# 🌱 Clustering del Suolo

Questo progetto ha l'obiettivo di analizzare e raggruppare i dati relativi a campioni di suolo per identificare pattern nascosti, similarità e caratteristiche geochimiche o fisiche comuni attraverso tecniche di machine learning non supervisionato.

---

## 🛠️ Installazione delle Dipendenze

Per eseguire questo progetto in locale, è necessario configurare l'ambiente Python. Segui questi passaggi:

1. Clona la repository:
```bash
git clone https://github.com/MarcoFranzoni03/clustering_suolo
```
2. Crea e attiva un ambiente virtuale per isolare le dipendenze:

# Su Windows:
python -m venv venv
.\venv\Scripts\activate

# Su macOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Installa le dipendenze:

pip install -r requirements.txt

---

## 📁 Struttura del Progetto

Il progetto è strutturato in modo modulare e sequenziale, seguendo passo dopo passo l'intera pipeline di data science, dal recupero dei dati grezzi fino alla simulazione agronomica:

```text
├── data/                            # Dati del suolo (grezzi e processati)
├── meteo_data/                      # Dati meteorologici formattati per AquaCrop
└── src/                             # Codice sorgente Python (.py e .ipynb)
    ├── 0_download/                  # Fase 0: Script per il download dei dati grezzi
    ├── 1_load/                      # Fase 1: Caricamento e parsing dei dataset
    ├── 2_data_analysis/             # Fase 2: Analisi esplorativa dei dati (EDA)
    ├── 3_data_preprocessing_pipeline/ # Fase 3: Pipeline di pulizia, rimozione outlier e PCA
    ├── 4_clustering/                # Fase 4: Algoritmi di clustering (K-Means, Bisecting K-Means, HDBSCAN, ecc.)
    ├── 5_aquacrop/                  # Fase 5: Elaborazione dati e integrazione con AquaCrop
    └── utils_plotting/              # Script e moduli per la generazione di grafici e mappe
```
## 🧹 Data Preprocessing
Il successo del clustering dipende fortemente dalla qualità della pipeline di pre-elaborazione. I dati grezzi del dataset sono stati trattati seguendo questi step sequenziali:

### 1. Gestione dei Valori Duplicati
* **Strategia utilizzata:** Identificazione dei duplicati tramite subset di colonne e rimozione se privi di coordinate geografiche distinte.
* **Motivazione:** La presenza di record identici può derivare da fenomeni diversi che richiedono un trattamento differenziato:
  * **Duplicati reali:** anomalie generate da ripetizioni nel campionamento sul campo, bug software durante le fasi di estrazione (ETL) o problemi di ridondanza nel database. 
  * **Duplicati naturali:** campioni di suolo fisicamente distinti che, per coincidenza o omogeneità della zona, presentano gli stessi identici valori analitici.

### 2. Gestione dei Valori Negativi e Corrotti
* **Strategia utilizzata:** Si verifica che la feature Water Volumetric sia non negativa ma anche l'azzeramento simultaneo delle frazioni di sabbia, limo e argilla; inoltre, si rimuovono valori come -32768
* **Motivazione:** La feature può assumere come valore minimo 0 che simboleggia l'aridità mentre l'azzeramento simultaneo delle tre quantità è fisicamente impossibile e valori come -32768 indicano un NoData

### 3. Aggregazione verticale dei livelli
* **Motivazione:** Si applica una media ponderata sui sei livelli di profondità dove il peso è lo spessore dei livelli.
* **Risultato:** Si ottengono 13 features a cui si aggiunge l'OCS.

### 4. Gestione dei Valori Mancanti
* **Strategia utilizzata:** Imputazione con la Mediana con SimpleImputer
* **Motivazione:** È stata preferita questa strategia per non distorcere la distribuzione delle proprietà chimico-fisiche del suolo in presenza di distribuzioni asimmetriche.

### 5. Standardizzazione z-score
* **Motivazione:** Algoritmi basati sulla distanza come K-Means sono sensibili alla scala delle features.
* **Metodo:**  StandardScaler con mu = 0 e rho = 1.

### 6. Rimozione degli Outlier
* **Tecnica:** Isolation Forest per  isolare ed eliminare automaticamente le anomalie

### 7. PCA - Riduzione della dimensionalità
* ** Riduzione:** Si riducono le dimensioni a 7 mantenendo il numero minimo di componenti necessari per spiegare
il 90% della varianza statistica.
---

## 📊 Algoritmi di Clustering e Configurazione

L'analisi ha previsto un confronto diretto tra due approcci principali per identificare la segmentazione ottimale del suolo.

### 1. K-Means
Configurato per cercare strutture sferiche nei dati basandosi sulle distanze euclidee medie.

#### 🎛️ Ottimizzazione Iperparametrica con Optuna
La ricerca dei parametri è stata automatizzata tramite un processo Bayesiano guidato dagli indici:
  * *Elbow Method*
  * *Davies - Bouldin Index*
  * *Calinski Index*
  * *Silhouette Analysis*

* **Iperparametri:**
  * `n_clusters`: 4 
  * `init`: `'k-means++'` (ottimizzazione dell'inizializzazione dei centroidi)
  * `max_iter`: 1000 (per garantire la convergenza assoluta)
* **Risultati chiave:** Silhouette Score di `[0.2249]`.

### 2. Bisecting K-Means
Un approccio "top-down" che inizia con un unico grande cluster e lo divide iterativamente in due sotto-cluster usando K-Means, fino a raggiungere il target impostato. Funziona direttamente sulle componenti ortogonali della PCA.
* **Iperparametri:**
  * `n_clusters`: 4
  * `init`: `'k-means++'` (per gli split interni)
  * `n_init`: 10
  * `random_state`: 42
* **Focus:** Ottimo per catturare strutture gerarchiche e annidate nelle proprietà del suolo.

### 3. Gaussian Mixture Model - GMM 
Modello basato sull'assunto che i dati siano generati da una mistura di distribuzioni gaussiane a tolleranza ellittica. Consente il **soft-clustering**.
* **Iperparametri:**
  * `n_components`: 4
  * `covariance_type`: `'full'` (permette ai cluster di assumere qualsiasi forma ellittica nello spazio PCA)
  * `n_init`: 10
* **Output Unici:** Oltre alle labels rigide, estrae le probabilità di appartenenza fondamentali per mappare le **zone di transizione** sfumate tra diversi tipi di suolo.

### 5. DBSCAN 
Raggruppa i punti in base alla densità spaziale, isolando i campioni che si trovano in aree a bassa densità.
* **Iperparametri:**
  * `eps`: 1.0 (raggio del vicinato individuato tramite NearestNeighbors)
  * `min_samples`: 14 (allineato con la regola dei 14-NN applicata sul grafico a gomito)
  * `n_jobs`: -1 (parallelizzazione completa su tutti i core della CPU)
* **Output Unici:** Identifica automaticamente il rumore di fondo. I punti non classificati vengono contrassegnati con l'etichetta `-1`.

### 6. HDBSCAN 
Estensione di DBSCAN che converte l'algoritmo in un algoritmo gerarchico per trovare cluster di densità variabile, eliminando la necessità di impostare un raggio `eps` globale e rigido.
#### 🎛️ Ottimizzazione Iperparametrica con Optuna (TPE)
Per evitare un tuning manuale inefficiente, la ricerca dei parametri è stata automatizzata tramite un processo Bayesiano guidato dall'algoritmo **TPE (Tree-Structured Parzen Estimator)** di Optuna. 

L'obiettivo impostato nello studio è la massimizzazione della **Cluster Persistence**, una metrica che quantifica la resistenza di una struttura ai cambiamenti di densità. 
La persistenza misura la "durata biologica" di un cluster all'interno della gerarchia prima che si fonda nel rumore o in un altro gruppo.

Per ogni punto $p$ appartenente a un cluster $C$, l'indice di persistenza è calcolato formalmente come:

$$\text{persistence}(p) = \lambda_{death}(p) - \lambda_{birth}(C)$$

* **Iperparametri configurabili:**
  * `min_cluster_size`: [Inserisci valore, es. 15] (dimensione minima per considerare un gruppo un cluster)
  * `min_samples`: [Inserisci valore] (restringe la definizione di densità)
  * `cluster_selection_epsilon`: [Inserisci valore] (soglia per il clustering)
  * `cluster_selection_method`: [Es. 'eom' o 'leaf']
  * `n_jobs`: -1

## 🎨 Visualizzazione e Interpretazione dei Cluster
Per validare i risultati sia dal punto di vista statistico che agronomico, il progetto genera una suite di visualizzazioni avanzate. 
Di seguito sono descritti i grafici principali prodotti e il loro significato analitico:

### 1. Mappa Geografica dei Campioni
* **Descrizione:** Un plot di dispersione spaziale (Scatter Plot) basato sulle coordinate geografiche reali (Latitudine e Longitudine) dei campionamenti.
* **Obiettivo:** Visualizzare la distribuzione fisica dei punti di campionamento sul territorio della Capitanata utile per identificare la copertura spaziale del dataset.

### 2. Mappa Geografica dei Cluster con Contextily
* **Descrizione:** La mappa geografica dei cluster sovrapposta a una mappa satellitare o stradale di sfondo (basemap) fornita dalla libreria `contextily`. Ogni punto è colorato in base al cluster di appartenenza.
* **Obiettivo:** Fornire una validazione visiva sul campo. Permette agli agronomi di verificare se i cluster identificati corrispondono a effettive zone geografiche omogenee, confini di bacini idrografici o specifiche conformazioni geomorfologiche del territorio.

### 3. Heatmap di Correlazione dei Cluster
* **Descrizione:** Una matrice di correlazione grafica (Heatmap) applicata alle feature all'interno dei singoli cluster o per valutare la relazione tra i centroidi/medoidi dei cluster.
* **Obiettivo:** Evidenziare come cambiano le relazioni tra le variabili da un cluster all'altro, rivelando dinamiche chimico-fisiche specifiche per ogni tipo di suolo.

### 4. Original Profile Features Heatmap
* **Descrizione:** Una mappa termica che mostra i valori medi (o mediani) standardizzati delle **feature originali** del suolo (pH, argilla, sostanza organica, ecc.) per ciascuno dei cluster identificati.
* **Obiettivo:** Definire l'identikit chimico-fisico di ogni cluster; permette di leggere il profilo del suolo.

### 5. InterCluster Projection (Mappa delle Distanze)
* **Descrizione:** Una proiezione bidimensionale (tramite tecniche come la Multi-Dimensional Scaling - MDS o la PCA dei centroidi) che mostra la distanza relativa tra i diversi cluster scoperti.
* **Obiettivo:** Valutare la separazione globale dei modelli. Cluster molto distanti indicano tipi di suolo nettamente separati, mentre cerchi vicini o sovrapposti indicano zone di transizione sfumate o un potenziale sovra-frazionamento dei dati.

### 6. Grafico della Silhouette (Silhouette Plot)
* **Descrizione:** Un grafico che mostra lo spessore e la lunghezza del coefficiente di silhouette per ogni singolo punto, raggruppato per cluster, confrontato con la linea della media globale.
* **Obiettivo:** Validazione matematica della qualità del clustering. Consente di vedere se ci sono punti assegnati erroneamente (valori negativi) o se un cluster è troppo debole e disomogeneo al suo interno (valori sotto la media).

## 🌾 Validazione ed Elaborazione Dati per AquaCrop

Questa fase si concentra sulla preparazione, pulizia e validazione del dataset agrometeo necessario per alimentare correttamente il software di simulazione colturale **AquaCrop**. I dati provenienti dalle stazioni meteorologiche locali sono stati integrati e sincronizzati con i dati globali rianalizzati.

---

### 🧹 Data Processing & Allineamento Temporale

* **Ristrutturazione delle Variabili:** Uniformazione del dataset tramite la rinominazione delle feature con la funzione `rename()`.
* **Rimozione della Temperatura Media (`t_mean`):** La colonna è stata rimossa dal dataset poiché rappresenta un dato puramente derivato dalla media aritmetica tra temperatura minima e massima. AquaCrop calcola questo valore in modo nativo e autonomo.
* **Verifica della Continuità Temporale:** È stata calcolata la serie temporale ideale per il periodo analizzato e confrontata con le date effettivamente presenti nel dataset tramite la funzione `difference()`. Il controllo ha confermato l'**assenza di giorni mancanti**.
* **Sincronizzazione Locale-ERA5:** I dati grezzi delle stazioni meteo locali sono stati sottoposti a sincronizzazione temporale per garantirne il perfetto allineamento con il dataset globale **ERA5**, condizione imprescindibile per l'accuratezza dei modelli di simulazione.
* **Normalizzazione dei Metadati:** Pulizia delle stringhe dei nomi delle stazioni meteorologiche tramite `strip()` (rimozione di spazi bianchi superflui) e controllo dell'univocità tramite `unique()`.
* **Allineamento Geo-Spaziale:** I dataframe delle precipitazioni (piogge) e le relative coordinate geografiche sono stati allineati specularmente sia a livello di record di dati sia a livello di corrispondenza dei nomi delle stazioni.

---

### 🩹 Gestione Avanzata dei Valori Mancanti (NaN)

Le interruzioni di servizio o i malfunzionamenti dei sensori sul campo introducono lacune nei dati che richiedono una strategia di ripristino robusta:

1. **Soglia di Tolleranza Critica (Drop 40%):** I record che presentavano una percentuale di valori mancanti **maggiore o uguale al 40%** sono stati definitivamente rimossi. Una ricostruzione su volumi di dati così degradati risulterebbe statisticamente inattendibile, introducendo bias pesanti nella simulazione.
2. **Stima Spaziale e KNN Imputation:** Per mappare il meteo nei punti geografici privi di una stazione fisica di rilevamento, ci si è basati sul principio di prossimità spaziale sfruttando le stazioni limitrofe. I rimanenti valori NaN sono stati gestiti tramite **`KNNImputer(n_neighbors=3, weights='uniform')`**.

> **Nota Metodologica:** La scelta del KNNImputer (basato sui 3 vicini più prossimi nello spazio) è cruciale in ambito meteorologico. Se un sensore si guasta durante un forte evento temporalesco, imputare il dato mancante usando la media annuale o mensile appiattirebbe il picco, rendendo la simulazione di AquaCrop irrealistica. Il KNN preserva il contesto meteo locale di quel preciso giorno.

---

### 📊 Analisi degli Outlier e Strategia di Validazione dei Cluster

Per identificare anomalie residue e verificare la bontà geochimica/meteo del clustering prima dell'input in AquaCrop, sono stati generati degli **Scatter Plot** mirati. La strategia di validazione ha previsto il confronto incrociato di specifici profili:

* **Validazione Intra-Cluster (Consistenza):** Sono stati selezionati e confrontati graficamente due punti appartenenti allo stesso gruppo: il **Medoide** (il punto più rappresentativo e centrale) e un secondo campione casuale del medesimo cluster. Il grafico ha confermato l'elevata similarità e coerenza dei comportamenti agronomici all'interno dello stesso gruppo.
* **Validazione Inter-Cluster (Separazione):** È stato eseguito un confronto diretto tra due punti estratti da **due cluster differenti**. La netta divergenza spaziale e temporale dei trend ha validato matematicamente la capacità dei modelli di separare il territorio in macro-aree dalle caratteristiche effettivamente distinte.
