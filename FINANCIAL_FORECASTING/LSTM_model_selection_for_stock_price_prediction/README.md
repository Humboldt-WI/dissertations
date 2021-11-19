# Finanzprognosen mit Deep Learning
Eine empirische Evaluation der Bayesschen Optimierung für die Selektion von Hyperparametern

Im Folgenden wird erklärt, wie der zur Verfügung gestellte Code dieser Arbeit zu verwenden ist.

Die Python Datei MAIN.py ist abhängig von den folgenden Dateien:
- LSTM_Model.py: Beinhaltet den Code für das reine LSTM Modell (wird in MAIN.py importiert)
- IBM_Adj-Cl-H-L-V.csv: Enthält die Finanzdaten (in Form einer CSV-Datei), die in der Arbeit verwendet wurden.

Die YahooFinanceAPI.py Datei enthält den Code, der die IBM_Adj-Cl-H-L-V.csv erstellt. Diese ist für die Ausführung
des Programms (MAIN.py) nicht vonnöten.

Damit die MAIN.py Datei ausgeführt werden kann, müssen die abhängigen Dateien im selben Verzeichnis gespeichert werden,
wie die MAIN.py Datei.

Die optM Variable der MAIN.py Datei ermöglicht die Auswahl der Hyperparameter Optimierungsmethode.
- "BO": Bayessche Optimierung
- "GS": Grid Search
- "RS": Random Search
