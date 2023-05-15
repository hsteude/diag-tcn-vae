# Datagen no.1 

Moin ihr beiden, hier die Minidoku zu den Datasets. 

Die Datasets sind nummeriert. Die Datasetnummer entspricht der Nummer des Processplant Setups (cf. fig. "cppsSetups.drawio.png").
Es gibt vier Module (mixer, filter, distill, bottling) (cf. figs. "<module>_drawing.png").
Den Aufbau der Komponenten könnt ihr den Bildern entnehmen oder einfach bei den Simulationsmodellen nachsehen: https://imb-git.hsu-hh.de/imb/datasets/benchmark-for-reconfiguration-planning 

Aktuell weisen die Datasets zwei Anomalien auf: leakage (l) und clogging (c), sowie deren Kombination (lc).
Bei der leakage Anomalie wird das Ventil `valve_leakage` geöffnet und ein Volumenstrom in die `sink` abgeleitet.
Bei der clogging Anomalie wird die Öffnung des Ventils `valve_clogging` reduziert. 
Bei der Kombination beider Anomalien werden logischerweise beide ventile manipuliert. 

Ein dataset beinhaltet einen Simluationsdurchlauf. Dabei werden 4000s Laufzeit des systems simuliert. Die Samolingzeit beträgt 1s.
Nach 3000s werden die Anomalien eingeleitet. 

Die Benahmung der Datasets ist: ds<dataset-number><anomaly>.csv

Die Datasets beinhalten ziemlich viel quatschige Simulationsparameter. Für den Benchmark hatte ich nur Columns aufgenommen, die die folgenden Strings enthalten:
["time", "v_flow", "level", "m_flow", "fluidVolume", "N_in", "opening", "medium.t", "port_a.p", "port_b.p"]


# TODOs 

Ich werde mich morgen mal daransetzen noch die weitere Anomalie "motorkaputt und dreht langsamer" (m) einzubauen, und datasets zu (m), (ml), (mc) und (mcl) auszuleiten.

