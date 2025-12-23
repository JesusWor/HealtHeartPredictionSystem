# HealtHeartPredictionSystem

Sistema de clasificación automática de problemas cardiovasculares basado en señales ECG y datos fisiológicos de PhysioNet.

El sistema analiza registros de electrocardiogramas reales, extrae características cardíacas y clasifica automáticamente problemas específicos con diferentes niveles de severidad.

---

## Project Overview

Este proyecto implementa un **clasificador automático de problemas cardíacos** utilizando técnicas de procesamiento de señales biomédicas y análisis de datos fisiológicos.

Al procesar señales ECG de bases de datos médicas reconocidas, el sistema identifica patrones cardíacos anormales y los clasifica en categorías específicas como bradicardia, taquicardia, arritmia e hipoxemia.

El objetivo principal del proyecto es explorar **procesamiento de señales biomédicas, análisis de datos médicos y machine learning aplicado a la salud** utilizando Python.

---

## Features

- Carga automática de datasets MIT-BIH y PTB desde PhysioNet
- Detección de picos R en señales ECG
- Cálculo de intervalos RR y variabilidad cardíaca (HRV)
- Clasificación de problemas cardíacos:
  - **Bradicardia** → Frecuencia cardíaca < 60 bpm
  - **Taquicardia** → Frecuencia cardíaca > 100 bpm
  - **Arritmia** → Ritmo irregular detectado por múltiples métricas
  - **Hipoxemia** → Saturación de oxígeno < 95%
- Niveles de severidad (leve, moderada, severa)
- Generación automática de dataset etiquetado
- Extracción de características demográficas y clínicas
- Exportación a CSV para análisis posterior
- Estadísticas detalladas de distribución de casos

---

## Technologies Used

- **Python 3.x**
- NumPy
- Pandas
- SciPy
- WFDB (WaveForm DataBase Python Package)
- Signal Processing
- Medical Data Analysis

---

## System Requirements

- Python **3.7+**
- Datasets descargados de PhysioNet:
  - MIT-BIH Arrhythmia Database
  - PTB Diagnostic ECG Database
- Mínimo 2GB de espacio en disco para datasets
- 4GB RAM recomendados

---

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/cardiac-problem-classifier.git
cd cardiac-problem-classifier
```

### Install dependencies
```bash
pip install numpy pandas scipy wfdb
```

### Download datasets

Descarga los datasets de PhysioNet y colócalos en el directorio raíz:
```
cardiac-problem-classifier/
├── mit-bih-arrhythmia-database-1.0.0/
├── ptb-diagnostic-ecg-database-1.0.0/
└── main.py
```

---

## Usage

### Generate Classified Dataset
```python
from main import load_classified_datasets

# Cargar y clasificar datos
df = load_classified_datasets(
    base_path='.',
    max_mitbih=50,
    max_ptb=50
)

print(df.head())
```

### Analyze Specific Record
```python
from main import PhysioNetDataLoader

loader = PhysioNetDataLoader(base_path='.')
data = loader.load_mitbih_record('100')
print(data['classification'])
```

### Run Full Pipeline
```bash
python main.py
```

---

## Classification System

### Problem Detection Criteria

| Problem | Description | Detection Criteria |
|---------|-------------|-------------------|
| **Bradicardia** | Frecuencia cardíaca baja | HR < 60 bpm |
| **Taquicardia** | Frecuencia cardíaca alta | HR > 100 bpm |
| **Arritmia** | Ritmo irregular | CV > 15%, cambios súbitos |
| **Hipoxemia** | Saturación baja | SpO2 < 95% |

### Severity Levels

- **Leve**: Desviación moderada
- **Moderada**: Desviación significativa
- **Severa**: Desviación crítica

---

## Project Status

**WORK IN PROGRESS**

### Completed
- [x] Dataset loading
- [x] ECG signal processing
- [x] Problem classification
- [x] Dataset generation

### In Development
- [ ] ML models
- [ ] User interface
- [ ] Real-time classification

---

## Datasets

### MIT-BIH Arrhythmia Database

**Citation:**
```
Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation [Online]. 101 (23), pp. e215–e220.
```

### PTB Diagnostic ECG Database

**Citation:**
```
Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation [Online]. 101 (23), pp. e215–e220.
```

---

## Authors

- **Jesus Eduardo Escobar Meza** - Development
- **Cristian Ricardo Luque Arambula** - Development

---

<!-- ## License

[To be defined]

--- -->

**Version:** 0.1.0 (Alpha)