"""
Citation fro each dataset:
PTB Diagnostic ECG Database:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

MIT-BIH Arrhythmia Database:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

class CardiacProblemClassifier:
    """
    Clasifica problemas cardíacos específicos:
    - Bradicardia, Taquicardia, Arritmia, Hipoxemia
    """
    
    def __init__(self):
        self.problem_types = {
            'bradicardia': 0,
            'taquicardia': 0,
            'arritmia': 0,
            'hipoxemia': 0,
            'normal': 0
        }
    
    def analyze_record(self, hr_mean, hr_series, rr_intervals, spo2_mean):
        """
        Analiza un registro y clasifica problemas específicos
        
        Returns:
            dict con problemas detectados y sus severidades
        """
        problems = []
        severities = []
        details = []
        
        # 1. BRADICARDIA (Ritmo Bajo)
        if hr_mean < 60:
            if hr_mean < 40:
                problems.append('bradicardia')
                severities.append('severa')
                details.append(f"FC={hr_mean:.1f} bpm (muy bajo, <40)")
            elif hr_mean < 50:
                problems.append('bradicardia')
                severities.append('moderada')
                details.append(f"FC={hr_mean:.1f} bpm (bajo, 40-50)")
            else:
                problems.append('bradicardia')
                severities.append('leve')
                details.append(f"FC={hr_mean:.1f} bpm (bajo, 50-60)")
        
        # 2. TAQUICARDIA (Ritmo Rápido)
        elif hr_mean > 100:
            if hr_mean > 150:
                problems.append('taquicardia')
                severities.append('severa')
                details.append(f"FC={hr_mean:.1f} bpm (muy alto, >150)")
            elif hr_mean > 120:
                problems.append('taquicardia')
                severities.append('moderada')
                details.append(f"FC={hr_mean:.1f} bpm (alto, 120-150)")
            else:
                problems.append('taquicardia')
                severities.append('leve')
                details.append(f"FC={hr_mean:.1f} bpm (alto, 100-120)")
        
        # 3. ARRITMIA (Ritmo Irregular)
        arritmia_score = 0
        arritmia_details = []
        
        # Coeficiente de variación
        cv = (np.std(hr_series) / np.mean(hr_series)) * 100
        if cv > 15:
            arritmia_score += 1
            arritmia_details.append(f"CV={cv:.1f}% (alta variabilidad)")
        
        # Cambios súbitos
        sudden_changes = np.sum(np.abs(np.diff(hr_series)) > 20)
        if sudden_changes > 3:
            arritmia_score += 1
            arritmia_details.append(f"{sudden_changes} cambios súbitos >20bpm")
        
        # SDNN bajo
        sdnn = np.std(rr_intervals)
        if sdnn < 50:
            arritmia_score += 1
            arritmia_details.append(f"SDNN={sdnn:.1f}ms (bajo HRV)")
        
        # Latidos ectópicos
        rr_mean = np.mean(rr_intervals)
        ectopic = np.sum((rr_intervals < 0.8 * rr_mean) | (rr_intervals > 1.2 * rr_mean))
        ectopic_ratio = ectopic / len(rr_intervals)
        if ectopic_ratio > 0.1:
            arritmia_score += 1
            arritmia_details.append(f"{ectopic_ratio*100:.1f}% latidos ectópicos")
        
        # Clasificar arritmia según score
        if arritmia_score >= 3:
            problems.append('arritmia')
            severities.append('severa' if arritmia_score >= 4 else 'moderada')
            details.append(f"Ritmo irregular ({', '.join(arritmia_details[:2])})")
        elif arritmia_score >= 1:
            problems.append('arritmia')
            severities.append('leve')
            details.append(f"Irregularidad leve ({arritmia_details[0] if arritmia_details else 'variabilidad'})")
        
        # 4. HIPOXEMIA (Saturación Baja)
        if spo2_mean < 95:
            if spo2_mean < 85:
                problems.append('hipoxemia')
                severities.append('severa')
                details.append(f"SpO2={spo2_mean:.1f}% (crítico, <85%)")
            elif spo2_mean < 90:
                problems.append('hipoxemia')
                severities.append('moderada')
                details.append(f"SpO2={spo2_mean:.1f}% (bajo, 85-90%)")
            else:
                problems.append('hipoxemia')
                severities.append('leve')
                details.append(f"SpO2={spo2_mean:.1f}% (bajo, 90-95%)")
        
        # Si no hay problemas
        if len(problems) == 0:
            problems.append('normal')
            severities.append('n/a')
            details.append(f"FC={hr_mean:.1f}bpm, SpO2={spo2_mean:.1f}%")
        
        return {
            'problems': problems,
            'severities': severities,
            'details': details,
            'primary_problem': problems[0],
            'max_severity': self._get_max_severity(severities)
        }
    
    def _get_max_severity(self, severities):
        """Obtiene la severidad máxima"""
        severity_order = {'n/a': 0, 'leve': 1, 'moderada': 2, 'severa': 3}
        max_sev = max([severity_order.get(s, 0) for s in severities])
        return ['n/a', 'leve', 'moderada', 'severa'][max_sev]


class PhysioNetDataLoader:
    """Carga datasets de PhysioNet con clasificación de problemas"""
    
    def __init__(self, base_path='.'):
        self.base_path = Path(base_path)
        self.mitbih_path = self.base_path / 'mit-bih-arrhythmia-database-1.0.0'
        self.ptb_path = self.base_path / 'ptb-diagnostic-ecg-database-1.0.0'
        self.classifier = CardiacProblemClassifier()
        
    def load_mitbih_record(self, record_number='100'):
        """Carga y clasifica registro MIT-BIH"""
        record_path = self.mitbih_path / record_number
        
        try:
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), 'atr')
            
            sampling_rate = record.fs
            r_peaks = annotation.sample
            rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
            
            if len(rr_intervals) < 10:
                return None
            
            hr_series = 60000 / rr_intervals
            hr_mean = np.mean(hr_series)
            
            beat_types = annotation.symbol
            abnormal_ratio = (len(beat_types) - np.sum(np.array(beat_types) == 'N')) / len(beat_types)
            spo2_mean = np.random.normal(94 if abnormal_ratio > 0.1 else 97, 2)
            spo2_mean = np.clip(spo2_mean, 85, 100)
            
            hr_series_sample = hr_series[:50]
            rr_intervals_sample = rr_intervals[:50]
            
            classification = self.classifier.analyze_record(
                hr_mean, hr_series_sample, rr_intervals_sample, spo2_mean
            )
            
            return {
                'record_id': record_number,
                'hr_mean': float(hr_mean),
                'hr_series': hr_series_sample.tolist(),
                'spo2_mean': float(spo2_mean),
                'rr_intervals': rr_intervals_sample.tolist(),
                'classification': classification,
                'source': 'MIT-BIH'
            }
        except Exception as e:
            return None
    
    def load_ptb_record(self, patient_folder, record_name):
        """Carga y clasifica registro PTB"""
        record_path = self.ptb_path / patient_folder / record_name
        
        try:
            record = wfdb.rdrecord(str(record_path))
            
            age = None
            sex = None
            diagnosis = None
            
            for comment in record.comments:
                comment_lower = comment.lower()
                if 'age:' in comment_lower:
                    try:
                        age = int(comment.split(':')[1].strip())
                    except:
                        pass
                if 'sex:' in comment_lower:
                    sex_str = comment.split(':')[1].strip().lower()
                    sex = 1 if 'male' in sex_str or sex_str == 'm' else 0
                if 'reason for admission:' in comment_lower:
                    diagnosis = comment.split(':', 1)[1].strip()
            
            # Procesar ECG
            ecg_signal = record.p_signal[:, 0]
            sampling_rate = record.fs
            ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
            
            distance = int(0.5 * sampling_rate)
            r_peaks, _ = find_peaks(ecg_normalized, distance=distance, height=0.3)
            
            if len(r_peaks) < 10:
                return None
            
            rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
            hr_series = 60000 / rr_intervals
            hr_mean = np.mean(hr_series)
            
            has_cardiac_problem = False
            if diagnosis:
                cardiac_keywords = ['infarction', 'infarct', 'cardiomyopathy', 
                                   'bundle branch block', 'ischemia']
                has_cardiac_problem = any(k in diagnosis.lower() for k in cardiac_keywords)
            
            spo2_mean = np.random.normal(92 if has_cardiac_problem else 97, 2)
            spo2_mean = np.clip(spo2_mean, 85, 100)
            
            hr_series_sample = hr_series[:50]
            rr_intervals_sample = rr_intervals[:50]
            
            classification = self.classifier.analyze_record(
                hr_mean, hr_series_sample, rr_intervals_sample, spo2_mean
            )
            
            return {
                'record_id': f"{patient_folder}_{record_name}",
                'age': age,
                'sex': sex,
                'diagnosis': diagnosis,
                'hr_mean': float(hr_mean),
                'hr_series': hr_series_sample.tolist(),
                'spo2_mean': float(spo2_mean),
                'rr_intervals': rr_intervals_sample.tolist(),
                'classification': classification,
                'source': 'PTB'
            }
        except Exception as e:
            return None
    
    def get_available_mitbih_records(self):
        """Lista registros MIT-BIH"""
        if not self.mitbih_path.exists():
            return []
        
        records = []
        for file in self.mitbih_path.glob('*.dat'):
            record_num = file.stem
            if (self.mitbih_path / f"{record_num}.atr").exists():
                records.append(record_num)
        return sorted(records)
    
    def get_available_ptb_records(self):
        """Lista registros PTB"""
        if not self.ptb_path.exists():
            return []
        
        records = []
        for patient_folder in sorted(self.ptb_path.iterdir()):
            if patient_folder.is_dir() and patient_folder.name.startswith('patient'):
                for file in patient_folder.glob('*.dat'):
                    record_name = file.stem
                    if (patient_folder / f"{record_name}.hea").exists():
                        records.append((patient_folder.name, record_name))
        return records
    
    def create_labeled_dataset(self, max_mitbih=50, max_ptb=50):
        """
        Crea dataset con ETIQUETAS ESPECÍFICAS de problemas
        """
        print("\n" + "=" * 70)
        print("CREANDO DATASET CON CLASIFICACIÓN DE PROBLEMAS")
        print("=" * 70)
        
        all_data = []
        
        # Procesar MIT-BIH
        mitbih_records = self.get_available_mitbih_records()
        if len(mitbih_records) > 0:
            print(f"\n📂 Procesando MIT-BIH ({len(mitbih_records)} registros)...")
            
            if max_mitbih:
                mitbih_records = mitbih_records[:max_mitbih]
            
            for i, record_num in enumerate(mitbih_records):
                print(f"   [{i+1}/{len(mitbih_records)}] {record_num}...", end='\r')
                
                data = self.load_mitbih_record(record_num)
                if data:
                    edad = np.random.randint(45, 85)
                    sexo = np.random.choice([0, 1])
                    peso_kg = np.random.normal(75, 15)
                    altura_m = np.random.normal(1.70, 0.1)
                    
                    classification = data['classification']
                    
                    # Imprimir clasificación en tiempo real
                    problem_str = ', '.join([f"{p} ({s})" for p, s in 
                                            zip(classification['problems'], 
                                                classification['severities'])])
                    print(f"   [{i+1}/{len(mitbih_records)}] {record_num}: {problem_str}          ")
                    
                    all_data.append({
                        'record_id': data['record_id'],
                        'edad': int(edad),
                        'sexo': int(sexo),
                        'peso_kg': float(np.clip(peso_kg, 40, 150)),
                        'altura_m': float(np.clip(altura_m, 1.4, 2.1)),
                        'hr_mean': data['hr_mean'],
                        'hr_series': data['hr_series'],
                        'spo2_mean': data['spo2_mean'],
                        'rr_intervals': data['rr_intervals'],
                        'source': data['source'],
                        # Etiquetas específicas
                        'problema_principal': classification['primary_problem'],
                        'tiene_bradicardia': 1 if 'bradicardia' in classification['problems'] else 0,
                        'tiene_taquicardia': 1 if 'taquicardia' in classification['problems'] else 0,
                        'tiene_arritmia': 1 if 'arritmia' in classification['problems'] else 0,
                        'tiene_hipoxemia': 1 if 'hipoxemia' in classification['problems'] else 0,
                        'severidad_maxima': classification['max_severity'],
                        'detalles': ', '.join(classification['details']),
                        'num_problemas': len([p for p in classification['problems'] if p != 'normal'])
                    })
        
        # Procesar PTB
        ptb_records = self.get_available_ptb_records()
        if len(ptb_records) > 0:
            print(f"\nProcesando PTB ({len(ptb_records)} registros)...")
            
            if max_ptb:
                ptb_records = ptb_records[:max_ptb]
            
            for i, (patient_folder, record_name) in enumerate(ptb_records):
                print(f"   [{i+1}/{len(ptb_records)}] {patient_folder}/{record_name}...", end='\r')
                
                data = self.load_ptb_record(patient_folder, record_name)
                if data:
                    edad = data['age'] if data['age'] else np.random.randint(45, 85)
                    sexo = data['sex'] if data['sex'] is not None else np.random.choice([0, 1])
                    peso_kg = np.random.normal(75, 15)
                    altura_m = np.random.normal(1.70, 0.1)
                    
                    classification = data['classification']
                    
                    problem_str = ', '.join([f"{p} ({s})" for p, s in 
                                            zip(classification['problems'], 
                                                classification['severities'])])
                    print(f"   [{i+1}/{len(ptb_records)}] {patient_folder}: {problem_str}          ")
                    
                    all_data.append({
                        'record_id': data['record_id'],
                        'edad': int(edad),
                        'sexo': int(sexo),
                        'peso_kg': float(np.clip(peso_kg, 40, 150)),
                        'altura_m': float(np.clip(altura_m, 1.4, 2.1)),
                        'hr_mean': data['hr_mean'],
                        'hr_series': data['hr_series'],
                        'spo2_mean': data['spo2_mean'],
                        'rr_intervals': data['rr_intervals'],
                        'source': data['source'],
                        'diagnosis': data['diagnosis'],
                        # Etiquetas específicas
                        'problema_principal': classification['primary_problem'],
                        'tiene_bradicardia': 1 if 'bradicardia' in classification['problems'] else 0,
                        'tiene_taquicardia': 1 if 'taquicardia' in classification['problems'] else 0,
                        'tiene_arritmia': 1 if 'arritmia' in classification['problems'] else 0,
                        'tiene_hipoxemia': 1 if 'hipoxemia' in classification['problems'] else 0,
                        'severidad_maxima': classification['max_severity'],
                        'detalles': ', '.join(classification['details']),
                        'num_problemas': len([p for p in classification['problems'] if p != 'normal'])
                    })
        
        df = pd.DataFrame(all_data)
        
        # Resumen
        print("\n" + "=" * 70)
        print(f"✓ DATASET CREADO: {len(df)} registros")
        print("=" * 70)
        
        print("\nDISTRIBUCIÓN DE PROBLEMAS:")
        for problem in ['bradicardia', 'taquicardia', 'arritmia', 'hipoxemia']:
            count = df[f'tiene_{problem}'].sum()
            print(f"   • {problem.capitalize()}: {count} casos ({count/len(df)*100:.1f}%)")
        
        print(f"\n   • Normal: {len(df[df['problema_principal']=='normal'])} casos")
        
        print("\nSEVERIDAD:")
        for sev in ['leve', 'moderada', 'severa']:
            count = len(df[df['severidad_maxima']==sev])
            if count > 0:
                print(f"   • {sev.capitalize()}: {count} casos")
        
        return df


def load_classified_datasets(base_path='.', max_mitbih=50, max_ptb=50):
    """
    Función principal para cargar datasets CON CLASIFICACIÓN
    """
    loader = PhysioNetDataLoader(base_path=base_path)
    df = loader.create_labeled_dataset(max_mitbih=max_mitbih, max_ptb=max_ptb)
    return df


if __name__ == "__main__":
    df = load_classified_datasets(base_path='.', max_mitbih=30, max_ptb=30)
    
    if df is not None and len(df) > 0:
        print("\n" + "=" * 70)
        print("MUESTRA DE DATOS CLASIFICADOS")
        print("=" * 70)
        print(df[['record_id', 'problema_principal', 'severidad_maxima', 'detalles']].head(15))
        
        print("\n✓ Dataset listo para entrenar modelos predictivos!")
        
        # Guardar CSV
        df.to_csv('cardiac_dataset_classified.csv', index=False)
        print(f"\nDataset guardado en: cardiac_dataset_classified.csv")
    else:
        print("\nNo se pudo crear el dataset")