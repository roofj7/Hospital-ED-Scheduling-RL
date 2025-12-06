# paper_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_paper_accurate_data(num_patients=11000):
    """
    Generate synthetic ED data following EXACT specifications from the paper
    """
    np.random.seed(42)
    
    # Acuity levels from Paper Table 4
    acuity_levels = [1, 2, 3, 4, 5]
    acuity_ratios = [0.10, 0.30, 0.40, 0.10, 0.10]
    acuity_weights = [30, 15, 1, 1, 1]
    
    acuity_distribution = np.random.choice(acuity_levels, size=num_patients, p=acuity_ratios)
    weight_distribution = [acuity_weights[acuity-1] for acuity in acuity_distribution]
    
    # Treatment patterns from Paper Table 1
    treatment_patterns = {
        1: ['A', 'B', 'C', 'H'],
        2: ['A', 'B', 'C', 'E', 'H'],
        3: ['A', 'B', 'C', 'D', 'E', 'H'],
        4: ['A', 'B', 'C', 'F', 'H'],
        5: ['A', 'B', 'C', 'D', 'E', 'F', 'H'],
        6: ['A', 'B', 'C', 'D', 'H'],
        7: ['A', 'B', 'C', 'E', 'D', 'H'],
        8: ['A', 'B', 'C', 'G', 'H'],
        9: ['A', 'B', 'C', 'G', 'E', 'F', 'I'],
        10: ['A', 'B', 'C', 'D', 'I'],
        11: ['A', 'B', 'C', 'E', 'D', 'I']
    }
    
    # Assign random treatment patterns
    pattern_ids = np.random.choice(list(treatment_patterns.keys()), size=num_patients)
    
    # Arrival times - Poisson process λ=7
    inter_arrival_times = np.random.exponential(1/7, num_patients)
    arrival_times = [datetime(2023, 1, 1) + timedelta(hours=float(np.sum(inter_arrival_times[:i]))) 
                    for i in range(num_patients)]
    
    # Generate patient data
    patient_data = []
    for i in range(num_patients):
        acuity = acuity_distribution[i]
        pattern_id = pattern_ids[i]
        treatment_sequence = treatment_patterns[pattern_id]
        
        # Generate processing times for each treatment
        processing_times = []
        for treatment in treatment_sequence:
            if treatment == 'A':  # Triage
                time = np.random.exponential(7)
            elif treatment == 'B':  # Registration
                time = np.random.exponential(5.5)
            elif treatment == 'C':  # Evaluation
                time = np.random.normal(14, 6)
            elif treatment == 'D':  # Laboratory
                time = np.random.normal(35, 15)
            elif treatment == 'E':  # X-ray
                time = np.random.exponential(12)
            elif treatment == 'F':  # Consultation
                time = np.random.normal(15, 8)
            elif treatment == 'G':  # CT scan
                time = np.random.normal(29, 14)
            elif treatment in ['H', 'I']:  # Discharge
                time = 30 if treatment == 'H' else 3
            processing_times.append(max(0.1, time))
        
        total_processing_time = sum(processing_times)
        
        patient_data.append({
            'patient_id': f"P{i+1:06d}",
            'arrival_time': arrival_times[i],
            'acuity_level': acuity,
            'weighted_acuity': weight_distribution[i],
            'treatment_pattern_id': pattern_id,
            'treatment_sequence': '→'.join(treatment_sequence),
            'total_processing_time': total_processing_time,
            'num_treatments': len(treatment_sequence),
            'age': np.random.randint(18, 90),
            'gender': np.random.choice(['M', 'F'], p=[0.55, 0.45]),
            'requires_admission': 1 if 'I' in treatment_sequence else 0,
            'pain_level': np.random.randint(1, 6),
            'critical_status': np.random.randint(1, 6)
        })
    
    # Create triage data
    triage_data = []
    for i, patient in enumerate(patient_data):
        triage_data.append({
            'triage_code': f"T{patient['patient_id'][1:]}",
            'TriageGrade': patient['acuity_level'],
            'PainGrade': patient['pain_level'],
            'CriticalStatus': patient['critical_status'],
            'age': patient['age'],
            'gender': patient['gender'],
            'arrival_timestamp': patient['arrival_time'],
            'acuity_weight': patient['weighted_acuity'],
            'BlooddpressurSystol': np.random.normal(130, 25),
            'BlooddpressurDiastol': np.random.normal(85, 15),
            'PulseRate': np.random.normal(80, 20),
            'RespiratoryRate': np.random.normal(18, 4),
            'Temperature': np.random.normal(36.8, 0.8),
            'O2Saturation': np.random.normal(97, 2),
            'AVPU': np.random.choice(['A', 'V', 'P', 'U'], p=[0.8, 0.1, 0.07, 0.03]),
            'marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], p=[0.3, 0.5, 0.1, 0.1])
        })
    
    triage_df = pd.DataFrame(triage_data)
    
    # Create admission data
    admission_data = []
    for i, patient in enumerate(patient_data):
        service_lab = len([t for t in treatment_patterns[patient['treatment_pattern_id']] if t == 'D'])
        service_action = len([t for t in treatment_patterns[patient['treatment_pattern_id']] if t in ['C', 'F']])
        service_graphy = len([t for t in treatment_patterns[patient['treatment_pattern_id']] if t in ['E', 'G']])
        
        admission_data.append({
            'triage_code': f"T{patient['patient_id'][1:]}",
            'admission_hour': patient['arrival_time'].hour + patient['arrival_time'].minute/60,
            'ResidentDay': patient['requires_admission'],
            'service_count_lab': service_lab,
            'service_count_action': service_action,
            'service_count_graphy': service_graphy,
            'service_count_instrument': np.random.poisson(1),
            'total_processing_time': patient['total_processing_time'],
            'num_treatments': patient['num_treatments'],
            'treatment_pattern': patient['treatment_pattern_id'],
            'StatusOnDischarge': 'Admitted' if patient['requires_admission'] else 'Home',
            'DischargeFromED': 0 if patient['requires_admission'] else 1,
            'Foreigners': np.random.choice([0, 1], p=[0.95, 0.05]),
            'operational_patient': np.random.choice([0, 1], p=[0.8, 0.2]),
            'ref_specialist': np.random.choice(['Cardio', 'Neuro', 'Ortho', 'General', 'None'], p=[0.1, 0.1, 0.15, 0.25, 0.4])
        })
    
    admission_df = pd.DataFrame(admission_data)
    
    return triage_df, admission_df

def print_paper_statistics(triage_df, admission_df):
    """Print statistics to verify paper specifications"""
    print("=== PAPER-ACCURATE SYNTHETIC DATA STATISTICS ===")
    print(f"Total patients: {len(triage_df)}")
    
    print("\nAcuity Level Distribution (Paper Table 4):")
    acuity_counts = triage_df['TriageGrade'].value_counts().sort_index()
    for level, count in acuity_counts.items():
        percentage = (count / len(triage_df)) * 100
        print(f"  Level {level}: {count} patients ({percentage:.1f}%)")
    
    print(f"\nAdmission Rate: {admission_df['ResidentDay'].mean()*100:.1f}%")
    print(f"Average processing time: {admission_df['total_processing_time'].mean():.1f} minutes")

if __name__ == "__main__":
    print("Generating paper-accurate synthetic ED data (11,000 patients)...")
    triage_df, admission_df = generate_paper_accurate_data(11000)
    
    # Save to CSV files
    triage_df.to_csv("ED_triage.csv", index=False)
    admission_df.to_csv("ED_admission.csv", index=False)
    
    print("Paper-accurate synthetic data generated!")
    print_paper_statistics(triage_df, admission_df)
    print(f"\nFiles saved: 'ED_triage.csv' ({triage_df.shape}), 'ED_admission.csv' ({admission_df.shape})")