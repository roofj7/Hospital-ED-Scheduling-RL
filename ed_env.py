# ed_env.py (MODIFIED for Single-Agent Multi-Resource Scheduling - MRS)
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings

# --- NEW GLOBAL DEFINITIONS ---
# Define Resource Types and their initial counts (Example, based on base paper's scenario)
RESOURCE_TYPES = {
    'Doctor': {'count': 3},  # Triage, Evaluation
    'Nurse': {'count': 5},   # Registration, Lab, Consult, Discharge
    'X_Ray_Tech': {'count': 1}, # X-ray
    'CT_Tech': {'count': 1}  # CT Scan
}
RESOURCE_NAMES = list(RESOURCE_TYPES.keys())
NUM_RESOURCES = sum(r['count'] for r in RESOURCE_TYPES.values())

# Mapping to know which feature column is satisfied by which resource type
SERVICE_TO_RESOURCE_MAP = {
    'service_count_action': 'Doctor',       # Action refers to complex procedures, map to Doctor/Nurse
    'service_count_lab': 'Nurse',           # Lab processing is often nurse-assisted or managed
    'service_count_graphy': 'X_Ray_Tech',   # Graphy (X-ray, CT) - simplify to X_Ray_Tech/CT_Tech for now
    # We rely on feature matching in the __init__ to connect these strings to actual columns
}

class EDEmergencyEnv(gym.Env):
    """
    ED environment modified for Single-Agent Multi-Resource Scheduling (MRS).
    Action: A discrete index representing (Patient Index, Resource Type Index) assignment.
    """

    def __init__(self, triage_csv: str, admission_csv: str, max_patients: int = 10, episode_length: int = 200):
        super().__init__()
        # --- MODIFICATION: Initialize resources and map to data features ---
        self.max_patients = max_patients
        self.episode_length = episode_length
        self.resource_names = RESOURCE_NAMES 
        self.resource_counts = {k: v['count'] for k, v in RESOURCE_TYPES.items()}
        # resource_pools stores the busy time remaining for each resource unit (0 = Free)
        self.resource_pools = {name: [0] * count for name, count in self.resource_counts.items()} 
        
        # Load files (will raise FileNotFoundError if path wrong)
        self.triage = pd.read_csv(triage_csv)
        self.admission = pd.read_csv(admission_csv)

        # ... [Existing merge and feature selection logic remains the same] ...
        join_key = "triage_code"
        if join_key in self.triage.columns and join_key in self.admission.columns:
            self.data = pd.merge(self.triage, self.admission, on=join_key, how="inner", suffixes=("_tri", "_adm"))
        else:
            common = set(self.triage.columns).intersection(set(self.admission.columns))
            if common:
                j = next(iter(common))
                warnings.warn(f"'{join_key}' not found in both files. Falling back to join on '{j}'.")
                self.data = pd.merge(self.triage, self.admission, on=j, how="inner", suffixes=("_tri", "_adm"))
            else:
                raise ValueError("No common join key found between triage and admission CSVs.")

        if self.data.shape[0] == 0:
            raise ValueError("Merged data is empty. Check your CSVs / join key.")

        # Candidate features (in order of preference). We'll keep only those present.
        candidate_numeric = [
            "age", "PainGrade", "CriticalStatus", "NeedFastExecute",
            "BlooddpressurSystol", "BlooddpressurDiastol", "PulseRate",
            "RespiratoryRate", "Temperature", "O2Saturation", "ResidentDay",
            "service_count_action", "service_count_instrument", "service_count_lab", "service_count_graphy"
        ]

        candidate_categorical = [
            "TriageGrade", "AVPU", "kindref", "explainer_id", "operational_patient", "ref_specialist",
            "StatusOnDischarge", "DischargeFromED", "Foreigners", "marital_Status", "gender"
        ]

        # Normalize column names for matching (but keep original columns accessible)
        cols_lower = {c.lower(): c for c in self.data.columns}

        # Helper to map desired col name (case-insensitive) to actual
        def get_actual(colname):
            return cols_lower.get(colname.lower(), None)

        # Build final feature list
        self.numeric_features = []
        self.categorical_features = []

        for c in candidate_numeric:
            a = get_actual(c)
            if a is not None:
                self.numeric_features.append(a)

        for c in candidate_categorical:
            a = get_actual(c)
            if a is not None:
                self.categorical_features.append(a)

        if len(self.numeric_features) + len(self.categorical_features) == 0:
            raise ValueError("No usable features detected in your CSVs. Columns found:\n" + ", ".join(self.data.columns))

        # Prepare a working DataFrame with only used columns
        used_cols = self.numeric_features + self.categorical_features
        self.data_work = self.data[used_cols].copy()

        # Fill missing values
        self.data_work[self.numeric_features] = self.data_work[self.numeric_features].fillna(0)
        self.data_work[self.categorical_features] = self.data_work[self.categorical_features].fillna("NA")

        # Encode categorical features with simple ordinal encoding (small and deterministic)
        self.cat_maps = {}
        for c in self.categorical_features:
            vals = sorted(list(self.data_work[c].astype(str).unique()))
            self.cat_maps[c] = {v: i for i, v in enumerate(vals)}
            self.data_work[c] = self.data_work[c].astype(str).map(self.cat_maps[c])

        # Normalize numeric features to [0,1]
        self.num_mins = {}
        self.num_ranges = {}
        for c in self.numeric_features:
            col = self.data_work[c].astype(float)
            mn = float(col.min())
            mx = float(col.max())
            rng = mx - mn if mx - mn > 0 else 1.0
            self.num_mins[c] = mn
            self.num_ranges[c] = rng
            self.data_work[c] = ((col - mn) / rng).astype(float)

        # Final feature vector order
        self.feature_cols = self.numeric_features + self.categorical_features
        self.n_features = len(self.feature_cols)
        
        # --- NEW: Track feature indices for assignment/reward calculation ---
        self.triage_idx = next((i for i, c in enumerate(self.feature_cols) if c.lower() == "triagegrade"), -1)
        self.service_indices = {
            k: next((i for i, c in enumerate(self.feature_cols) if k in c.lower()), -1)
            for k in ['service_count_action', 'service_count_lab', 'service_count_graphy']
        }
        
        # --- MODIFICATION: Action Space (Combinatorial) ---
        # Action space is Patient index (0..M-1) * Resource type index (0..R_types-1) + 1 (for No-Op)
        self.n_resource_types = len(self.resource_names)
        self.action_space_size = self.max_patients * self.n_resource_types + 1 # +1 for No-Op
        self.action_space = spaces.Discrete(self.action_space_size)

        # --- MODIFICATION: Observation Space (Includes resource status) ---
        # State: flattened queue (M * N) + resource availability ratio (R_types)
        total_obs_size = self.max_patients * self.n_features + self.n_resource_types 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(total_obs_size,), dtype=np.float32)

        self.reset()
        
    def _sample_new_patient_row(self):
        # return a 1 x n_features numpy array
        row = self.data_work.sample(1).iloc[0]
        return row[self.feature_cols].to_numpy(dtype=float)

    def reset(self):
        # sample initial queue
        sampled = self.data_work.sample(min(self.max_patients, len(self.data_work)), replace=False)
        # If fewer rows than max_patients, pad with random samples (with replacement)
        if sampled.shape[0] < self.max_patients:
            extra = self.data_work.sample(self.max_patients - sampled.shape[0], replace=True)
            sampled = pd.concat([sampled, extra], ignore_index=True)
        self.queue = sampled[self.feature_cols].to_numpy(dtype=float)  # shape (max_patients, n_features)
        self.waiting_time = np.zeros(self.max_patients, dtype=float)
        
        # --- MODIFICATION: Reset Resource Pools ---
        self.resource_pools = {name: [0] * count for name, count in self.resource_counts.items()}
        self.steps = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # --- MODIFICATION: Flatten patient data and append resource status ---
        patient_features = self.queue.flatten().astype(np.float32)
        
        # Calculate resource availability ratio (Resource Status Features)
        resource_status = []
        for name, pool in self.resource_pools.items():
            busy_count = sum(1 for time in pool if time > 0)
            availability_ratio = (self.resource_counts[name] - busy_count) / self.resource_counts[name]
            resource_status.append(availability_ratio)
            
        resource_status = np.array(resource_status, dtype=np.float32)
        
        return np.concatenate([patient_features, resource_status])

    def step(self, action: int):
        # --- MODIFICATION: Step logic for MRS ---
        
        # 1. Decode Action
        if action == self.action_space_size - 1:
            patient_idx = -1
            resource_name = 'No-Op'
        else:
            patient_idx = action // self.n_resource_types
            resource_type_idx = action % self.n_resource_types
            resource_name = self.resource_names[resource_type_idx]
            
        served_patient_idx = -1
        reward_for_action = 0.0
        
        # 2. Update Time and Calculate Base Waiting Penalty
        self.steps += 1
        
        weighted_wait_penalty = 0.0
        for i in range(self.max_patients):
            # Calculate weighted penalty for *all* waiting patients (utility part of reward)
            triage_val = self.queue[i, self.triage_idx] if self.triage_idx != -1 else 0.5
            # Invert triage (lower grade = higher acuity = higher penalty/weight)
            weight = (1.0 - triage_val) * 30.0 + 1.0 
            weighted_wait_penalty += - (weight / (self.max_patients * 30.0)) * 0.1   # tiny per-step penalty
            self.waiting_time[i] += 1.0
        
        # Decrement resource busy timers
        for pool in self.resource_pools.values():
            for i in range(len(pool)):
                pool[i] = max(0, pool[i] - 1)
        
        # 3. Process Assignment (if not No-Op)
        if patient_idx != -1:
            
            # Check for needed resource and service count
            resource_needed = None
            service_count = 0.0
            
            for service_col, required_resource in SERVICE_TO_RESOURCE_MAP.items():
                if required_resource == resource_name:
                    idx = self.service_indices.get(service_col)
                    if idx != -1 and self.queue[patient_idx, idx] > 0:
                        resource_needed = required_resource
                        service_count = self.queue[patient_idx, idx]
                        break 
                        
            # Check for idle resource unit
            idle_resource_unit_idx = -1
            if resource_needed == resource_name:
                pool = self.resource_pools[resource_name]
                try:
                    idle_resource_unit_idx = pool.index(0)
                except ValueError:
                    pass # No idle unit
            
            # Successful assignment/completion
            if idle_resource_unit_idx != -1:
                
                # Assume service count (normalized 0-1) is proportional to process time
                approx_proc_time = int(service_count * 10) + 1 
                
                # Mark resource as busy
                self.resource_pools[resource_name][idle_resource_unit_idx] = approx_proc_time
                served_patient_idx = patient_idx # Mark patient as served
                
                # --- NEW: Reward for successful assignment/completion ---
                # Reward is based on acuity (utility) - Higher is better
                triage_val = self.queue[patient_idx, self.triage_idx] if self.triage_idx != -1 else 0.5
                reward_for_action = 100.0 * (1.0 - triage_val) # Higher Acuity (lower triage) = higher reward
                
                # Since we must maintain fixed queue size, we assume this service completes instantly.
                # Decrease the service count, and if zero, the patient is discharged/replaced.
                self.queue[patient_idx, self.service_indices.get(service_col)] = 0.0
                
                # NOTE: For simplicity, assume ANY successful assignment (even if not final) leads to replacement,
                # mimicking the core DRL loop and ensuring a non-stuck environment.
        
        # 4. Queue Management: Replace if served or if all services are done
        if served_patient_idx != -1:
            # 1. Remove served patient
            self.queue = np.delete(self.queue, served_patient_idx, axis=0)
            self.waiting_time = np.delete(self.waiting_time, served_patient_idx)
            
            # 2. Add new patient
            self.queue = np.vstack([self.queue, self._sample_new_patient_row()])
            self.waiting_time = np.append(self.waiting_time, 0.0)
            
            # 3. Final Reward Calculation
            reward = weighted_wait_penalty + reward_for_action
        else:
            # Penalty for No-Op or Invalid Action when resources were idle (Wasted Step)
            idle_resources = sum(pool.count(0) for pool in self.resource_pools.values())
            if idle_resources > 0 and patient_idx == -1:
                 # Penalty for explicit No-Op when resources are free
                reward = weighted_wait_penalty - 5.0 * idle_resources
            else:
                # Neutral step (e.g., resources were busy, or agent tried invalid action)
                reward = weighted_wait_penalty
        
        # Check termination
        terminated = self.steps >= self.episode_length
        truncated = False 
        
        return self._get_obs(), float(reward), terminated, truncated, {}