import pandas as pd
import numpy as np

def compute_driver_energy_features(driver_laps: pd.DataFrame, driver_telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lap-by-lap energy features for a single driver across a race.
    
    driver_laps: FastF1 laps DataFrame for the driver
    driver_telemetry: FastF1 telemetry DataFrame for the driver
    
    Returns:
        DataFrame aligned with driver_laps containing energy features:
        - inferred_soc
        - deploy_time_s
        - harvest_time_s
        - clipping_time_s
    """
    if driver_laps.empty or driver_telemetry.empty:
        return pd.DataFrame()
        
    features = []
    
    # Simple physics model constants (approximations)
    # Assumes SoC is on a 0.0 to 1.0 scale
    MAX_SOC = 1.0
    MIN_SOC = 0.0
    DEPLOY_RATE = 0.05  # SoC consumed per second of deployment
    HARVEST_RATE = 0.03 # SoC gained per second of heavy harvesting
    LIFT_COAST_RATE = 0.01 # SoC gained per second of lift and coast
    
    current_soc = 0.8  # Assume starting the race at 80% charge
    
    # Ensure telemetry is sorted by time
    tel = driver_telemetry.sort_values('Time').copy()
    tel['dt'] = tel['Time'].dt.total_seconds().diff().fillna(0.0)
    
    # Basic states
    # FastF1 telemetry might have these as numeric or boolean
    tel['is_braking'] = tel['Brake'] > 0
    tel['is_full_throttle'] = tel['Throttle'] >= 95.0
    tel['is_lift_coast'] = (tel['Throttle'] < 10.0) & (~tel['is_braking'])
    
    # Calculate acceleration (km/h per second)
    tel['acceleration'] = tel['Speed'].diff() / tel['dt'].replace(0, np.nan)
    tel['acceleration'] = tel['acceleration'].fillna(0)
    
    # Clipping detection: full throttle, high speed, negligible acceleration
    # Thresholds: > 280 km/h, accel between -2 and +2 km/h per second
    tel['is_clipping'] = tel['is_full_throttle'] & (tel['Speed'] > 280) & (tel['acceleration'].abs() < 2.0)
    
    # Active deployment: full throttle and NOT clipping
    tel['is_deploying'] = tel['is_full_throttle'] & (~tel['is_clipping'])
    
    # We iterate lap by lap to maintain the running SoC
    for _, lap in driver_laps.iterrows():
        lap_num = lap['LapNumber']
        
        if pd.isna(lap_num):
            continue
            
        start_time = lap.get('LapStartTime')
        end_time = lap.get('Time')
        
        if pd.isna(start_time) or pd.isna(end_time):
            features.append({
                'LapNumber': lap_num,
                'inferred_soc': current_soc,
                'deploy_time_s': 0.0,
                'harvest_time_s': 0.0,
                'clipping_time_s': 0.0,
            })
            continue
            
        lap_tel = tel[(tel['Time'] >= start_time) & (tel['Time'] <= end_time)]
        
        if lap_tel.empty:
            features.append({
                'LapNumber': lap_num,
                'inferred_soc': current_soc,
                'deploy_time_s': 0.0,
                'harvest_time_s': 0.0,
                'clipping_time_s': 0.0,
            })
            continue
            
        # Aggregate times
        deploy_t = lap_tel.loc[lap_tel['is_deploying'], 'dt'].sum()
        brake_t = lap_tel.loc[lap_tel['is_braking'], 'dt'].sum()
        lift_t = lap_tel.loc[lap_tel['is_lift_coast'], 'dt'].sum()
        clip_t = lap_tel.loc[lap_tel['is_clipping'], 'dt'].sum()
        
        # Update SoC
        soc_gained = (brake_t * HARVEST_RATE) + (lift_t * LIFT_COAST_RATE)
        soc_spent = deploy_t * DEPLOY_RATE
        
        current_soc = current_soc + soc_gained - soc_spent
        current_soc = max(MIN_SOC, min(MAX_SOC, current_soc))
        
        features.append({
            'LapNumber': lap_num,
            'inferred_soc': current_soc,
            'deploy_time_s': deploy_t,
            'harvest_time_s': brake_t + lift_t,
            'clipping_time_s': clip_t,
        })
        
    return pd.DataFrame(features)
