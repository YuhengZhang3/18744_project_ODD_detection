import os
import json
import glob

# Adjust project_root if you are running this from a different location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MERGED_DIR = os.path.join(project_root, "outputs", "merged_json")

def check_absurd_predictions(filepath):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return

    filename = os.path.basename(filepath)
    absurdities = []

    # Safely extract dictionaries (default to empty dict if None or missing)
    synth = data.get("synth_outputs") or {}
    sensors = synth.get("sensors") or {}
    
    yuheng = data.get("yuheng") or {}
    yuheng_preds = yuheng.get("predictions") or {}
    
    weather_data = data.get("weather") or {}
    # Handle both nested {"weather": {...}} and flat structures
    weather = weather_data.get("weather", weather_data)
    
    clouds = data.get("cloud_detection") or {}

    # ---------------------------------------------------------
    # 1. Temperature Constraints
    # ---------------------------------------------------------
    temp_c = sensors.get("temperature_c")
    if temp_c is not None:
        # If it's safely above freezing (e.g., > 5°C / 41°F)
        if temp_c > 5.0:
            
            # Check for high snow prediction
            snow_sev = weather.get("snow_severity")
            if snow_sev is not None and snow_sev > 0.3:
                absurdities.append(
                    f"Temp is {temp_c}°C, but weather model predicts high snow_severity ({snow_sev:.4f})"
                )
            
            # Check for icy road prediction
            road_cond = yuheng_preds.get("road_condition", {}).get("label")
            ice_labels = ["snow_covered", "slush", "black_ice", "ice"]
            if road_cond and any(ice in road_cond.lower() for ice in ice_labels):
                absurdities.append(
                    f"Temp is {temp_c}°C, but ODD model predicts road as '{road_cond}'"
                )

    # ---------------------------------------------------------
    # 2. Lighting / Time Constraints
    # ---------------------------------------------------------
    clock_time = sensors.get("clock_time")
    time_label = yuheng_preds.get("time", {}).get("label", "").lower()
    
    if clock_time and time_label:
        try:
            hour = int(clock_time.split(":")[0])
            
            # Use broad safe bounds (10 AM to 2 PM is guaranteed day; 11 PM to 3 AM is guaranteed night)
            if 10 <= hour <= 14 and time_label in ["night", "nighttime"]:
                absurdities.append(f"Clock time is {clock_time}, but ODD model predicts '{time_label}'")
                
            elif (hour >= 23 or hour <= 3) and time_label in ["day", "daytime"]:
                absurdities.append(f"Clock time is {clock_time}, but ODD model predicts '{time_label}'")
        except ValueError:
            pass

    # ---------------------------------------------------------
    # 3. Humidity Constraints
    # ---------------------------------------------------------
    humidity = sensors.get("humidity_pct")
    cloud_fraction = clouds.get("cloud_fraction")
    
    if humidity is not None and cloud_fraction is not None:
        # If humidity is near saturation (> 95%), skies are rarely perfectly clear
        if humidity > 95.0 and cloud_fraction < 0.1:
            absurdities.append(
                f"Humidity is saturated ({humidity}%), but cloud_fraction is absurdly low ({cloud_fraction:.4f})"
            )

    # ---------------------------------------------------------
    # Output Findings
    # ---------------------------------------------------------
    if absurdities:
        print(f"\n--- {filename} ---")
        for a in absurdities:
            print(f"  [!] {a}")

def main():
    if not os.path.exists(MERGED_DIR):
        print(f"Error: Directory not found -> {MERGED_DIR}")
        print("Please run the merge script first.")
        return

    json_files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.json")))
    if not json_files:
        print(f"No JSON files found in {MERGED_DIR}")
        return

    print(f"Scanning {len(json_files)} merged files for absurd predictions...")
    
    found_any = False
    for jf in json_files:
        # Check file size/validity roughly before processing
        if os.path.getsize(jf) > 0:
            # check_absurd_predictions will print directly if it finds anything
            check_absurd_predictions(jf)

if __name__ == "__main__":
    main()