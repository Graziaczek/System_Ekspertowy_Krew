import pandas as pd

# -----------------------------
# Funkcja do przygotowania progów dla wszystkich parametrów
# -----------------------------
def compute_thresholds(data, parameters):
    """
    Oblicza kwartyle i progi outlierów dla wszystkich parametrów.
    Zwraca słownik: param -> {Q1, Q2, Q3, lower_bound, upper_bound}
    """
    thresholds = {}
    for param in parameters:
        series = data[param]
        Q1 = series.quantile(0.25)
        Q2 = series.quantile(0.50)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        thresholds[param] = {
            "Q1": Q1,
            "Q2": Q2,
            "Q3": Q3,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    return thresholds

# -----------------------------
# Funkcja do dyskretyzacji jednej próbki
# -----------------------------
def discretize_sample(sample, thresholds):
    """
    sample: pd.Series z wartościami parametrów jednej próbki
    thresholds: wynik compute_thresholds()
    
    Zwraca listę faktów w formie ["Parametr jest niski", ...]
    """
    facts = []
    
    for param, bounds in thresholds.items():
        value = sample[param]
        Q1 = bounds["Q1"]
        Q2 = bounds["Q2"]
        Q3 = bounds["Q3"]
        
        # klasyfikacja kwartylowa
        if value < Q1:
            level = "niski"
        elif value < Q2:
            level = "średni-dolny"
        elif value < Q3:
            level = "średni-górny"
        else:
            level = "wysoki"
        
        facts.append(f"{param} jest {level}")
        
        # opcjonalnie: oznaczanie outlierów
        if value < bounds["lower_bound"]:
            facts.append(f"{param} jest ekstremalnie niski")
        if value > bounds["upper_bound"]:
            facts.append(f"{param} jest ekstremalnie wysoki")
    
    return facts

# -----------------------------
# PRZYKŁAD UŻYCIA
# -----------------------------
if __name__ == "__main__":
    # wczytanie danych do wyznaczenia progów
    data = pd.read_csv("Blood_samples.csv")
    
    parameters = [
        "Glucose","Cholesterol","Hemoglobin","Platelets","White Blood Cells","Red Blood Cells",
        "Hematocrit","Mean Corpuscular Volume","Mean Corpuscular Hemoglobin",
        "Mean Corpuscular Hemoglobin Concentration","Insulin","BMI","Systolic Blood Pressure",
        "Diastolic Blood Pressure","Triglycerides","HbA1c","LDL Cholesterol","HDL Cholesterol",
        "ALT","AST","Heart Rate","Creatinine","Troponin","C-reactive Protein"
    ]
    
    # obliczamy progi
    thresholds = compute_thresholds(data, parameters)
    
    # wybieramy jedną próbkę do dyskretyzacji
    sample = data.iloc[0]
    
    # generujemy fakty
    facts = discretize_sample(sample, thresholds)
    for f in facts:
        print(f)
