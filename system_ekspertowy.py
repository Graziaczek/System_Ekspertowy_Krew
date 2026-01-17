import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules as arules

# -----------------------------
# Funkcja do przygotowania progów dla wszystkich parametrów
# -----------------------------
def compute_thresholds(data, parameters):
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
# Funkcja do dyskretyzacji jednej próbki (zwraca słownik parametr->poziom)
# -----------------------------
def discretize_sample(sample, thresholds):
    discretized = {}
    for param, bounds in thresholds.items():
        value = sample[param]
        Q1 = bounds["Q1"]
        Q2 = bounds["Q2"]
        Q3 = bounds["Q3"]

        if value < Q1:
            level = "niski"
        elif value < Q2:
            level = "średni-dolny"
        elif value < Q3:
            level = "średni-górny"
        else:
            level = "wysoki"

        # opcjonalnie oznaczenie outlierów
        if value < bounds["lower_bound"]:
            level = "ekstremalnie niski"
        elif value > bounds["upper_bound"]:
            level = "ekstremalnie wysoki"

        discretized[param] = level

    # dodajemy chorobę jako ostatnią kolumnę
    discretized["Disease"] = sample["Disease"]
    return discretized

# -----------------------------
# Funkcja do dyskretyzacji całego zbioru
# -----------------------------
def discretize_dataset(data, thresholds):
    discretized_rows = []
    for _, sample in data.iterrows():
        discretized_rows.append(discretize_sample(sample, thresholds))
    return pd.DataFrame(discretized_rows)

# -----------------------------
# Funkcja do generowania reguł asocjacyjnych
# -----------------------------
def generate_association_rules(discret_dataset, min_support=0.05, min_confidence=0.7):
    # tworzymy listę transakcji (każdy wiersz jako lista faktów Parametr=Poziom)
    transactions = []
    for _, row in discret_dataset.iterrows():
        facts = [f"{col}={row[col]}" for col in discret_dataset.columns if pd.notna(row[col])]
        transactions.append(facts)

    # konwersja do formatu one-hot
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = arules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # filtrowanie reguł prowadzących do choroby
    disease_rules = rules[rules['consequents'].apply(lambda x: any('Disease=' in item for item in x))]
    disease_rules = disease_rules.sort_values(by='confidence', ascending=False)
    return disease_rules

# -----------------------------
# PRZYKŁAD UŻYCIA
# -----------------------------
if __name__ == "__main__":
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

    # dyskretyzujemy cały zbiór danych
    discret_dataset = discretize_dataset(data, thresholds)
    print(discret_dataset.head())

    # generujemy reguły asocjacyjne
    rules = generate_association_rules(discret_dataset)
    pd.DataFrame(rules).to_csv("association_rules.csv", index=False)
