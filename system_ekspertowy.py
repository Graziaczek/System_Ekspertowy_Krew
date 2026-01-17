import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules as arules

# -----------------------------
# Funkcja do przygotowania progów dla wszystkich parametrów
# -----------------------------
def compute_thresholds(data, parameters):
    """
    Oblicza kwartyle i progi ekstremalne dla wszystkich parametrów.
    Zwraca słownik thresholds[param] = {"Q1", "Q2", "Q3", "lower_bound", "upper_bound"}
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
    Zwraca słownik: {parametr: poziom}
    """
    discretized = {}
    for param, bounds in thresholds.items():
        value = sample[param]
        Q1, Q2, Q3 = bounds["Q1"], bounds["Q2"], bounds["Q3"]

        # klasyfikacja kwartylowa
        if value < Q1:
            level = "niski"
        elif value < Q2:
            level = "średni-dolny"
        elif value < Q3:
            level = "średni-górny"
        else:
            level = "wysoki"

        # oznaczenie outlierów
        if value < bounds["lower_bound"]:
            level = "ekstremalnie niski"
        elif value > bounds["upper_bound"]:
            level = "ekstremalnie wysoki"

        discretized[param] = level

    # dodajemy chorobę
    discretized["Disease"] = sample["Disease"]
    return discretized

# -----------------------------
# Dyskretyzacja całego zbioru danych
# -----------------------------
def discretize_dataset(data, thresholds):
    return pd.DataFrame([discretize_sample(row, thresholds) for _, row in data.iterrows()])

def generate_association_rules(discret_dataset, min_support=0.05, min_confidence=0.7):
    """
    Generuje reguły asocjacyjne, w których:
    - Disease NIE występuje w antecedents
    - Disease WYSTĘPUJE w consequents
    """

    transactions = []
    for _, row in discret_dataset.iterrows():
        # ❗ choroba trafia do transakcji, ale będzie kontrolowana później
        facts = [f"{col}={row[col]}" for col in discret_dataset.columns if pd.notna(row[col])]
        transactions.append(facts)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = arules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # ✅ Disease tylko w consequents
    rules = rules[
        rules['consequents'].apply(lambda x: any(i.startswith('Disease=') for i in x)) &
        rules['antecedents'].apply(lambda x: not any(i.startswith('Disease=') for i in x))
    ]

    rules = rules.sort_values(by='confidence', ascending=False)
    return rules


# -----------------------------
# Konwersja reguł do formatu eksperckiego (tylko choroby w consequents)
# -----------------------------
def rules_to_expert_format(rules_df, top_n=None):
    """
    Zwraca listę słowników:
    {"antecedents": {...}, "consequents": {...}, "confidence": ...}

    W polu consequents będą tylko choroby (parametr 'Disease').
    """
    rules_df = rules_df.sort_values(by='confidence', ascending=False)
    if top_n is not None:
        rules_df = rules_df.head(top_n)

    expert_rules = []
    for _, row in rules_df.iterrows():
        # Antecedents: wszystkie cechy oprócz Disease
        antecedents = {item.split('=')[0]: item.split('=')[1] 
                       for item in row['antecedents'] 
                       if not item.startswith("Disease=")}

        # Consequents: tylko choroba
        consequents = {item.split('=')[0]: item.split('=')[1] 
                       for item in row['consequents'] 
                       if item.startswith("Disease=")}

        # jeśli consequents jest puste, pomijamy regułę
        if not consequents:
            continue

        confidence = row['confidence']
        expert_rules.append({
            "antecedents": antecedents,
            "consequents": consequents,
            "confidence": confidence
        })

    return expert_rules


# -----------------------------
# Wnioskowanie dla jednej próbki
# -----------------------------
def apply_rules(discretized_sample, expert_rules, min_confidence=0.0):
    """
    Zwraca listę pasujących reguł w formie słowników:
    {"consequents": {...}, "confidence": ...}
    """
    inferred = []
    for rule in expert_rules:
        if rule['confidence'] < min_confidence:
            continue
        # Sprawdzenie, czy wszystkie antecedents pasują do próbki
        match = all(discretized_sample.get(param) == value
                    for param, value in rule['antecedents'].items())
        if match:
            inferred.append({
                "consequents": rule['consequents'],
                "confidence": rule['confidence']
            })

    # sortowanie po confidence malejąco
    inferred.sort(key=lambda x: x['confidence'], reverse=True)
    return inferred[0]['consequents']['Disease']  # np. 5 najbardziej pewnych reguł


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

    # 1️⃣ przygotowanie systemu
    thresholds = compute_thresholds(data, parameters)
    discret_dataset = discretize_dataset(data, thresholds)

    rules_df = generate_association_rules(discret_dataset)
    expert_rules = rules_to_expert_format(rules_df, top_n=800)

    # 2️⃣ test skuteczności
    correct = 0
    total = 0
    no_prediction = 0

    for idx, row in data.iterrows():
        true_disease = row["Disease"]

        sample_disc = discretize_sample(row, thresholds)
        sample_disc.pop("Disease")  # ❗ ukrywamy prawdziwą chorobę

        predicted = apply_rules(sample_disc, expert_rules, min_confidence=0.7)


        if predicted is None:
            no_prediction += 1
            continue

        if predicted == true_disease:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    print("=== WYNIKI TESTU ===")
    print(f"Liczba próbek: {len(data)}")
    print(f"Przewidziane: {total}")
    print(f"Brak reguły: {no_prediction}")
    print(f"Poprawne: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

