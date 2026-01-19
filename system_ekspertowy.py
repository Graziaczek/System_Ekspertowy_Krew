import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules as arules


# =========================================================
# 1. DYSKRETYZACJA – PROGI (LICZONE TYLKO NA TRAIN)
# =========================================================

def compute_thresholds(data, parameters):
    thresholds = {}
    for param in parameters:
        series = data[param]
        Q1 = series.quantile(0.25)
        Q2 = series.quantile(0.50)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        thresholds[param] = {
            "Q1": Q1,
            "Q2": Q2,
            "Q3": Q3,
            "lower": Q1 - 1.5 * IQR,
            "upper": Q3 + 1.5 * IQR
        }
    return thresholds


def discretize_sample(sample, thresholds):
    discretized = {}
    for param, t in thresholds.items():
        v = sample[param]

        if v < t["Q1"]:
            level = "niski"
        elif v < t["Q2"]:
            level = "średni_dolny"
        elif v < t["Q3"]:
            level = "średni_górny"
        else:
            level = "wysoki"

        if v < t["lower"]:
            level = "ekstremalnie_niski"
        elif v > t["upper"]:
            level = "ekstremalnie_wysoki"

        discretized[param] = level

    discretized["Disease"] = sample["Disease"]
    return discretized


def discretize_dataset(data, thresholds):
    return pd.DataFrame(
        [discretize_sample(row, thresholds) for _, row in data.iterrows()]
    )


# =========================================================
# 2. SYSTEM REGUŁOWY – APRIORI
# =========================================================

def generate_association_rules(discret_dataset,
                               min_support=0.05,
                               min_confidence=0.7):

    transactions = []
    for _, row in discret_dataset.iterrows():
        transactions.append([f"{c}={row[c]}" for c in discret_dataset.columns])

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_bin = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(df_bin, min_support=min_support, use_colnames=True)
    rules = arules(frequent, metric="confidence", min_threshold=min_confidence)

    # Disease tylko w consequents
    rules = rules[
        rules["consequents"].apply(lambda x: any(i.startswith("Disease=") for i in x)) &
        rules["antecedents"].apply(lambda x: not any(i.startswith("Disease=") for i in x))
    ]

    return rules.sort_values(by="confidence", ascending=False)


def rules_to_expert_format(rules_df, top_n=800):
    expert_rules = []
    for _, row in rules_df.head(top_n).iterrows():
        antecedents = {
            i.split("=")[0]: i.split("=")[1]
            for i in row["antecedents"]
        }
        consequents = {
            i.split("=")[0]: i.split("=")[1]
            for i in row["consequents"]
        }
        expert_rules.append({
            "antecedents": antecedents,
            "consequents": consequents,
            "confidence": row["confidence"]
        })
    return expert_rules


def apply_rules(discretized_sample, expert_rules, min_confidence=0.7):
    matches = []
    for rule in expert_rules:
        if rule["confidence"] < min_confidence:
            continue
        if all(discretized_sample.get(k) == v
               for k, v in rule["antecedents"].items()):
            matches.append(rule)

    if not matches:
        return None

    matches.sort(key=lambda x: x["confidence"], reverse=True)
    return matches[0]["consequents"]["Disease"]


# =========================================================
# 3. UCZENIE MASZYNOWE – DRZEWO DECYZYJNE
# =========================================================

def train_ml_model(train_df, test_df, parameters):
    X_train = train_df[parameters]
    y_train = train_df["Disease"]

    X_test = test_df[parameters]
    y_test = test_df["Disease"]

    model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== UCZENIE MASZYNOWE (TEST) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


# =========================================================
# 4. MAIN – TRAIN / TEST DLA OBU SYSTEMÓW
# =========================================================

if __name__ == "__main__":

    data = pd.read_csv("Blood_samples.csv")

    parameters = [
        "Glucose","Cholesterol","Hemoglobin","Platelets","White Blood Cells",
        "Red Blood Cells","Hematocrit","Mean Corpuscular Volume",
        "Mean Corpuscular Hemoglobin",
        "Mean Corpuscular Hemoglobin Concentration","Insulin","BMI",
        "Systolic Blood Pressure","Diastolic Blood Pressure","Triglycerides",
        "HbA1c","LDL Cholesterol","HDL Cholesterol","ALT","AST",
        "Heart Rate","Creatinine","Troponin","C-reactive Protein"
    ]

    # -----------------------------
    # PODZIAŁ TRAIN / TEST
    # -----------------------------
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["Disease"]
    )

    # =============================
    # SYSTEM REGUŁOWY – TRAIN
    # =============================
    thresholds = compute_thresholds(train_df, parameters)
    train_disc = discretize_dataset(train_df, thresholds)

    rules_df = generate_association_rules(train_disc)
    expert_rules = rules_to_expert_format(rules_df, top_n=800)

    # =============================
    # SYSTEM REGUŁOWY – TEST
    # =============================
    correct, total, no_pred = 0, 0, 0

    for _, row in test_df.iterrows():
        true = row["Disease"]
        disc = discretize_sample(row, thresholds)
        disc.pop("Disease")

        pred = apply_rules(disc, expert_rules)

        if pred is None:
            no_pred += 1
            continue

        total += 1
        if pred == true:
            correct += 1

    acc_rules = correct / total if total else 0

    print("\n=== SYSTEM REGUŁOWY (TEST) ===")
    print("Accuracy:", acc_rules)
    print("Brak reguły:", no_pred)

    # =============================
    # UCZENIE MASZYNOWE
    # =============================
    ml_model = train_ml_model(train_df, test_df, parameters)

    # =============================
    # PORÓWNANIE
    # =============================
    print("\n=== PORÓWNANIE KOŃCOWE ===")
    print(f"Reguły IF–THEN Accuracy: {acc_rules:.2%}")
