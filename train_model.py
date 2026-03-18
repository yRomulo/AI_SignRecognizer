import os
import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/gesture_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"
DATASET_CSV = "dataset/gestures.csv"

def generate_synthetic_data(model, encoder, num_samples_per_class=200):
    """
    Gera dados sintéticos usando as árvores do Random Forest ou carrega X_train salvo no próprio modelo se estivesse lá.
    Como não temos os dados originais brutos, vamos usar um truque: 
    Como RandomForestClassifier não reverte inferência facilmente, e não queremos perder as classes originais,
    a melhor estratégia é simplesmente usar `.fit()` passando o conjunto TODO (antigo simulado + novo).
    
    Porém, sem o dataset antigo X_train,Y_train, treinar do zero APENAS com o CSV novo vai esquecer os gestos antigos (Catastrophic Forgetting).
    Como o Random Forest do scikit-learn não suporta `partial_fit` nativamente para adicionar novas classes mantendo as antigas sem os dados,
    nós não podemos apenas 'adicionar' árvores sem ter as features antigas, a menos que tenhamos acesso ao conjunto antigo.
    
    Portanto, como este é um projeto educacional, a melhor forma de "manter os antigos" sem o arquivo inicial CSV 
    é avisar o usuário que para gestos novos funcionarem em conjunto perfeitamente com RandomForest, ele precisa gravar os gestos 
    neste CSV ou nós precisamos salvar os antigos. 
    
    Contudo, podemos criar um script que aceita dados adicionais. Como o usuário pediu para "manter os antigos", 
    e o projeto original treina a RandomForest do zero no `train.py` original que não temos, vamos avisá-lo ou usar `warm_start=True` 
    para tentar adicionar estimadores, mas `warm_start` só funciona se o NÚMERO DE CLASSES for o mesmo.
    Se adicionarmos um gesto novo, as classes aumentam e `warm_start` falha.
    """
    pass

def train_new_model():
    print("Iniciando treinamento do modelo de gestos!")
    
    if not os.path.exists(DATASET_CSV):
        print(f"Erro: O arquivo de dataset {DATASET_CSV} não foi encontrado.")
        print("Use o collect_landmarks.py primeiro para capturar gestos.")
        return

    # 1. Carregar os dados recém gravados do CSV
    X_new = []
    y_new = []
    
    with open(DATASET_CSV, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) != 65: # 1 label + 1 handedness + 63 landmarks = 65
                continue
            
            label = row[0]
            # row[1:] são as "features", que consistem de handedness + coordenadas x,y,z
            features = [float(val) for val in row[1:]]
            
            X_new.append(features)
            y_new.append(label)

    if len(X_new) == 0:
        print("Erro: Nenhum dado de treinamento válido encontrado no CSV.")
        return

    X = np.array(X_new)
    y = np.array(y_new)
    
    print(f"Carregados {len(X)} exemplos do CSV com {len(set(y))} gestos diferentes: {set(y)}")

    # 2. Encode Labels
    # Para suportar as antigas e novas, nós DEVERÍAMOS ter os dados das antigas aqui.
    # Como não temos os dados das antigas (e o Random Forest esquecerá o que aprendeu se apenas treinarmos com as novas),
    # nós usamos um artifício para explicar ou treinamos apenas com as do CSV (assumindo que o usuário vai gravar todos os gestos de interesse).
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Treinar
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"Treinando Random Forest com {len(X_train)} amostras.")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Validação
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no conjunto de teste: {acc * 100:.1f}%")

    # 4. Salvar
    print("Salvando modelo...")
    import shutil
    # Faz backup do anterior
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, MODEL_PATH + ".bak")
    if os.path.exists(ENCODER_PATH):
        shutil.copy(ENCODER_PATH, ENCODER_PATH + ".bak")
        
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    print(f"\nModelo salvo com sucesso em {MODEL_PATH}!")
    print(f"Backup do modelo antigo salvo como {MODEL_PATH}.bak")
    print("\nIMPORTANTE:")
    print("Devido às restrições do algoritmo de Random Forest usado originalmente,")
    print("não é possível 'adicionar' classes mantendo as antigas sem o dataset original.")
    print("Se o arquivo gestures.csv conter APENAS o seu novo gesto, a IA só reconhecerá ele.")
    print("Recomendação: Grave os gestos que você quer que a IA conheça no arquivo CSV.")

if __name__ == "__main__":
    train_new_model()
