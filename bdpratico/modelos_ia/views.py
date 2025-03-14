from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import DadosVinho
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score
import plotly.graph_objects as go
import joblib
import csv
import os
from django.conf import settings
from django.contrib.auth import authenticate, login, logout

# Pasta para salvar os modelos
MODELS_DIR = os.path.join(settings.BASE_DIR, 'modelos_ia', 'modelos')
os.makedirs(MODELS_DIR, exist_ok=True)

def index(request):    
    if request.method == 'GET':
        return render(request, 'modelos_ia/login.html')
        
    usuario = request.POST.get('username')
    senha = request.POST.get('password')
    user = authenticate(username=usuario, password=senha)
    if (user is not None):
        login(request, user)
        request.session['username'] = usuario
        request.session['password'] = senha
        request.session['usernamefull'] = user.get_full_name()
        print(request.session['username'])
        print(request.session['password'])
        print(request.session['usernamefull'])
        return redirect('modelos_index')
    else:
        data = {}
        if (usuario):
            data['msg'] = "Usuário ou Senha Incorretos " + usuario
        return render(request, 'modelos_ia/login.html', data)

@login_required
def modelos_index(request):
    """View principal do dashboard"""
    return render(request, 'modelos_ia/index.html')

@login_required
def importar_dados(request):
    """Página para upload de arquivo CSV"""
    return render(request, 'modelos_ia/importar.html')

@login_required
def importar_dados_save(request):
    """Processa o arquivo CSV enviado e salva no banco"""
    if request.method == 'POST' and request.FILES.get('arquivo_csv'):
        arquivo = request.FILES['arquivo_csv']
        
        # Limpa dados existentes
        DadosVinho.objects.all().delete()
        
        # Lê o CSV
        df = pd.read_csv(arquivo)
        
        # Importa dados para o banco
        for _, row in df.iterrows():
            DadosVinho.objects.create(
                fixed_acidity=row['fixed acidity'],
                volatile_acidity=row['volatile acidity'],
                citric_acid=row['citric acid'],
                residual_sugar=row['residual sugar'],
                chlorides=row['chlorides'],
                free_sulfur_dioxide=row['free sulfur dioxide'],
                total_sulfur_dioxide=row['total sulfur dioxide'],
                density=row['density'],
                pH=row['pH'],
                sulphates=row['sulphates'],
                alcohol=row['alcohol'],
                quality=row['quality']
            )
        
        return redirect('listar_dados')
    return redirect('importar_dados')

@login_required
def exportar_dados(request):
    """Exporta dados do banco para CSV"""
    dados = DadosVinho.objects.all()
    
    # Cria arquivo CSV em memória
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="dados_modelos.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])
    
    for dado in dados:
        writer.writerow([
            dado.fixed_acidity,
            dado.volatile_acidity,
            dado.citric_acid,
            dado.residual_sugar,
            dado.chlorides,
            dado.free_sulfur_dioxide,
            dado.total_sulfur_dioxide,
            dado.density,
            dado.pH,
            dado.sulphates,
            dado.alcohol,
            dado.quality
        ])
    
    return response

@login_required
def listar_dados(request):
    """Lista todos os dados do banco"""
    dados = DadosVinho.objects.all()
    return render(request, 'modelos_ia/listar.html', {'dados': dados})

def _preparar_dados():
    """Função auxiliar para preparar os dados para treinamento"""
    dados = DadosVinho.objects.all()
    df = pd.DataFrame(list(dados.values()))
    
    # Todas as colunas exceto 'id' e 'quality' são features
    feature_columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 
                      'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                      'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 
                      'alcohol']
    
    X = df[feature_columns]
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criar e aplicar o scaler apenas para SVM e Random Forest
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def _calcular_metricas(y_true, y_pred):
    """Calcula todas as métricas de avaliação"""
    # Acurácia
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision e Recall (média para multiclasse)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Matriz de confusão para calcular sensibilidade e especificidade
    cm = confusion_matrix(y_true, y_pred)
    
    # Para multiclasse, calculamos a média das sensibilidades e especificidades
    sensibilidades = []
    especificidades = []
    
    for i in range(len(cm)):
        # Verdadeiros positivos para a classe atual
        tp = cm[i][i]
        # Falsos negativos para a classe atual
        fn = sum(cm[i]) - tp
        # Falsos positivos para a classe atual
        fp = sum(cm[:, i]) - tp
        # Verdadeiros negativos para a classe atual
        tn = sum(sum(cm)) - tp - fp - fn
        
        # Sensibilidade (recall) para a classe atual
        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensibilidades.append(sensibilidade)
        
        # Especificidade para a classe atual
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
        especificidades.append(especificidade)
    
    # Calculamos a média das métricas
    sensibilidade_media = sum(sensibilidades) / len(sensibilidades)
    especificidade_media = sum(especificidades) / len(especificidades)
    
    return {
        'acuracia': round(accuracy * 100, 2),
        'sensibilidade': round(sensibilidade_media * 100, 2),
        'especificidade': round(especificidade_media * 100, 2),
        'recall': round(recall * 100, 2),
        'precision': round(precision * 100, 2)
    }

@login_required
def treinar_svm(request):
    """Treina o modelo SVM"""
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = _preparar_dados()
    
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Salva o modelo usando caminho absoluto
    model_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'svm_scaler.pkl')
    
    joblib.dump(svm, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Calcula todas as métricas
    y_pred = svm.predict(X_test_scaled)
    metricas = _calcular_metricas(y_test, y_pred)
    
    return render(request, 'modelos_ia/resultado_treino.html', {
        'modelo': 'SVM',
        **metricas
    })

@login_required
def treinar_knn(request):
    """Treina o modelo KNN"""
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = _preparar_dados()
    
    # Para KNN, vamos usar os dados não escalonados
    X_train = scaler.inverse_transform(X_train_scaled)
    X_test = scaler.inverse_transform(X_test_scaled)
    
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',    # todos os vizinhos têm o mesmo peso
        metric='euclidean',   # métrica de distância explícita
        p=2                   # poder da distância de Minkowski (2 = euclidiana)
    )
    knn.fit(X_train, y_train)
    
    # Salva o modelo usando caminho absoluto
    model_path = os.path.join(MODELS_DIR, 'knn_model.pkl')
    
    joblib.dump(knn, model_path)
    
    # Calcula todas as métricas
    y_pred = knn.predict(X_test)
    metricas = _calcular_metricas(y_test, y_pred)
    
    return render(request, 'modelos_ia/resultado_treino.html', {
        'modelo': 'KNN',
        **metricas
    })

@login_required
def treinar_random_forest(request):
    """Treina o modelo Random Forest"""
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = _preparar_dados()
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Salva o modelo usando caminho absoluto (usando 'rf' em vez de 'random forest')
    model_path = os.path.join(MODELS_DIR, 'rf_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'rf_scaler.pkl')
    
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Calcula todas as métricas
    y_pred = rf.predict(X_test_scaled)
    metricas = _calcular_metricas(y_test, y_pred)
    
    return render(request, 'modelos_ia/resultado_treino.html', {
        'modelo': 'Random Forest',
        **metricas
    })

@login_required
def matriz_confusao(request, modelo):
    """Gera e exibe a matriz de confusão"""
    X_train_scaled, X_test_scaled, y_train, y_test, _ = _preparar_dados()
    
    # Ajusta o nome do modelo para o arquivo
    modelo_arquivo = 'rf' if modelo == 'random forest' else modelo
    
    # Carrega o modelo usando caminho absoluto
    model_path = os.path.join(MODELS_DIR, f'{modelo_arquivo}_model.pkl')
    model = joblib.load(model_path)
    
    # Gera previsões
    y_pred = model.predict(X_test_scaled)
    
    # Calcula matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Prepara dados para o template
    labels = sorted(list(set(y_test)))
    
    return render(request, 'modelos_ia/matriz_confusao.html', {
        'matriz': cm.tolist(),
        'labels': labels,
        'modelo': modelo.upper()
    })

@login_required
def curva_roc(request, modelo):
    """Gera e exibe a curva ROC para classificação multiclasse"""
    X_train_scaled, X_test_scaled, y_train, y_test, _ = _preparar_dados()
    
    # Ajusta o nome do modelo para o arquivo
    modelo_arquivo = 'rf' if modelo == 'random forest' else modelo
    
    # Carrega o modelo
    model_path = os.path.join(MODELS_DIR, f'{modelo_arquivo}_model.pkl')
    model = joblib.load(model_path)
    
    # Gera probabilidades para cada classe
    y_pred_prob = model.predict_proba(X_test_scaled)
    
    # Cria gráfico com plotly
    fig = go.Figure()
    
    # Calcula curva ROC para cada classe (one-vs-rest)
    classes = sorted(list(set(y_test)))
    for i, classe in enumerate(classes):
        # Converte para classificação binária (one-vs-rest)
        y_test_bin = (y_test == classe).astype(int)
        y_score = y_pred_prob[:, i]
        
        # Calcula curva ROC
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Adiciona curva ao gráfico
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Classe {classe} (AUC = {roc_auc:.2f})',
            line=dict(width=2)
        ))
    
    # Adiciona linha de referência
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=f'Curvas ROC (One-vs-Rest) - {modelo.upper()}',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        showlegend=True
    )
    
    graph = fig.to_html(full_html=False)
    
    return render(request, 'modelos_ia/curva_roc.html', {
        'graph': graph,
        'modelo': modelo.upper()
    })

@login_required
def precision_recall(request, modelo):
    """Gera e exibe a curva Precision-Recall para classificação multiclasse"""
    X_train_scaled, X_test_scaled, y_train, y_test, _ = _preparar_dados()
    
    # Ajusta o nome do modelo para o arquivo
    modelo_arquivo = 'rf' if modelo == 'random forest' else modelo
    
    # Carrega o modelo
    model_path = os.path.join(MODELS_DIR, f'{modelo_arquivo}_model.pkl')
    model = joblib.load(model_path)
    
    # Gera probabilidades para cada classe
    y_pred_prob = model.predict_proba(X_test_scaled)
    
    # Cria gráfico com plotly
    fig = go.Figure()
    
    # Calcula curva Precision-Recall para cada classe (one-vs-rest)
    classes = sorted(list(set(y_test)))
    for i, classe in enumerate(classes):
        # Converte para classificação binária (one-vs-rest)
        y_test_bin = (y_test == classe).astype(int)
        y_score = y_pred_prob[:, i]
        
        # Calcula curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test_bin, y_score)
        pr_auc = auc(recall, precision)
        
        # Adiciona curva ao gráfico
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'Classe {classe} (AUC = {pr_auc:.2f})',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f'Curvas Precision-Recall (One-vs-Rest) - {modelo.upper()}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    
    graph = fig.to_html(full_html=False)
    
    return render(request, 'modelos_ia/precision_recall.html', {
        'graph': graph,
        'modelo': modelo.upper()
    })

@login_required
def limpar_dados(request):
    """Limpa todos os dados do banco"""
    DadosVinho.objects.all().delete()
    return redirect('modelos_index')

def logout_view(request):
    """View para fazer logout do usuário"""
    logout(request)
    return redirect('modelos_login')


