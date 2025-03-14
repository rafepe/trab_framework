from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='modelos_login'),
    path('modelos_index', views.modelos_index, name='modelos_index'),
    path('importar/', views.importar_dados, name='importar_dados'),
    path('importar/save/', views.importar_dados_save, name='importar_dados_save'),
    path('exportar/', views.exportar_dados, name='exportar_dados'),
    path('listar/', views.listar_dados, name='listar_dados'),
    path('limpar/', views.limpar_dados, name='limpar_dados'),
    path('treinar/svm/', views.treinar_svm, name='treinar_svm'),
    path('treinar/knn/', views.treinar_knn, name='treinar_knn'),
    path('treinar/rf/', views.treinar_random_forest, name='treinar_rf'),
    path('matriz/<str:modelo>/', views.matriz_confusao, name='matriz_confusao'),
    path('roc/<str:modelo>/', views.curva_roc, name='curva_roc'),
    path('precision-recall/<str:modelo>/', views.precision_recall, name='precision_recall'),
    path('logout/', views.logout_view, name='logout'),
]