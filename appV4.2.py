import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
from datetime import datetime
import io
import base64

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Rodamientos - UTN",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con el mismo estilo de la aplicación de referencia
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-teal: #08596C;
        --secondary-teal: #0A6B82;
        --accent-green: #10b981;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-800: #1f2937;
        --gray-900: #111827;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .header-container {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-teal) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(8, 89, 108, 0.3);
    }
    
    .section-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        border: 2px solid var(--gray-200);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: var(--secondary-teal);
    }
    
    .metric-container {
        background: linear-gradient(135deg, var(--gray-100) 0%, white 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--gray-200);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-teal) 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(8, 89, 108, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(8, 89, 108, 0.4);
    }
    
    .filter-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .step-indicator {
        background: var(--primary-teal);
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .dataframe {
        font-size: 12px;
    }
    
    .big-subtitle {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2.5rem !important;
        padding: 1.5rem !important;
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        border: 2px solid var(--secondary-teal);
        text-align: center;
    }
    
    .calculation-explanation {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de la sesión
if 'seccion_actual' not in st.session_state:
    st.session_state.seccion_actual = 'inicio'

if 'resultados_conicos' not in st.session_state:
    st.session_state.resultados_conicos = pd.DataFrame()

if 'resultados_mixtos' not in st.session_state:
    st.session_state.resultados_mixtos = pd.DataFrame()

if 'condiciones_iniciales' not in st.session_state:
    st.session_state.condiciones_iniciales = {}

# Base de datos unificada de rodamientos
@st.cache_data
def cargar_base_datos_rodamientos():
    # Rodamientos Cónicos
    rodamientos_conicos = [
        # Diámetro 25mm (Empotramiento A)
        {"tipo": "Cónico", "d": 25, "D": 47, "B": 15, "C": 27000, "C0": 32500, "e": 0.43, "Y": 1.4, "Y0": 0.8, "designation": "32005 X/Q"},
        {"tipo": "Cónico", "d": 25, "D": 52, "B": 16.25, "C": 30800, "C0": 33500, "e": 0.37, "Y": 1.6, "Y0": 0.9, "designation": "30205 J2/Q"},
        {"tipo": "Cónico", "d": 25, "D": 52, "B": 19.25, "C": 35800, "C0": 44000, "e": 0.57, "Y": 1.05, "Y0": 0.6, "designation": "32205 BJ2/Q"},
        {"tipo": "Cónico", "d": 25, "D": 52, "B": 22, "C": 54000, "C0": 56000, "e": 0.35, "Y": 1.7, "Y0": 0.9, "designation": "32205/Q"},
        {"tipo": "Cónico", "d": 25, "D": 62, "B": 18.25, "C": 44600, "C0": 43000, "e": 0.3, "Y": 2, "Y0": 1.1, "designation": "30305 J2"},
        {"tipo": "Cónico", "d": 25, "D": 62, "B": 18.25, "C": 38000, "C0": 40000, "e": 0.83, "Y": 0.72, "Y0": 0.4, "designation": "31305 J2"},
        {"tipo": "Cónico", "d": 25, "D": 62, "B": 25.25, "C": 60500, "C0": 63000, "e": 0.3, "Y": 2, "Y0": 1.1, "designation": "32305 J2"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "Cónico", "d": 30, "D": 55, "B": 17, "C": 35800, "C0": 44000, "e": 0.43, "Y": 1.4, "Y0": 0.8, "designation": "32006 X/Q"},
        {"tipo": "Cónico", "d": 30, "D": 62, "B": 17.25, "C": 40200, "C0": 44000, "e": 0.37, "Y": 1.6, "Y0": 0.9, "designation": "30206 J2/Q"},
        {"tipo": "Cónico", "d": 30, "D": 62, "B": 21.25, "C": 50100, "C0": 57000, "e": 0.37, "Y": 1.6, "Y0": 0.9, "designation": "32206 J2/Q"},
        {"tipo": "Cónico", "d": 30, "D": 62, "B": 21.25, "C": 49500, "C0": 58500, "e": 0.57, "Y": 1.05, "Y0": 0.6, "designation": "BJ2/QCL7CVA606"},
        {"tipo": "Cónico", "d": 30, "D": 62, "B": 25, "C": 64400, "C0": 76500, "e": 0.35, "Y": 1.7, "Y0": 0.9, "designation": "33206/Q"},
        {"tipo": "Cónico", "d": 30, "D": 72, "B": 20.75, "C": 56100, "C0": 56000, "e": 0.31, "Y": 1.9, "Y0": 1.1, "designation": "30306 J2/Q"},
        {"tipo": "Cónico", "d": 30, "D": 72, "B": 20.75, "C": 47300, "C0": 50000, "e": 0.83, "Y": 0.72, "Y0": 0.4, "designation": "31306 J2/Q"},
        {"tipo": "Cónico", "d": 30, "D": 72, "B": 28.75, "C": 76500, "C0": 85000, "e": 0.31, "Y": 1.9, "Y0": 1.1, "designation": "32306 J2/Q"},
    ]
    
    # Rodamientos NU
    rodamientos_nu = [
        # Diámetro 25mm (Empotramiento A)
        {"tipo": "NU", "d": 25, "D": 47, "B": 12, "C": 14200, "C0": 13200, "serie": 10, "designation": "NU 1005"},
        {"tipo": "NU", "d": 25, "D": 52, "B": 15, "C": 28600, "C0": 27000, "serie": 2, "designation": "NU 205 ECP"},
        {"tipo": "NU", "d": 25, "D": 52, "B": 18, "C": 34100, "C0": 34000, "serie": 22, "designation": "NU 2205 ECP"},
        {"tipo": "NU", "d": 25, "D": 62, "B": 17, "C": 46500, "C0": 36500, "serie": 3, "designation": "NU 305 ECP"},
        {"tipo": "NU", "d": 25, "D": 62, "B": 17, "C": 64000, "C0": 55000, "serie": 23, "designation": "NU 2305 ECP"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "NU", "d": 30, "D": 55, "B": 13, "C": 17900, "C0": 17300, "serie": 10, "designation": "NU 1006"},
        {"tipo": "NU", "d": 30, "D": 62, "B": 16, "C": 44000, "C0": 36500, "serie": 2, "designation": "NU 206 ECP"},
        {"tipo": "NU", "d": 30, "D": 62, "B": 20, "C": 55000, "C0": 49000, "serie": 22, "designation": "NU 2206 ECP"},
        {"tipo": "NU", "d": 30, "D": 72, "B": 19, "C": 58500, "C0": 48000, "serie": 3, "designation": "NU 306 ECP"},
        {"tipo": "NU", "d": 30, "D": 72, "B": 27, "C": 83000, "C0": 75000, "serie": 23, "designation": "NU 2306"},
        {"tipo": "NU", "d": 30, "D": 90, "B": 23, "C": 60500, "C0": 53000, "serie": 4, "designation": "NU 406"},
    ]
    
    # Rodamientos NUP
    rodamientos_nup = [
        # Diámetro 25mm (Empotramiento A)
        {"tipo": "NUP", "d": 25, "D": 52, "B": 15, "C": 28600, "C0": 27000, "serie": 2, "designation": "NUP 205 ECP"},
        {"tipo": "NUP", "d": 25, "D": 52, "B": 18, "C": 34100, "C0": 34000, "serie": 22, "designation": "NUP 2205 ECP"},
        {"tipo": "NUP", "d": 25, "D": 62, "B": 17, "C": 46500, "C0": 36500, "serie": 3, "designation": "NUP 305 ECP"},
        {"tipo": "NUP", "d": 25, "D": 62, "B": 24, "C": 64000, "C0": 55000, "serie": 23, "designation": "NUP 2305 ECP"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "NUP", "d": 30, "D": 62, "B": 16, "C": 44000, "C0": 36500, "serie": 2, "designation": "NUP 206 ECP"},
        {"tipo": "NUP", "d": 30, "D": 62, "B": 20, "C": 55000, "C0": 49000, "serie": 22, "designation": "NUP 2206 ECP"},
        {"tipo": "NUP", "d": 30, "D": 72, "B": 19, "C": 58500, "C0": 48000, "serie": 3, "designation": "NUP 306 ECP"},
        {"tipo": "NUP", "d": 30, "D": 72, "B": 27, "C": 83000, "C0": 75000, "serie": 23, "designation": "NUP 2306 ECP"},
    ]
    
    # Rodamientos de Bolas (agregados según el informe)
    rodamientos_bolas = [
        # Diámetro 25mm (Empotramiento A)
        {"tipo": "Bolas", "d": 25, "D": 47, "B": 12, "C": 14000, "C0": 6550, "f0": 14, "designation": "98205"},
        {"tipo": "Bolas", "d": 25, "D": 52, "B": 15, "C": 19500, "C0": 11400, "f0": 14, "designation": "6205"},
        {"tipo": "Bolas", "d": 25, "D": 62, "B": 17, "C": 35800, "C0": 19300, "f0": 12, "designation": "6405"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "Bolas", "d": 30, "D": 55, "B": 13, "C": 17800, "C0": 9560, "f0": 14, "designation": "6006"},
        {"tipo": "Bolas", "d": 30, "D": 62, "B": 16, "C": 28100, "C0": 15600, "f0": 14, "designation": "6206"},
        {"tipo": "Bolas", "d": 30, "D": 72, "B": 19, "C": 43300, "C0": 26300, "f0": 12, "designation": "6406"},
    ]
    
    # Combinar todas las bases de datos
    todos_rodamientos = rodamientos_conicos + rodamientos_nu + rodamientos_nup + rodamientos_bolas
    
    # Crear DataFrame
    df = pd.DataFrame(todos_rodamientos)
    df['ID'] = range(1, len(df) + 1)
    
    # Rellenar NaNs en columnas 'serie' o 'f0' (para rodamientos que no las usan)
    df['serie'] = df['serie'].fillna(0)
    df['f0'] = df['f0'].fillna(0)
    
    return df

def interpolar_bolas(relacion_fa_c0):
    """
    Interpola los valores de e, X, Y para rodamientos de bolas
    basado en la tabla del informe.
    """
    # Tabla de interpolación del informe
    relaciones = [0.172, 0.345, 0.689, 1.03, 1.38, 2.07, 3.45, 5.17, 6.89]
    e_vals = [0.19, 0.22, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.44]
    X_vals = [0.56] * 9  # X es constante
    Y_vals = [2.30, 1.99, 1.71, 1.55, 1.45, 1.31, 1.15, 1.04, 1.00]

    # Usar numpy.interp para encontrar los valores
    # np.interp(x, xp, fp)
    e = np.interp(relacion_fa_c0, relaciones, e_vals)
    X = 0.56  # Es constante
    Y = np.interp(relacion_fa_c0, relaciones, Y_vals)
    
    return e, X, Y

def calcular_vida_fs_comun(A, B, PA, PB, P0A, P0B, p_A, p_B, condiciones, FaA_calc, FaB_calc, apoyo_fijo_caso):
    """Función auxiliar para cálculos comunes de vida y FS"""
    FrA = condiciones['FrA']
    FrB = condiciones['FrB'] 
    n_rpm = condiciones['n_rpm']

    CA = A['C']
    CB = B['C']
    C0A = A['C0']
    C0B = B['C0']
    
    FSA = C0A / P0A if P0A > 0 else np.inf
    FSB = C0B / P0B if P0B > 0 else np.inf
    
    L10hA = (1_000_000 / (60 * n_rpm)) * (CA / PA) ** p_A if PA > 0 else np.inf
    L10hB = (1_000_000 / (60 * n_rpm)) * (CB / PB) ** p_B if PB > 0 else np.inf
    
    return {
        "ID_A": A['ID'], "Tipo_A": A['tipo'], "Designación_A": A['designation'],
        "d_A": A['d'], "D_A": A['D'], "B_A": A['B'],
        "ID_B": B['ID'], "Tipo_B": B['tipo'], "Designación_B": B['designation'],
        "d_B": B['d'], "D_B": B['D'], "B_B": B['B'],
        "Apoyo_Fijo": apoyo_fijo_caso,
        "FrA (N)": FrA, "FaA (N)": FaA_calc, "FrB (N)": FrB, "FaB (N)": FaB_calc,
        "PA (N)": round(PA, 2), "PB (N)": round(PB, 2),
        "P0A (N)": round(P0A, 2), "P0B (N)": round(P0B, 2),
        "C_A (N)": CA, "C_B (N)": CB, "C0_A (N)": C0A, "C0_B (N)": C0B,
        "FS_A": round(FSA, 2), "FS_B": round(FSB, 2),
        "p_A": p_A, "p_B": p_B,
        "Vida_A (h)": round(L10hA, 1), "Vida_B (h)": round(L10hB, 1),
        "C/P_A": round(CA / PA, 2) if PA > 0 else np.inf,
        "C/P_B": round(CB / PB, 2) if PB > 0 else np.inf,
        "Vida_Min (h)": round(min(L10hA, L10hB), 1)
    }

def calcular_combinaciones_conicos(df_A, df_B, condiciones):
    """Calcula combinaciones Cónico-Cónico (back-to-back)"""
    FrA = condiciones['FrA']
    FrB = condiciones['FrB'] 
    Faext = condiciones['Fa_ext_conicos'] # Usar la carga axial de cónicos
    
    resultados = []
    p = 10/3
    
    for _, A in df_A.iterrows():
        for _, B in df_B.iterrows():
            YA = A['Y']
            YB = B['Y']
            FiA = 0.5 * FrA / YA
            FiB = 0.5 * FrB / YB
            
            # Condición del informe (back-to-back)
            if FiA <= (FiB + Faext):  # CASO 1
                PA = 0.4 * FrA + YA * (FiB + Faext)
                PB = FrB
            else:  # CASO 2
                PA = FrA
                PB = 0.4 * FrB + YB * (FiA - Faext)
            
            P0A = 0.5 * FrA + A['Y0'] * (FiB + Faext)
            P0B = 0.5 * FrB + B['Y0'] * (FiA - Faext)
            
            # Para cónicos, las cargas axiales "calculadas" son las inducidas + externas
            # Esto es solo para mostrar en la tabla, el cálculo de P ya está hecho.
            FaA_calc = YA * (FiB + Faext) if FiA <= (FiB + Faext) else 0
            FaB_calc = YB * (FiA - Faext) if FiA > (FiB + Faext) else 0

            resultado = calcular_vida_fs_comun(A, B, PA, PB, P0A, P0B, p, p, condiciones, FaA_calc, FaB_calc, apoyo_fijo_caso='Cónico')
            resultados.append(resultado)
            
    return pd.DataFrame(resultados)

def calcular_combinaciones_mixtas(df_A, df_B, condiciones):
    """Calcula combinaciones mixtas (Bolas, NU, NUP) aplicando reglas de carga axial"""
    FrA = condiciones['FrA']
    FrB = condiciones['FrB'] 
    Fa_ext_mixtos = condiciones['Fa_ext_mixtos']
    rodamiento_fijo = condiciones['rodamiento_fijo'] # 'A' o 'B'
    
    # Asignar cargas axiales según el rodamiento fijo
    hay_carga_axial_total = Fa_ext_mixtos > 0
    
    FaA_calc = 0.0
    FaB_calc = 0.0
    
    if rodamiento_fijo == 'A':
        FaA_calc = Fa_ext_mixtos
    else:
        FaB_calc = Fa_ext_mixtos
    
    resultados = []
    
    for _, A in df_A.iterrows():
        p_A = 3.0 if A['tipo'] == 'Bolas' else 10/3
        
        for _, B in df_B.iterrows():
            p_B = 3.0 if B['tipo'] == 'Bolas' else 10/3
            
            # --- APLICAR RESTRICCIÓN DE DISEÑO ---
            # Si hay carga axial total, la combinación NU+NU no es válida
            if hay_carga_axial_total and A['tipo'] == 'NU' and B['tipo'] == 'NU':
                continue # Saltar esta combinación
            
            # --- CÁLCULO DE CARGAS PARA A ---
            PA = 0.0
            P0A = 0.0
            
            if A['tipo'] == 'Bolas':
                # Implementación de Interpolación
                relacion = 0
                if A['C0'] > 0 and A['f0'] > 0:
                    relacion = (A['f0'] * FaA_calc) / A['C0']
                
                e_A, X_A, Y_A = interpolar_bolas(relacion)
                
                if (FrA == 0) or (FaA_calc == 0) or (FaA_calc / FrA <= e_A):
                    PA = FrA
                else:
                    PA = X_A * FrA + Y_A * FaA_calc
                P0A = 0.6 * FrA + 0.5 * FaA_calc
            
            elif A['tipo'] == 'NU':
                PA = FrA
                P0A = FrA
                
            elif A['tipo'] == 'NUP':
                # Implementación de lógica de Serie
                serie_A = A['serie']
                if serie_A in [10, 2, 3, 4]:
                    e_A = 0.2
                    Y_A = 0.6
                else:
                    e_A = 0.3
                    Y_A = 0.4
                
                if (FrA == 0) or (FaA_calc == 0) or (FaA_calc / FrA <= e_A):
                    PA = FrA
                else:
                    PA = 0.92 * FrA + Y_A * FaA_calc
                P0A = FrA

            # --- CÁLCULO DE CARGAS PARA B ---
            PB = 0.0
            P0B = 0.0
            
            if B['tipo'] == 'Bolas':
                relacion = 0
                if B['C0'] > 0 and B['f0'] > 0:
                    relacion = (B['f0'] * FaB_calc) / B['C0']
                
                e_B, X_B, Y_B = interpolar_bolas(relacion)
                
                if (FrB == 0) or (FaB_calc == 0) or (FaB_calc / FrB <= e_B):
                    PB = FrB
                else:
                    PB = X_B * FrB + Y_B * FaB_calc
                P0B = 0.6 * FrB + 0.5 * FaB_calc
            
            elif B['tipo'] == 'NU':
                PB = FrB
                P0B = FrB
                
            elif B['tipo'] == 'NUP':
                serie_B = B['serie']
                if serie_B in [10, 2, 3, 4]:
                    e_B = 0.2
                    Y_B = 0.6
                else:
                    e_B = 0.3
                    Y_B = 0.4

                if (FrB == 0) or (FaB_calc == 0) or (FaB_calc / FrB <= e_B):
                    PB = FrB
                else:
                    PB = 0.92 * FrB + Y_B * FaB_calc
                P0B = FrB
            
            # --- GUARDAR RESULTADO ---
            resultado = calcular_vida_fs_comun(A, B, PA, PB, P0A, P0B, p_A, p_B, condiciones, FaA_calc, FaB_calc, apoyo_fijo_caso=rodamiento_fijo)
            resultados.append(resultado)
            
    return pd.DataFrame(resultados)


def crear_grafico_vida_util(df_resultados, fila_seleccionada):
    """Crea gráfico de vida útil para la combinación seleccionada"""
    
    if fila_seleccionada is None or df_resultados.empty:
        return None
    
    row = df_resultados.iloc[fila_seleccionada]
    
    # Crear curva teórica usando los exponentes específicos
    C_P_max = max(10, row.get('C/P_A', 10), row.get('C/P_B', 10)) + 10
    C_P = np.linspace(1, C_P_max, 500)
    n_rpm = st.session_state.condiciones_iniciales.get('n_rpm', 1400)
    p_A = row['p_A']
    p_B = row['p_B']
    
    L10h_curve_A = (1_000_000 / (60 * n_rpm)) * (C_P) ** p_A
    L10h_curve_B = (1_000_000 / (60 * n_rpm)) * (C_P) ** p_B
    
    fig = go.Figure()
    
    # Curvas teóricas
    fig.add_trace(go.Scatter(
        x=C_P, 
        y=L10h_curve_A,
        mode='lines',
        name=f'Duración teórica A (p={p_A})',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    if p_A != p_B:
        fig.add_trace(go.Scatter(
            x=C_P, 
            y=L10h_curve_B,
            mode='lines',
            name=f'Duración teórica B (p={p_B})',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Puntos de los rodamientos
    fig.add_trace(go.Scatter(
        x=[row['C/P_A']], 
        y=[row['Vida_A (h)']],
        mode='markers',
        name=f"Rodamiento A ({row['Designación_A']})",
        marker=dict(color='blue', size=12, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=[row['C/P_B']], 
        y=[row['Vida_B (h)']],
        mode='markers',
        name=f"Rodamiento B ({row['Designación_B']})",
        marker=dict(color='red', size=12, symbol='square')
    ))
    
    # Líneas verticales
    fig.add_vline(x=row['C/P_A'], line_dash="dot", line_color="blue", opacity=0.7)
    fig.add_vline(x=row['C/P_B'], line_dash="dot", line_color="red", opacity=0.7)
    
    fig.update_layout(
        title="Duración del rodamiento para la combinación seleccionada",
        xaxis_title="Relación C/P",
        yaxis_title="Duración en horas (L10h)",
        showlegend=True,
        template="plotly_white",
        height=500
    )
    
    return fig

def mostrar_recomendaciones(df_resultados):
    """Muestra las 3 mejores combinaciones según el criterio de vidas útiles similares"""
    st.markdown("### 🏆 Combinaciones Recomendadas (Vidas Útiles Similares)")
    st.info("Top 3 combinaciones donde la vida útil de A y B son más parecidas, según el criterio del informe.")
    
    df_reco = df_resultados.copy()
    
    # Limpiar vidas infinitas o nulas antes de calcular la relación
    df_reco['Vida_A (h)'] = df_reco['Vida_A (h)'].replace([np.inf, -np.inf], np.nan)
    df_reco['Vida_B (h)'] = df_reco['Vida_B (h)'].replace([np.inf, -np.inf], np.nan)
    df_reco = df_reco.dropna(subset=['Vida_A (h)', 'Vida_B (h)'])
    df_reco = df_reco[df_reco['Vida_A (h)'] > 0] # Evitar división por cero

    if not df_reco.empty:
        df_reco["Relación_Vida (B/A)"] = df_reco["Vida_B (h)"] / df_reco["Vida_A (h)"]
        df_reco["Diferencia_Vida"] = (df_reco["Relación_Vida (B/A)"] - 1).abs()
        df_reco = df_reco.sort_values(by="Diferencia_Vida", ascending=True)
        
        # Formatear "Relación_Vida (B/A)" para mostrarla
        df_reco["Relación_Vida (B/A)"] = df_reco["Relación_Vida (B/A)"].round(3)
        
        cols_display = [
            "Apoyo_Fijo", "Designación_A", "Designación_B", 
            "Vida_A (h)", "Vida_B (h)", 
            "Relación_Vida (B/A)", "FS_A", "FS_B"
        ]
        
        # Asegurarse de que las columnas existan antes de mostrarlas
        cols_to_show = [col for col in cols_display if col in df_reco.columns]
        
        st.dataframe(df_reco.head(3)[cols_to_show], use_container_width=True, height=150)
    else:
        st.warning("No se pudieron calcular recomendaciones (posiblemente vidas útiles nulas o infinitas).")
    
    st.divider()
    st.markdown("###  Explorar todas las combinaciones")

# Sidebar
with st.sidebar:
    st.markdown("### 🧭 Navegación")
    
    if st.button("🏠 Inicio", use_container_width=True):
        st.session_state.seccion_actual = 'inicio'
        st.rerun()
    
    if st.button("⚙️ Análisis de Rodamientos", use_container_width=True):
        st.session_state.seccion_actual = 'analisis'
        st.rerun()
    
    st.divider()
    
    st.markdown("### ℹ️ Información")
    st.markdown(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}")
    st.markdown(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")
    
    total_combinaciones = len(st.session_state.resultados_conicos) + \
                          len(st.session_state.resultados_mixtos)
    
    if total_combinaciones > 0:
        st.markdown(f"**Combinaciones calculadas:** {total_combinaciones}")

# Contenido principal
if st.session_state.seccion_actual == 'inicio':
    # Página de inicio
    st.markdown("""
    <div class="header-container">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            Análisis de Rodamientos
        </h1>
        <h2 style="font-size: 1.8rem; margin-bottom: 0; opacity: 0.9;">
            Sistema Inteligente de Selección
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección principal
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h3 style="color: #08596C; font-size: 2.5rem; margin-bottom: 1.5rem; text-align: center;">
                ⚙️ ANÁLISIS SIMULTÁNEO
            </h3>
            <h4 class="big-subtitle">
                Sistema Automático de Cálculo de Selección de Rodamientos
            </h4>
            <p style="color: #4b5563; line-height: 1.8; margin-bottom: 2rem; text-align: center; font-size: 1.1rem;">
                Análisis completo de combinaciones de rodamientos para empotramientos específicos.<br><br>
                • <strong>Configuración flexible:</strong> Diámetros de empotramiento personalizables<br>
                • <strong>Tipos soportados:</strong> Cónicos, NU, NUP, Bolas<br>
                • <strong>Cálculos precisos:</strong> Basados en normas SKF y fórmulas del informe técnico<br>
                • <strong>Análisis completo:</strong> Todas las combinaciones posibles<br>
                • <strong>Visualización:</strong> Gráficos de vida útil interactivos<br>
                • <strong>Exportación:</strong> Resultados en Excel
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ACCEDER AL ANÁLISIS", key="analisis_btn", type="primary", use_container_width=True):
            st.session_state.seccion_actual = 'analisis'
            st.rerun()

elif st.session_state.seccion_actual == 'analisis':
    st.markdown("# ⚙️ Análisis Simultáneo de Rodamientos")
    st.markdown("Análisis completo de combinaciones entre rodamientos según especificaciones técnicas")
    
    # Cargar base de datos
    df_rodamientos = cargar_base_datos_rodamientos()
    
    # Paso 1: Mostrar base de datos
    st.markdown("## 📊 Paso 1: Base de Datos de Rodamientos")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Base de Datos Unificada")
        st.info("🔧 **Base de datos completa con todos los tipos de rodamientos disponibles**")
        
        tipos_disponibles = df_rodamientos['tipo'].unique()
        tipo_filtro = st.multiselect("Filtrar por tipo de rodamiento:", tipos_disponibles, default=tipos_disponibles)
        
        diametros_disponibles = sorted(df_rodamientos['d'].unique())
        diametro_filtro = st.multiselect("Filtrar por diámetro (mm):", diametros_disponibles, default=diametros_disponibles)
        
        df_filtrado = df_rodamientos[
            (df_rodamientos['tipo'].isin(tipo_filtro)) & 
            (df_rodamientos['d'].isin(diametro_filtro))
        ]
        
        st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### Estadísticas")
        st.metric("Total de rodamientos", len(df_rodamientos))
        
        # Distribución por tipo
        tipo_counts = df_rodamientos['tipo'].value_counts()
        for tipo, count in tipo_counts.items():
            st.metric(f"Rodamientos {tipo}", count)
        
        # Gráfico de distribución
        fig_pie = px.pie(values=tipo_counts.values, names=tipo_counts.index, title="Distribución por Tipo")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("### 📁 Cargar Base de Datos")
        uploaded_file = st.file_uploader("Cargar Excel personalizado", type=['xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                df_custom = pd.read_excel(uploaded_file)
                st.success("✅ Archivo cargado correctamente")
                st.dataframe(df_custom.head(), use_container_width=True)

                # --- Validar columnas esperadas ---
                columnas_necesarias = {"tipo", "d", "D", "B", "C", "C0", "designation"}
                if not columnas_necesarias.issubset(df_custom.columns):
                    st.error(f"❌ El archivo debe incluir al menos las columnas: {', '.join(columnas_necesarias)}")
                else:
                    # --- Rellenar columnas faltantes con NaN o valores por defecto ---
                    for col in ["e", "Y", "Y0", "serie", "f0"]:
                        if col not in df_custom.columns:
                            df_custom[col] = np.nan

                    # --- Asegurar tipos numéricos ---
                    for col in ["d", "D", "B", "C", "C0", "e", "Y", "Y0", "serie", "f0"]:
                        df_custom[col] = pd.to_numeric(df_custom[col], errors="coerce")

                    # --- Asignar IDs nuevos y concatenar ---
                    max_id = df_rodamientos["ID"].max() if "ID" in df_rodamientos.columns else 0
                    df_custom["ID"] = range(max_id + 1, max_id + 1 + len(df_custom))

                    # --- Rellenar NaN por 0 donde corresponde ---
                    df_custom["serie"] = df_custom["serie"].fillna(0)
                    df_custom["f0"] = df_custom["f0"].fillna(0)

                    # --- Combinar con la base existente ---
                    df_rodamientos = pd.concat([df_rodamientos, df_custom], ignore_index=True)

                    st.success(f"✅ Se agregaron {len(df_custom)} nuevos rodamientos a la base existente.")
                    st.dataframe(df_rodamientos.tail(len(df_custom)), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {e}")
    
    st.divider()
    
    st.markdown("## 📚 Explicación de Cálculos por Tipo de Rodamiento")
    
    st.markdown("""
    <div class="calculation-explanation">
        <p><strong>Rodamientos de Bolas:</strong></p>
        <ul>
            <li>Exponente de vida: p = 3</li>
            <li>Valores 'e', 'X', 'Y' se interpolan según (f₀·Fa)/C₀</li>
            <li>Carga equivalente: P = Fr (si Fa/Fr ≤ e) o P = X·Fr + Y·Fa (si Fa/Fr > e)</li>
            <li>Vida útil: L₁₀ₕ = (1,000,000)/(60·n) × (C/P)³</li>
        </ul>
        <p><strong>Rodamientos Cónicos (Back-to-Back):</strong></p>
        <ul>
            <li>Exponente de vida: p = 10/3</li>
            <li>Cálculo de fuerzas axiales inducidas: Fi = 0.5·Fr / Y</li>
            [cite_start]<li>Cálculo de P según la condición de cargas axiales (2 casos) [cite: 187-193]</li>
            <li>Vida útil: L₁₀ₕ = (1,000,000)/(60·n) × (C/P)^(10/3)</li>
        </ul>
        </p>  
        <p><strong>Rodamientos NU:</strong></p>
        <ul>
            <li>Exponente de vida: p = 10/3</li>
            <li>Solo carga radial: P = Fr</li>
            <li>Vida útil: L₁₀ₕ = (1,000,000)/(60·n) × (C/P)^(10/3)</li>
        </ul>   
        <p><strong>Rodamientos NUP:</strong></p>
        <ul>
            <li>Exponente de vida: p = 10/3</li>
            [cite_start]<li>Valores 'e' e 'Y' dependen de la serie del rodamiento [cite: 233-238]</li>
            <li>P = Fr (si Fa/Fr ≤ e) o P = 0.92·Fr + Y·Fa (si Fa/Fr > e)</li>
            <li>Vida útil: L₁₀ₕ = (1,000,000)/(60·n) × (C/P)^(10/3)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Paso 2: Condiciones iniciales
    st.markdown("## ⚙️ Paso 2: Condiciones de Contorno del Problema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 Configuración de Empotramientos")
        diametro_A = st.number_input("Diámetro Empotramiento A (mm):", value=25, min_value=10, max_value=100, step=5)
        diametro_B = st.number_input("Diámetro Empotramiento B (mm):", value=30, min_value=10, max_value=100, step=5)
        
        st.markdown("### ⚡ Condiciones de Operación")
        n_rpm = st.number_input("Velocidad (RPM):", value=1400, min_value=1)
        
    with col2:
        st.markdown("### 💪 Cargas Radiales")
        FrA = st.number_input("Fuerza radial en A (N):", value=1400, min_value=0)
        FrB = st.number_input("Fuerza radial en B (N):", value=2300, min_value=0)
        
        st.markdown("### 💪 Carga Axial (Cónicos)")
        Fa_ext_conicos = st.number_input("Fuerza axial externa (Cónicos) (N):", value=350.0, min_value=0.0)

    with col3:
        st.markdown("### 💪 Carga Axial (Bolas/NU/NUP)")
        Fa_ext_mixtos = st.number_input("Fuerza axial (Bolas/NU/NUP) (N):", value=350.0, min_value=0.0)
        
        st.info("""
        **Nota de Diseño (Bolas/NU/NUP):**
        Para cargas axiales, la aplicación sigue el principio de **"rodamiento fijo y libre"**. 
        Calculará automáticamente dos escenarios:
        1.  **Fijo en A:** A soporta el 100% de la carga axial.
        2.  **Fijo en B:** B soporta el 100% de la carga axial.
        
        Ambos resultados aparecerán en la tabla del "Paso 3".
        """)
        
        st.markdown("### 🚀 Acciones")
        if st.button("🔄 Calcular Combinaciones", type="primary", use_container_width=True):
            rodamientos_disponibles_A = df_rodamientos[df_rodamientos['d'] == diametro_A]
            rodamientos_disponibles_B = df_rodamientos[df_rodamientos['d'] == diametro_B]
            
            if len(rodamientos_disponibles_A) == 0:
                st.error(f"❌ No hay rodamientos disponibles para diámetro {diametro_A}mm")
            elif len(rodamientos_disponibles_B) == 0:
                st.error(f"❌ No hay rodamientos disponibles para diámetro {diametro_B}mm")
            else:
                condiciones_base = {
                    'FrA': FrA, 'FrB': FrB, 
                    'Fa_ext_conicos': Fa_ext_conicos,
                    'Fa_ext_mixtos': Fa_ext_mixtos,
                    'n_rpm': n_rpm,
                    'diametro_A': diametro_A, 'diametro_B': diametro_B
                }
                
                with st.spinner("Calculando todas las combinaciones..."):
                    # --- Cálculo Cónicos ---
                    df_A_conico = rodamientos_disponibles_A[rodamientos_disponibles_A['tipo'] == 'Cónico']
                    df_B_conico = rodamientos_disponibles_B[rodamientos_disponibles_B['tipo'] == 'Cónico']
                    st.session_state.resultados_conicos = calcular_combinaciones_conicos(df_A_conico, df_B_conico, condiciones_base)
                    
                    # --- Cálculo Mixtas (Bolas, NU, NUP) ---
                    df_A_mixtos = rodamientos_disponibles_A[rodamientos_disponibles_A['tipo'].isin(['Bolas', 'NU', 'NUP'])]
                    df_B_mixtos = rodamientos_disponibles_B[rodamientos_disponibles_B['tipo'].isin(['Bolas', 'NU', 'NUP'])]
                    
                    # Caso 1: Fijo en A
                    condiciones_A = {**condiciones_base, 'rodamiento_fijo': 'A'}
                    resultados_A = calcular_combinaciones_mixtas(df_A_mixtos, df_B_mixtos, condiciones_A)
                    
                    # Caso 2: Fijo en B
                    condiciones_B = {**condiciones_base, 'rodamiento_fijo': 'B'}
                    resultados_B = calcular_combinaciones_mixtas(df_A_mixtos, df_B_mixtos, condiciones_B)
                    
                    # Unir resultados
                    st.session_state.resultados_mixtos = pd.concat([resultados_A, resultados_B], ignore_index=True)

                total_calc = len(st.session_state.resultados_conicos) + \
                             len(st.session_state.resultados_mixtos)
                
                st.success(f"✅ Se calcularon {total_calc} combinaciones en total")
                # Guardar las condiciones base para referencia
                st.session_state.condiciones_iniciales = condiciones_base
                st.rerun()
        
        # Botón de descarga
        total_combinaciones_descarga = len(st.session_state.resultados_conicos) + \
                                       len(st.session_state.resultados_mixtos)

        if total_combinaciones_descarga > 0:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if not st.session_state.resultados_conicos.empty:
                    st.session_state.resultados_conicos.to_excel(writer, sheet_name='Resultados_Conicos', index=False)
                if not st.session_state.resultados_mixtos.empty:
                    st.session_state.resultados_mixtos.to_excel(writer, sheet_name='Resultados_Mixtos', index=False)
                
                df_rodamientos.to_excel(writer, sheet_name='Base_Datos', index=False)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer.getvalue(),
                file_name=f"analisis_rodamientos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# Paso 3: Resultados
total_combinaciones_global = len(st.session_state.resultados_conicos) + \
                             len(st.session_state.resultados_mixtos)

if total_combinaciones_global > 0:
    st.divider()
    st.markdown("## 📈 Paso 3: Resultados y Análisis")
    
    tab_conico, tab_mixtas = st.tabs([
        "Cónico + Cónico", 
        "Combinaciones Mixtas (Bolas/NU/NUP)"
    ])
    
    # --- Pestaña Cónicos ---
    with tab_conico:
        df_resultados_conico = st.session_state.resultados_conicos
        if df_resultados_conico.empty:
            st.warning("No se encontraron combinaciones Cónico + Cónico para los diámetros seleccionados.")
        else:
            # Mostrar recomendaciones
            mostrar_recomendaciones(df_resultados_conico)
            
            # Filtros para la tabla completa
            col1, col2 = st.columns(2)
            with col1:
                vida_min_conico = st.number_input("Vida mínima (h):", value=0, min_value=0, key="vida_min_conico")
            with col2:
                ordenar_por_conico = st.selectbox("Ordenar por:", 
                                                 ['Vida_Min (h)', 'Vida_A (h)', 'Vida_B (h)', 'FS_A', 'FS_B'],
                                                 key="orden_conico")
            
            df_filtrado_conico = df_resultados_conico.copy()
            if vida_min_conico > 0:
                df_filtrado_conico = df_filtrado_conico[df_filtrado_conico['Vida_Min (h)'] >= vida_min_conico]
            
            df_filtrado_conico = df_filtrado_conico.sort_values(by=ordenar_por_conico, ascending=False)
            
            st.markdown(f"**Tabla Completa ({len(df_filtrado_conico)} combinaciones)**")
            event_conico = st.dataframe(
                df_filtrado_conico, use_container_width=True, height=400,
                on_select="rerun", selection_mode="single-row", key="df_conico"
            )
            
            if event_conico.selection.rows:
                fila_sel = event_conico.selection.rows[0]
                st.markdown("### 📊 Gráfico de Vida Útil - Combinación Seleccionada")
                fig = crear_grafico_vida_util(df_filtrado_conico, fila_sel)
                st.plotly_chart(fig, use_container_width=True)
                
                row_sel = df_filtrado_conico.iloc[fila_sel]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"#### 🔧 Rodamiento A (Empotramiento {row_sel['d_A']}mm)")
                    st.write(f"**Tipo:** {row_sel['Tipo_A']}")
                    st.write(f"**Designación:** {row_sel['Designación_A']}")
                    st.write(f"**Vida útil:** {row_sel['Vida_A (h)']} horas")
                    st.write(f"**Factor de seguridad:** {row_sel['FS_A']}")
                with c2:
                    st.markdown(f"#### 🔧 Rodamiento B (Empotramiento {row_sel['d_B']}mm)")
                    st.write(f"**Tipo:** {row_sel['Tipo_B']}")
                    st.write(f"**Designación:** {row_sel['Designación_B']}")
                    st.write(f"**Vida útil:** {row_sel['Vida_B (h)']} horas")
                    st.write(f"**Factor de seguridad:** {row_sel['FS_B']}")

    # --- Pestaña Mixtas ---
    with tab_mixtas:
        df_resultados_mixtos = st.session_state.resultados_mixtos
        if df_resultados_mixtos.empty:
            st.warning("No se encontraron combinaciones Mixtas (Bolas/NU/NUP) para los diámetros seleccionados.")
        else:
            if (st.session_state.condiciones_iniciales.get('Fa_ext_mixtos', 0) > 0):
                st.info("Nota: Las combinaciones NU+NU fueron excluidas debido a la presencia de carga axial.")
            
            # Mostrar recomendaciones
            mostrar_recomendaciones(df_resultados_mixtos)
            
            # Filtros para la tabla completa
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                apoyo_fijo_filtro = st.selectbox("Apoyo Fijo:", ["Todos"] + list(df_resultados_mixtos['Apoyo_Fijo'].unique()), key="apoyo_fijo_filtro")
            with col2:
                tipo_A_mixto = st.selectbox("Tipo A:", ["Todos"] + list(df_resultados_mixtos['Tipo_A'].unique()), key="tipo_A_mixto")
            with col3:
                tipo_B_mixto = st.selectbox("Tipo B:", ["Todos"] + list(df_resultados_mixtos['Tipo_B'].unique()), key="tipo_B_mixto")
            with col4:
                vida_min_mixto = st.number_input("Vida mínima (h):", value=0, min_value=0, key="vida_min_mixto")
            
            ordenar_por_mixto = st.selectbox("Ordenar por:", 
                                             ['Vida_Min (h)', 'Vida_A (h)', 'Vida_B (h)', 'FS_A', 'FS_B'],
                                             key="orden_mixto")
            
            df_filtrado_mixto = df_resultados_mixtos.copy()
            
            if apoyo_fijo_filtro != "Todos":
                df_filtrado_mixto = df_filtrado_mixto[df_filtrado_mixto['Apoyo_Fijo'] == apoyo_fijo_filtro]
            if tipo_A_mixto != "Todos":
                df_filtrado_mixto = df_filtrado_mixto[df_filtrado_mixto['Tipo_A'] == tipo_A_mixto]
            if tipo_B_mixto != "Todos":
                df_filtrado_mixto = df_filtrado_mixto[df_filtrado_mixto['Tipo_B'] == tipo_B_mixto]
            if vida_min_mixto > 0:
                df_filtrado_mixto = df_filtrado_mixto[df_filtrado_mixto['Vida_Min (h)'] >= vida_min_mixto]
            
            df_filtrado_mixto = df_filtrado_mixto.sort_values(by=ordenar_por_mixto, ascending=False)
            
            st.markdown(f"**Tabla Completa ({len(df_filtrado_mixto)} combinaciones)**")
            event_mixto = st.dataframe(
                df_filtrado_mixto, use_container_width=True, height=400,
                on_select="rerun", selection_mode="single-row", key="df_mixto"
            )
            
            if event_mixto.selection.rows:
                fila_sel = event_mixto.selection.rows[0]
                st.markdown("### 📊 Gráfico de Vida Útil - Combinación Seleccionada")
                fig = crear_grafico_vida_util(df_filtrado_mixto, fila_sel)
                st.plotly_chart(fig, use_container_width=True)
                
                row_sel = df_filtrado_mixto.iloc[fila_sel]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"#### 🔧 Rodamiento A (Empotramiento {row_sel['d_A']}mm)")
                    st.write(f"**Tipo:** {row_sel['Tipo_A']}")
                    st.write(f"**Designación:** {row_sel['Designación_A']}")
                    st.write(f"**Apoyo Fijo (Axial):** {'Sí' if row_sel['Apoyo_Fijo'] == 'A' else 'No'}")
                    st.write(f"**Vida útil:** {row_sel['Vida_A (h)']} horas")
                    st.write(f"**Factor de seguridad:** {row_sel['FS_A']}")
                with c2:
                    st.markdown(f"#### 🔧 Rodamiento B (Empotramiento {row_sel['d_B']}mm)")
                    st.write(f"**Tipo:** {row_sel['Tipo_B']}")
                    st.write(f"**Designación:** {row_sel['Designación_B']}")
                    st.write(f"**Apoyo Fijo (Axial):** {'Sí' if row_sel['Apoyo_Fijo'] == 'B' else 'No'}")
                    st.write(f"**Vida útil:** {row_sel['Vida_B (h)']} horas")

                    st.write(f"**Factor de seguridad:** {row_sel['FS_B']}")
