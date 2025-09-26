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

if 'resultados_calculados' not in st.session_state:
    st.session_state.resultados_calculados = pd.DataFrame()

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
        {"tipo": "NU", "d": 25, "D": 47, "B": 12, "C": 14200, "C0": 13200, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 1005"},
        {"tipo": "NU", "d": 25, "D": 52, "B": 15, "C": 28600, "C0": 27000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 205 ECP"},
        {"tipo": "NU", "d": 25, "D": 52, "B": 18, "C": 34100, "C0": 34000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 2205 ECP"},
        {"tipo": "NU", "d": 25, "D": 62, "B": 17, "C": 46500, "C0": 36500, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 305 ECP"},
        {"tipo": "NU", "d": 25, "D": 62, "B": 17, "C": 64000, "C0": 55000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 2305 ECP"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "NU", "d": 30, "D": 55, "B": 13, "C": 17900, "C0": 17300, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 1006"},
        {"tipo": "NU", "d": 30, "D": 62, "B": 16, "C": 44000, "C0": 36500, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 206 ECP"},
        {"tipo": "NU", "d": 30, "D": 62, "B": 20, "C": 55000, "C0": 49000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 2206 ECP"},
        {"tipo": "NU", "d": 30, "D": 72, "B": 19, "C": 58500, "C0": 48000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 306 ECP"},
        {"tipo": "NU", "d": 30, "D": 72, "B": 27, "C": 83000, "C0": 75000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 2306"},
        {"tipo": "NU", "d": 30, "D": 90, "B": 23, "C": 60500, "C0": 53000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NU 406"},
    ]
    
    # Rodamientos NUP
    rodamientos_nup = [
        # Diámetro 25mm (Empotramiento A)
        {"tipo": "NUP", "d": 25, "D": 52, "B": 15, "C": 28600, "C0": 27000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 205 ECP"},
        {"tipo": "NUP", "d": 25, "D": 52, "B": 18, "C": 34100, "C0": 34000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 2205 ECP"},
        {"tipo": "NUP", "d": 25, "D": 62, "B": 17, "C": 46500, "C0": 36500, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 305 ECP"},
        {"tipo": "NUP", "d": 25, "D": 62, "B": 24, "C": 64000, "C0": 55000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 2305 ECP"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "NUP", "d": 30, "D": 62, "B": 16, "C": 44000, "C0": 36500, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 206 ECP"},
        {"tipo": "NUP", "d": 30, "D": 62, "B": 20, "C": 55000, "C0": 49000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 2206 ECP"},
        {"tipo": "NUP", "d": 30, "D": 72, "B": 19, "C": 58500, "C0": 48000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 306 ECP"},
        {"tipo": "NUP", "d": 30, "D": 72, "B": 27, "C": 83000, "C0": 75000, "e": 0.3, "Y": 0.6, "Y0": 0.6, "designation": "NUP 2306 ECP"},
    ]
    
    # Rodamientos de Bolas (agregados según el informe)
    rodamientos_bolas = [
        # Diámetro 25mm (Empotramiento A)
        {"tipo": "Bolas", "d": 25, "D": 47, "B": 12, "C": 14000, "C0": 6550, "e": 0.19, "Y": 2.30, "Y0": 1.4, "designation": "98205"},
        {"tipo": "Bolas", "d": 25, "D": 52, "B": 15, "C": 19500, "C0": 11400, "e": 0.22, "Y": 1.99, "Y0": 1.2, "designation": "6205"},
        {"tipo": "Bolas", "d": 25, "D": 62, "B": 17, "C": 35800, "C0": 19300, "e": 0.20, "Y": 2.22, "Y0": 1.3, "designation": "6405"},
        
        # Diámetro 30mm (Empotramiento B)
        {"tipo": "Bolas", "d": 30, "D": 55, "B": 13, "C": 17800, "C0": 9560, "e": 0.23, "Y": 1.85, "Y0": 1.1, "designation": "6006"},
        {"tipo": "Bolas", "d": 30, "D": 62, "B": 16, "C": 28100, "C0": 15600, "e": 0.23, "Y": 1.85, "Y0": 1.1, "designation": "6206"},
        {"tipo": "Bolas", "d": 30, "D": 72, "B": 19, "C": 43300, "C0": 26300, "e": 0.22, "Y": 1.99, "Y0": 1.2, "designation": "6406"},
    ]
    
    # Combinar todas las bases de datos
    todos_rodamientos = rodamientos_conicos + rodamientos_nu + rodamientos_nup + rodamientos_bolas
    
    # Crear DataFrame
    df = pd.DataFrame(todos_rodamientos)
    df['ID'] = range(1, len(df) + 1)
    
    return df

def calcular_combinaciones_rodamientos(df, condiciones):
    """Calcula todas las combinaciones posibles entre rodamientos según el informe técnico"""
    
    FrA = condiciones['FrA']
    FaA = condiciones['FaA']
    FrB = condiciones['FrB'] 
    FaB = condiciones['FaB']
    Faext = condiciones['Faext']
    n_rpm = condiciones['n_rpm']
    diametro_A = condiciones['diametro_A']
    diametro_B = condiciones['diametro_B']
    
    rodamientos_A = df[df['d'] == diametro_A].copy()
    rodamientos_B = df[df['d'] == diametro_B].copy()
    
    resultados = []
    
    for _, A in rodamientos_A.iterrows():
        for _, B in rodamientos_B.iterrows():
            
            p_A = 3.0 if A['tipo'] == 'Bolas' else 10/3  # Bolas: p=3, Rodillos: p=10/3
            p_B = 3.0 if B['tipo'] == 'Bolas' else 10/3
            
            # Cálculos según el tipo de rodamiento (basado en el informe)
            if A['tipo'] == 'Cónico' and B['tipo'] == 'Cónico':
                # Configuración back-to-back para cónicos
                YA = A['Y']
                YB = B['Y']
                FiA = 0.5 * FrA / YA
                FiB = 0.5 * FrB / YB
                
                # Condición del informe
                if FiA <= (FiB + Faext):  # CASO 1
                    PA = 0.4 * FrA + YA * (FiB + Faext)
                    PB = FrB
                else:  # CASO 2
                    PA = FrA
                    PB = 0.4 * FrB + YB * (FiA - Faext)
                
                P0A = 0.5 * FrA + A['Y0'] * (FiB + Faext)
                P0B = 0.5 * FrB + B['Y0'] * (FiA - Faext)
                
            elif A['tipo'] == 'Bolas':
                # Cálculo para rodamientos de bolas según el informe
                f0 = 14  # Factor f0 para bolas
                relacion_axial = (f0 * FaA) / A['C0']
                
                # Interpolación del factor e (simplificada)
                e_A = A['e']
                
                if FaA / FrA <= e_A:
                    PA = FrA
                else:
                    X = 0.56
                    Y = A['Y']
                    PA = X * FrA + Y * FaA
                
                P0A = 0.6 * FrA + 0.5 * FaA
                
                # Para B
                if B['tipo'] == 'Bolas':
                    if FaB == 0:
                        PB = FrB
                    else:
                        e_B = B['e']
                        if FaB / FrB <= e_B:
                            PB = FrB
                        else:
                            PB = 0.56 * FrB + B['Y'] * FaB
                    P0B = 0.6 * FrB + 0.5 * FaB
                else:
                    PB = FrB
                    P0B = FrB
                    
            elif A['tipo'] == 'NU':
                # Rodamientos NU solo soportan carga radial
                PA = FrA
                P0A = FrA
                
                if B['tipo'] == 'NUP':
                    # NUP puede soportar axial
                    if FaB / FrB <= B['e']:
                        PB = FrB
                    else:
                        PB = 0.92 * FrB + B['Y'] * FaB
                    P0B = FrB
                else:
                    PB = FrB
                    P0B = FrB
                    
            elif A['tipo'] == 'NUP':
                # Rodamientos NUP según el informe
                if FaA / FrA <= A['e']:
                    PA = FrA
                else:
                    PA = 0.92 * FrA + A['Y'] * FaA
                P0A = FrA
                
                if B['tipo'] == 'NU':
                    PB = FrB
                    P0B = FrB
                elif B['tipo'] == 'NUP':
                    if FaB / FrB <= B['e']:
                        PB = FrB
                    else:
                        PB = 0.92 * FrB + B['Y'] * FaB
                    P0B = FrB
                else:
                    PB = FrB
                    P0B = FrB
            else:
                # Caso general
                PA = FrA
                PB = FrB
                P0A = FrA
                P0B = FrB
            
            # Cálculos comunes
            CA = A['C']
            CB = B['C']
            C0A = A['C0']
            C0B = B['C0']
            
            FSA = C0A / P0A if P0A > 0 else np.inf
            FSB = C0B / P0B if P0B > 0 else np.inf
            
            L10hA = (1_000_000 / (60 * n_rpm)) * (CA / PA) ** p_A if PA > 0 else np.inf
            L10hB = (1_000_000 / (60 * n_rpm)) * (CB / PB) ** p_B if PB > 0 else np.inf
            
            resultado = {
                "ID_A": A['ID'],
                "Tipo_A": A['tipo'],
                "Designación_A": A['designation'],
                "d_A": A['d'],
                "D_A": A['D'],
                "B_A": A['B'],
                "ID_B": B['ID'],
                "Tipo_B": B['tipo'],
                "Designación_B": B['designation'],
                "d_B": B['d'],
                "D_B": B['D'],
                "B_B": B['B'],
                "FrA (N)": FrA,
                "FaA (N)": FaA,
                "FrB (N)": FrB,
                "FaB (N)": FaB,
                "Fa_ext (N)": Faext,
                "PA (N)": round(PA, 2),
                "PB (N)": round(PB, 2),
                "P0A (N)": round(P0A, 2),
                "P0B (N)": round(P0B, 2),
                "C_A (N)": CA,
                "C_B (N)": CB,
                "C0_A (N)": C0A,
                "C0_B (N)": C0B,
                "FS_A": round(FSA, 2),
                "FS_B": round(FSB, 2),
                "p_A": p_A,
                "p_B": p_B,
                "Vida_A (h)": round(L10hA, 1),
                "Vida_B (h)": round(L10hB, 1),
                "C/P_A": round(CA / PA, 2) if PA > 0 else np.inf,
                "C/P_B": round(CB / PB, 2) if PB > 0 else np.inf,
                "Vida_Min (h)": round(min(L10hA, L10hB), 1)
            }
            
            resultados.append(resultado)
    
    return pd.DataFrame(resultados)

def crear_grafico_vida_util(df_resultados, fila_seleccionada):
    """Crea gráfico de vida útil para la combinación seleccionada"""
    
    if fila_seleccionada is None or df_resultados.empty:
        return None
    
    row = df_resultados.iloc[fila_seleccionada]
    
    # Crear curva teórica usando los exponentes específicos
    C_P = np.linspace(1, 40, 500)
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
    
    if not st.session_state.resultados_calculados.empty:
        st.markdown(f"**Combinaciones calculadas:** {len(st.session_state.resultados_calculados)}")

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
            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {e}")
    
    st.divider()
    
    st.markdown("## 📚 Explicación de Cálculos por Tipo de Rodamiento")
    
    st.markdown("""
    <div class="calculation-explanation">
        <h4>🔧 Cálculos para Empotramiento A</h4>
        <p><strong>Rodamientos de Bolas:</strong></p>
        <ul>
            <li>Exponente de vida: p = 3</li>
            <li>Carga equivalente: P = Fr (si Fa/Fr ≤ e) o P = X·Fr + Y·Fa (si Fa/Fr > e)</li>
            <li>Vida útil: L₁₀ₕ = (1,000,000)/(60·n) × (C/P)³</li>
        </ul>
        <p><strong>Rodamientos Cónicos:</strong></p>
        <ul>
            <li>Exponente de vida: p = 10/3</li>
            <li>Configuración back-to-back con cálculo de fuerzas axiales inducidas</li>
            <li>Fi = 0.5 × Fr / Y</li>
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
            <li>P = Fr (si Fa/Fr ≤ e) o P = 0.92·Fr + Y·Fa (si Fa/Fr > e)</li>
            <li>Vida útil: L₁₀ₕ = (1,000,000)/(60·n) × (C/P)^(10/3)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        '<div class="calculation-explanation">'
        '<h4>🔧 Cálculos para Empotramiento B</h4>'
        '<p>Los cálculos para el empotramiento B siguen la misma metodología que el empotramiento A, '
        'pero considerando las cargas específicas aplicadas en B y las interacciones entre rodamientos '
        'en configuraciones especiales como los cónicos back-to-back.</p>'
        '<p><strong>Consideraciones especiales:</strong></p>'
        '<ul>'
        '<li>En configuración cónica back-to-back, las fuerzas axiales se distribuyen entre ambos rodamientos</li>'
        '<li>Los factores de seguridad se calculan como FS = C₀/P₀</li>'
        '<li>La vida mínima del sistema está limitada por el rodamiento con menor duración</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Paso 2: Condiciones iniciales
    st.markdown("## ⚙️ Paso 2: Condiciones de Contorno del Problema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 Configuración de Empotramientos")
        diametro_A = st.number_input("Diámetro Empotramiento A (mm):", value=25, min_value=10, max_value=100, step=5)
        diametro_B = st.number_input("Diámetro Empotramiento B (mm):", value=30, min_value=10, max_value=100, step=5)
        
        st.markdown("### 💪 Cargas Aplicadas")
        FrA = st.number_input("Fuerza radial en A (N):", value=1400, min_value=0)
        FaA = st.number_input("Fuerza axial en A (N):", value=350, min_value=0)
        FrB = st.number_input("Fuerza radial en B (N):", value=2300, min_value=0)
        FaB = st.number_input("Fuerza axial en B (N):", value=0, min_value=0)
        Faext = st.number_input("Fuerza axial externa (N):", value=350.0, min_value=0.0)
    
    with col2:
        st.markdown("### ⚡ Condiciones de Operación")
        n_rpm = st.number_input("Velocidad (RPM):", value=1400, min_value=1)
        
        st.markdown("### 📊 Exponentes de Vida (Automáticos)")
        st.info("Los exponentes 'p' se asignan automáticamente según el tipo:")
        st.write("• **Rodamientos de Bolas:** p = 3")
        st.write("• **Rodamientos de Rodillos (Cónicos, NU, NUP):** p = 10/3")
        
        # Verificar disponibilidad de rodamientos
        rodamientos_disponibles_A = df_rodamientos[df_rodamientos['d'] == diametro_A]
        rodamientos_disponibles_B = df_rodamientos[df_rodamientos['d'] == diametro_B]
        
        st.markdown("### 📋 Disponibilidad")
        st.metric("Rodamientos disponibles A", len(rodamientos_disponibles_A))
        st.metric("Rodamientos disponibles B", len(rodamientos_disponibles_B))
    
    with col3:
        st.markdown("### 🚀 Acciones")
        if st.button("🔄 Calcular Combinaciones", type="primary", use_container_width=True):
            if len(rodamientos_disponibles_A) == 0:
                st.error(f"❌ No hay rodamientos disponibles para diámetro {diametro_A}mm")
            elif len(rodamientos_disponibles_B) == 0:
                st.error(f"❌ No hay rodamientos disponibles para diámetro {diametro_B}mm")
            else:
                condiciones = {
                    'FrA': FrA,
                    'FaA': FaA,
                    'FrB': FrB,
                    'FaB': FaB,
                    'Faext': Faext,
                    'n_rpm': n_rpm,
                    'diametro_A': diametro_A,
                    'diametro_B': diametro_B
                }
                
                st.session_state.condiciones_iniciales = condiciones
                
                with st.spinner("Calculando todas las combinaciones..."):
                    st.session_state.resultados_calculados = calcular_combinaciones_rodamientos(df_rodamientos, condiciones)
                
                st.success(f"✅ Se calcularon {len(st.session_state.resultados_calculados)} combinaciones")
                st.rerun()
        
        if not st.session_state.resultados_calculados.empty:
            # Botón de descarga
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.resultados_calculados.to_excel(writer, sheet_name='Resultados', index=False)
                df_rodamientos.to_excel(writer, sheet_name='Base_Datos', index=False)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer.getvalue(),
                file_name=f"analisis_rodamientos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# Paso 3: Resultados
if not st.session_state.resultados_calculados.empty:
    st.divider()
    st.markdown("## 📈 Paso 3: Resultados y Análisis")
    
    df_resultados = st.session_state.resultados_calculados
    
    # Filtros para resultados
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tipos_A = df_resultados['Tipo_A'].unique()
        filtro_tipo_A = st.selectbox("Tipo Rodamiento A:", ['Todos'] + list(tipos_A))
    
    with col2:
        tipos_B = df_resultados['Tipo_B'].unique()
        filtro_tipo_B = st.selectbox("Tipo Rodamiento B:", ['Todos'] + list(tipos_B))
    
    with col3:
        vida_min = st.number_input("Vida mínima (h):", value=0, min_value=0)
    
    with col4:
        ordenar_por = st.selectbox("Ordenar por:", 
                                 ['Vida_Min (h)', 'Vida_A (h)', 'Vida_B (h)', 'FS_A', 'FS_B'])
    
    # Aplicar filtros
    df_filtrado_resultados = df_resultados.copy()
    
    if filtro_tipo_A != 'Todos':
        df_filtrado_resultados = df_filtrado_resultados[df_filtrado_resultados['Tipo_A'] == filtro_tipo_A]
    
    if filtro_tipo_B != 'Todos':
        df_filtrado_resultados = df_filtrado_resultados[df_filtrado_resultados['Tipo_B'] == filtro_tipo_B]
    
    if vida_min > 0:
        df_filtrado_resultados = df_filtrado_resultados[df_filtrado_resultados['Vida_Min (h)'] >= vida_min]
    
    # Ordenar
    df_filtrado_resultados = df_filtrado_resultados.sort_values(by=ordenar_por, ascending=False)
    
    st.markdown(f"### Tabla de Resultados ({len(df_filtrado_resultados)} combinaciones)")
    
    # Mostrar tabla con selección
    event = st.dataframe(
        df_filtrado_resultados,
        use_container_width=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Gráfico para la fila seleccionada
    if event.selection.rows:
        fila_seleccionada = event.selection.rows[0]
        st.markdown("### 📊 Gráfico de Vida Útil - Combinación Seleccionada")
        
        fig = crear_grafico_vida_util(df_filtrado_resultados, fila_seleccionada)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar detalles de la selección
        row_seleccionada = df_filtrado_resultados.iloc[fila_seleccionada]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 🔧 Rodamiento A (Empotramiento {row_seleccionada['d_A']}mm)")
            st.write(f"**Tipo:** {row_seleccionada['Tipo_A']}")
            st.write(f"**Designación:** {row_seleccionada['Designación_A']}")
            st.write(f"**Dimensiones:** d={row_seleccionada['d_A']}mm, D={row_seleccionada['D_A']}mm, B={row_seleccionada['B_A']}mm")
            st.write(f"**Vida útil:** {row_seleccionada['Vida_A (h)']} horas")
            st.write(f"**Factor de seguridad:** {row_seleccionada['FS_A']}")
            st.write(f"**Exponente p:** {row_seleccionada['p_A']}")
        
        with col2:
            st.markdown(f"#### 🔧 Rodamiento B (Empotramiento {row_seleccionada['d_B']}mm)")
            st.write(f"**Tipo:** {row_seleccionada['Tipo_B']}")
            st.write(f"**Designación:** {row_seleccionada['Designación_B']}")
            st.write(f"**Dimensiones:** d={row_seleccionada['d_B']}mm, D={row_seleccionada['D_B']}mm, B={row_seleccionada['B_B']}mm")
            st.write(f"**Vida útil:** {row_seleccionada['Vida_B (h)']} horas")
            st.write(f"**Factor de seguridad:** {row_seleccionada['FS_B']}")
            st.write(f"**Exponente p:** {row_seleccionada['p_B']}")
    
    else:
        st.info("👆 Selecciona una fila en la tabla para ver el gráfico de vida útil")
