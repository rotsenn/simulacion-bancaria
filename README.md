# Simulación bancaria: cajeros M/M/1 con colas FIFO

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/simulacion-bancaria)

Simulación estocástica de eventos discretos de una jornada de un banco (Banco de Colombia) con cajeros individuales. Cada cajero es un servidor con su propia cola FIFO. El script compara **9 escenarios** de distribución de cajeros (3 y 4 cajeros, mixtos o especializados en retiros/pagos), calcula indicadores con intervalos de confianza del 95 %, valida el modelo y exporta tablas y gráficas.

## Qué hace

- Genera usuarios de ocho clases (acción × tipo) con tiempos entre llegadas y de servicio exponenciales.
- Simula **30 jornadas independientes de 480 min** por escenario (configurable) con semillas reproducibles.
- Usa **números aleatorios comunes**: todos los escenarios de una réplica atienden exactamente a los mismos usuarios, lo que permite comparaciones pareadas.
- Calcula espera, tiempo de servicio, longitud de cola, utilización, usuarios pendientes al cierre, etc., con media, desviación estándar, mediana, mín., máx. e IC 95 % (t de Student).
- Ejecuta **18 pruebas de validación** (FIFO, no solapamiento, reproducibilidad, recálculo independiente, contraste con teoría M/M/1). Si alguna falla, el programa se detiene y **no exporta resultados**.
- Exporta un libro de Excel y 7 gráficas PNG.

## Escenarios

| Clave | Cajeros | Distribución | Política de asignación | Grupo |
|-------|:-------:|--------------|------------------------|-------|
| E1  | 3 | 3 mixtos | Menor carga proyectada | Principal |
| E2  | 3 | 1 retiros + 2 pagos | Menor carga proyectada | Principal |
| E3  | 3 | 2 retiros + 1 pago | Menor carga proyectada | Principal |
| E4a | 4 | 2 retiros + 2 pagos | Menor carga proyectada | Principal |
| E4b | 4 | 1 retiros + 3 pagos | Menor carga proyectada | Principal |
| E4c | 4 | 3 retiros + 1 pago | Menor carga proyectada | Principal |
| E4d | 4 | 4 mixtos | Menor carga proyectada | Principal |
| E1u | 3 | 3 mixtos | Uniforme aleatoria | Sensibilidad |
| E1q | 3 | 3 mixtos | Menor número de usuarios | Sensibilidad |

## Requisitos

- Python 3.9 o superior (probado con 3.12), o una cuenta de Google para usar Colab
- Librerías: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl` (ver `requirements.txt`)

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/simulacion-bancaria.git
cd simulacion-bancaria

# 2. (Recomendado) crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows (CMD o PowerShell)

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Cómo ejecutarlo

Ejecución estándar (30 réplicas, resultados en `./salida`):

```bash
python simulacion_bancaria.py --reps 30 --out ./salida
```

En algunos sistemas el comando es `python3` en lugar de `python`.

### Argumentos

| Argumento | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--reps`  | `30` | Número de réplicas (jornadas) por escenario. |
| `--seed`  | `20261122` | Semilla base. La réplica *r* usa la semilla `[seed, r]`. |
| `--out`   | `.` | Carpeta de salida (se crea si no existe). |
| `--cache` | *(ninguno)* | Ruta de un archivo `.pickle` para guardar los resultados y reutilizarlos en otros scripts. |

Ejemplos:

```bash
python simulacion_bancaria.py --reps 10 --out ./prueba_rapida     # prueba rápida (~10 s)
python simulacion_bancaria.py --reps 100 --seed 123 --out ./salida_100
```

Con 30 réplicas la ejecución tarda del orden de decenas de segundos.

## Ejecutarlo en Google Colab

Colab ya incluye `numpy`, `pandas`, `scipy`, `matplotlib` y `openpyxl`, por lo que no hay que instalar nada. Abre [colab.research.google.com](https://colab.research.google.com), crea un cuaderno nuevo y ejecuta cada bloque en una celda distinta.

### Opción A: subir el script directamente

```python
# Celda 1: subir el archivo (selecciona simulacion_bancaria.py)
from google.colab import files
files.upload()
```

```python
# Celda 2: ejecutar la simulación
!python simulacion_bancaria.py --reps 30 --out ./salida
```

```python
# Celda 3: ver las gráficas dentro del cuaderno
from IPython.display import Image, display
import glob
for ruta in sorted(glob.glob("salida/graficas/*.png")):
    display(Image(ruta))
```

```python
# Celda 4 (opcional): explorar el Excel
import pandas as pd
pd.read_excel("salida/resultados_simulacion_banco.xlsx", sheet_name="Validación")
```

```python
# Celda 5: descargar todos los resultados
!zip -r resultados.zip salida
from google.colab import files
files.download("resultados.zip")
```

### Opción B: clonar desde GitHub

```python
!git clone https://github.com/TU_USUARIO/simulacion-bancaria.git
%cd simulacion-bancaria
!pip install -r requirements.txt
!python simulacion_bancaria.py --reps 30 --out ./salida
```

El repositorio debe ser público (o usar un token de acceso si es privado). Después usa las celdas 3 a 5 de la opción A.

### Opción C: guardar en Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!python simulacion_bancaria.py --reps 30 --out "/content/drive/MyDrive/simulacion_bancaria/salida"
```

### Notas para Colab

- Los archivos de Colab son temporales: al cerrar o reiniciar el entorno, `salida/` se borra. Descarga el zip o guarda en Drive.
- Si aparece `ModuleNotFoundError`, ejecuta `!pip install numpy pandas scipy matplotlib openpyxl`.
- También puede llamarse desde Python: `from simulacion_bancaria import main; main(["--reps", "30", "--out", "./salida"])`.

## Salidas

Dentro de la carpeta indicada en `--out`:

```
salida/
├── resultados_simulacion_banco.xlsx
└── graficas/
    ├── 01_tiempo_servicio_por_cajero.png
    ├── 02_tiempo_espera_por_escenario.png
    ├── 03_utilizacion_cajeros.png
    ├── 04_usuarios_por_tipo.png
    ├── 05_comparacion_3_vs_4_cajeros.png
    ├── 06_longitud_promedio_colas.png
    └── 07_variabilidad_espera_entre_replicas.png
```

El libro de Excel incluye: parámetros, usuarios generados, resultados por réplica y por cajero, estadísticas generales, comparación de escenarios, intervalos de confianza, diferencias pareadas, tablas de los puntos 1 a 3 y la hoja de **Validación**. En consola se imprime la tabla de validación y la comparación de escenarios.

## Estructura del código

`simulacion_bancaria.py` está organizado en secciones numeradas:

| # | Sección | Función principal |
|---|---------|-------------------|
| 1 | Configuración de parámetros | constantes al inicio del archivo |
| 2–3 | Generación de usuarios y tiempos exponenciales | `generar_usuarios` |
| 4–6 | Eventos, colas, asignación y simulación de una jornada | `simular_jornada` |
| 7 | Ejecución de réplicas | `ejecutar_todo` |
| 8 | Cálculo de indicadores | `resumir_replica` |
| 9 | Comparación de escenarios | `tabla_comparacion`, `diferencias_pareadas` |
| 10 | Exportación (Excel y gráficas) | `exportar_excel`, `generar_graficas` |
| 11 | Validación del modelo | `validar_modelo`, `validar_mm1` |

## Parámetros del modelo

Tiempos medios en minutos, con distribución exponencial:

| Acción | Usuario | Servicio | Entre llegadas |
|--------|---------|:--------:|:--------------:|
| Retiro | Rápido    | 1 | 1 |
| Retiro | Normal    | 2 | 2 |
| Retiro | Lento     | 3 | 3 |
| Retiro | Muy lento | 4 | 3 |
| Pago   | Rápido    | 3 | 1 |
| Pago   | Normal    | 3 | 2 |
| Pago   | Lento     | 5 | 3 |
| Pago   | Muy lento | 7 | 4 |

Probabilidades: 70 % retiros y 30 % pagos. Tipo condicionado a la acción: retiros 23/40/17/20 % y pagos 10/20/30/40 % (rápido/normal/lento/muy lento).

Para modificar parámetros, edita las constantes de la sección 1 del script (`P_ACCION`, `P_TIPO`, `MEDIA_SERVICIO`, `MEDIA_LLEGADA`, `T_CIERRE`, `ESCENARIOS`).

## Supuestos y decisiones metodológicas

- **Llegadas:** se elige primero la acción y el tipo, y luego el intervalo ~ Exp(media del tipo). Es un proceso de renovación; no equivale a ocho procesos de Poisson independientes.
- **Asignación:** el usuario va, entre los cajeros compatibles, al de menor carga proyectada (suma de medias de servicio de quienes están en cola y en servicio). Desempate: más tiempo libre, luego menor índice.
- **Cierre:** las llegadas se generan en [0, 480] min y todos los que llegaron se atienden hasta terminar (sin abandonos). Utilización y longitud de cola se miden solo en [0, 480].
- No hay abandonos, descansos ni diferencias de eficiencia entre cajeros; el sistema inicia vacío.
- Las fórmulas M/M/1 solo se usan para validar el motor de colas; el sistema completo se evalúa por simulación.

## Reproducibilidad

Con la misma `--seed` y `--reps` se obtienen resultados idénticos. La réplica *r* usa `numpy.random.default_rng([seed, r])`, y todos los escenarios comparten los mismos usuarios dentro de cada réplica.

## Autor

Néstor [Apellido] — IU Digital de Antioquia, Medellín, Colombia.

## Licencia

Proyecto académico. Agrega aquí la licencia que prefieras (por ejemplo MIT).