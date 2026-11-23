#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulacion_bancaria.py
======================
Simulación estocástica de la jornada de un banco con cajeros individuales
(cada cajero = servidor M/M/1 con cola FIFO propia).

Estructura (secciones numeradas, tal como pide el enunciado)
------------------------------------------------------------
 1. Configuración de parámetros
 2. Generación de usuarios                (generar_usuarios)
 3. Generación de tiempos exponenciales   (dentro de generar_usuarios, rng.exponential)
 4. Gestión de eventos y colas            (simular_jornada: recursión de Lindley por cajero)
 5. Asignación de cajeros                 (simular_jornada: política de menor carga proyectada)
 6. Simulación de una jornada             (simular_jornada)
 7. Ejecución de réplicas                 (ejecutar_todo)
 8. Cálculo de indicadores                (resumir_replica)
 9. Comparación de escenarios             (tabla_comparacion, diferencias_pareadas)
10. Exportación de resultados             (exportar_excel, generar_graficas)
11. Validación del modelo                 (validar_modelo, validar_mm1)

Decisiones metodológicas (ver también el documento Word)
--------------------------------------------------------
* Llegadas: se elige primero la acción (Retiro 0.70 / Pago 0.30) y el tipo de usuario
  (probabilidades condicionadas) y luego se genera el tiempo entre llegadas ~ Exp(media del
  tipo seleccionado). El proceso resultante es un proceso de renovación con intervalos
  exponenciales condicionados al tipo; NO equivale a ocho procesos de Poisson independientes
  ni, en general, a un proceso de Poisson global.
* Orden de generación por usuario: acción, tipo, intervalo desde la llegada anterior,
  tiempo de servicio ~ Exp(media del tipo) y un número uniforme auxiliar para la
  política de asignación aleatoria (solo usada en el análisis de sensibilidad).
* Política de asignación: cada usuario se envía, entre los cajeros compatibles con su
  operación, al de MENOR CARGA PROYECTADA = suma de los tiempos medios de servicio de los
  usuarios que ese cajero tiene en el sistema (en cola + en servicio). Como el servicio
  es exponencial (sin memoria), el trabajo restante esperado del usuario en servicio es su
  media completa. Desempate: cajero que lleva más tiempo libre y luego menor índice.
  Análisis de sensibilidad: (E1u) asignación uniforme aleatoria y (E1q) menor número de
  usuarios en el sistema, que no exige conocer el tipo del usuario al llegar.
* Cada cajero atiende un solo usuario a la vez y tiene su propia cola FIFO.
* Cierre de jornada: las llegadas se generan en [0, 480]. Quien llegó antes de las 480 min
  se atiende hasta terminar (sin abandonos ni rechazos); por eso hay "tiempo extra de
  cierre". Los indicadores de utilización y longitud de cola se calculan SOLO sobre la
  ventana de observación [0, 480] (se recorta el trabajo que ocurre después de 480).
  Los tiempos de espera/servicio/sistema se calculan para todos los usuarios que llegaron.
* Números aleatorios comunes (CRN): la réplica r usa la semilla [SEMILLA_BASE, r] y los
  mismos usuarios (llegadas, tipos, servicios) en TODOS los escenarios, lo que permite
  comparar escenarios de forma pareada.

Uso:  python simulacion_bancaria.py --reps 30 --out ./salida
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PARÁMETROS
# ----------------------------------------------------------------------------
ACCIONES = ["Retiro", "Pago"]
TIPOS = ["Rápido", "Normal", "Lento", "Muy lento"]
SLUG = {"Rápido": "rapido", "Normal": "normal", "Lento": "lento", "Muy lento": "muy_lento"}

P_ACCION = {"Retiro": 0.70, "Pago": 0.30}
P_TIPO = {  # probabilidades condicionadas por acción (Tabla 2)
    "Retiro": [0.23, 0.40, 0.17, 0.20],
    "Pago": [0.10, 0.20, 0.30, 0.40],
}
# Tabla 1 (minutos)
MEDIA_SERVICIO = {
    ("Retiro", "Rápido"): 1, ("Retiro", "Normal"): 2, ("Retiro", "Lento"): 3, ("Retiro", "Muy lento"): 4,
    ("Pago", "Rápido"): 3, ("Pago", "Normal"): 3, ("Pago", "Lento"): 5, ("Pago", "Muy lento"): 7,
}
MEDIA_LLEGADA = {
    ("Retiro", "Rápido"): 1, ("Retiro", "Normal"): 2, ("Retiro", "Lento"): 3, ("Retiro", "Muy lento"): 3,
    ("Pago", "Rápido"): 1, ("Pago", "Normal"): 2, ("Pago", "Lento"): 3, ("Pago", "Muy lento"): 4,
}

T_CIERRE = 480.0          # minutos de la jornada
N_REPLICAS = 30           # >= 10 exigidas
SEMILLA_BASE = 20261122
CONFIANZA = 0.95
POLITICA = "carga_proyectada"

ROLES = {"M": "Mixto", "R": "Retiros", "P": "Pagos"}
# grupo "principal": se grafica y se usa para decidir; "sensibilidad": solo análisis complementario
ESCENARIOS = {
    "E1":  {"nombre": "E1: 3 mixtos",   "etiqueta": "E1\n3 mixtos", "roles": ["M", "M", "M"],      "politica": POLITICA, "grupo": "principal"},
    "E2":  {"nombre": "E2: 1R + 2P",    "etiqueta": "E2\n1R+2P",    "roles": ["R", "P", "P"],      "politica": POLITICA, "grupo": "principal"},
    "E3":  {"nombre": "E3: 2R + 1P",    "etiqueta": "E3\n2R+1P",    "roles": ["R", "R", "P"],      "politica": POLITICA, "grupo": "principal"},
    "E4a": {"nombre": "E4a: 2R + 2P",   "etiqueta": "E4a\n2R+2P",   "roles": ["R", "R", "P", "P"], "politica": POLITICA, "grupo": "principal"},
    "E4b": {"nombre": "E4b: 1R + 3P",   "etiqueta": "E4b\n1R+3P",   "roles": ["R", "P", "P", "P"], "politica": POLITICA, "grupo": "principal"},
    "E4c": {"nombre": "E4c: 3R + 1P",   "etiqueta": "E4c\n3R+1P",   "roles": ["R", "R", "R", "P"], "politica": POLITICA, "grupo": "principal"},
    "E4d": {"nombre": "E4d: 4 mixtos (adicional)", "etiqueta": "E4d\n4 mixtos", "roles": ["M"] * 4, "politica": POLITICA, "grupo": "principal"},
    "E1u": {"nombre": "E1u: 3 mixtos, asignación uniforme aleatoria (sensibilidad)", "etiqueta": "E1u", "roles": ["M", "M", "M"],
            "politica": "uniforme", "grupo": "sensibilidad"},
    "E1q": {"nombre": "E1q: 3 mixtos, asignación por menor número de usuarios (sensibilidad)", "etiqueta": "E1q", "roles": ["M", "M", "M"],
            "politica": "menos_usuarios", "grupo": "sensibilidad"},
}
PRINCIPALES = [k for k, v in ESCENARIOS.items() if v["grupo"] == "principal"]
COLORES_ROL = {"M": "#6C7A89", "R": "#2E86AB", "P": "#E07A1F"}

ESTADO_OK = "Atendido completamente en jornada"
ESTADO_SERV = "En servicio al cierre"
ESTADO_COLA = "En cola al cierre"


def verificar_probabilidades() -> None:
    """Verifica que cada grupo de probabilidades sume 1.0."""
    assert abs(sum(P_ACCION.values()) - 1.0) < 1e-12, "P_ACCION no suma 1"
    for a in ACCIONES:
        assert abs(sum(P_TIPO[a]) - 1.0) < 1e-12, f"P_TIPO[{a}] no suma 1"


# ----------------------------------------------------------------------------
# 2-3. GENERACIÓN DE USUARIOS Y TIEMPOS EXPONENCIALES
# ----------------------------------------------------------------------------
def generar_usuarios(replica: int, semilla_base: int = SEMILLA_BASE, t_cierre: float = T_CIERRE) -> pd.DataFrame:
    """Genera los usuarios que llegan en [0, t_cierre] para una réplica (reproducible)."""
    rng = np.random.default_rng([semilla_base, replica])
    p_acc = [P_ACCION[a] for a in ACCIONES]
    filas, t, uid = [], 0.0, 0
    while True:
        accion = ACCIONES[rng.choice(2, p=p_acc)]
        tipo = TIPOS[rng.choice(4, p=P_TIPO[accion])]
        t += rng.exponential(MEDIA_LLEGADA[(accion, tipo)])           # tiempo entre llegadas
        servicio = rng.exponential(MEDIA_SERVICIO[(accion, tipo)])    # tiempo de servicio
        u_asig = rng.random()                                         # auxiliar (asignación uniforme)
        if t > t_cierre:                                              # llegó tras el cierre: se descarta
            break
        uid += 1
        filas.append((uid, accion, tipo, t, servicio, float(MEDIA_SERVICIO[(accion, tipo)]), u_asig))
    return pd.DataFrame(filas, columns=["usuario_id", "accion", "tipo", "llegada", "servicio", "media_servicio", "u_asig"])


# ----------------------------------------------------------------------------
# 4-6. EVENTOS, COLAS, ASIGNACIÓN Y SIMULACIÓN DE UNA JORNADA
# ----------------------------------------------------------------------------
def simular_jornada(usuarios: pd.DataFrame, roles: list[str], politica: str = POLITICA,
                    t_cierre: float = T_CIERRE) -> pd.DataFrame:
    """Simula una jornada. Cada cajero es un servidor con cola FIFO propia.

    Como cada cola es FIFO con un único servidor, el inicio de servicio de un usuario es
    max(llegada, fin del usuario anterior del mismo cajero) (recursión de Lindley), lo que
    es equivalente a una simulación por eventos de llegada/inicio/fin.
    """
    k = len(roles)
    compat = {
        "Retiro": [i for i, r in enumerate(roles) if r in ("M", "R")],
        "Pago": [i for i, r in enumerate(roles) if r in ("M", "P")],
    }
    for a in ACCIONES:
        if not compat[a]:
            raise ValueError(f"No hay cajeros para {a}")
    ultimo_fin = [0.0] * k
    en_sistema = [[] for _ in range(k)]  # (fin, media_servicio) de usuarios aún en el sistema

    llegada = usuarios["llegada"].to_numpy()
    servicio = usuarios["servicio"].to_numpy()
    media = usuarios["media_servicio"].to_numpy()
    accion = usuarios["accion"].to_numpy()
    u_asig = usuarios["u_asig"].to_numpy()
    n = len(usuarios)
    cajero, inicio, fin = np.empty(n, int), np.empty(n), np.empty(n)

    for i in range(n):
        t = llegada[i]
        cand = compat[accion[i]]
        if politica in ("carga_proyectada", "menos_usuarios"):
            mejor, clave_mejor = None, None
            for c in cand:
                en_sistema[c] = [(f, m) for (f, m) in en_sistema[c] if f > t]  # salen los ya terminados
                if politica == "carga_proyectada":
                    carga = sum(m for _, m in en_sistema[c])     # trabajo medio proyectado (min)
                else:
                    carga = len(en_sistema[c])                    # nº de usuarios (no requiere conocer el tipo)
                clave = (carga, ultimo_fin[c], c)          # carga, más tiempo libre, índice
                if clave_mejor is None or clave < clave_mejor:
                    mejor, clave_mejor = c, clave
            c = mejor
        elif politica == "uniforme":
            c = cand[min(int(u_asig[i] * len(cand)), len(cand) - 1)]
            en_sistema[c] = [(f, m) for (f, m) in en_sistema[c] if f > t]
        else:
            raise ValueError(politica)
        ini = max(t, ultimo_fin[c])
        fn = ini + servicio[i]
        ultimo_fin[c] = fn
        en_sistema[c].append((fn, media[i]))
        cajero[i], inicio[i], fin[i] = c, ini, fn

    df = usuarios.copy()
    df["cajero"] = cajero + 1
    df["rol_cajero"] = [roles[c] for c in cajero]
    df["inicio"], df["fin"] = inicio, fin
    df["espera"] = df["inicio"] - df["llegada"]
    df["sistema"] = df["fin"] - df["llegada"]
    df["inicio_en_jornada"] = df["inicio"] <= t_cierre
    df["estado"] = np.where(df["fin"] <= t_cierre, ESTADO_OK,
                            np.where(df["inicio"] <= t_cierre, ESTADO_SERV, ESTADO_COLA))
    return df


# ----------------------------------------------------------------------------
# 8. CÁLCULO DE INDICADORES
# ----------------------------------------------------------------------------
def _solape(a, b, T):
    """Tiempo de [a, b] que cae dentro de la ventana [0, T] (a >= 0, b >= a)."""
    return np.minimum(b, T) - np.minimum(a, T)


def resumir_replica(df: pd.DataFrame, roles: list[str], escenario: str, replica: int,
                    t_cierre: float = T_CIERRE):
    """Devuelve (fila de réplica, lista de filas por cajero)."""
    T = t_cierre
    n, k = len(df), len(roles)
    fila = dict(escenario=escenario, replica=replica, n_cajeros=k, usuarios_llegados=n,
                retiros=int((df.accion == "Retiro").sum()), pagos=int((df.accion == "Pago").sum()))
    for t in TIPOS:
        fila[{"Rápido": "rapidos", "Normal": "normales", "Lento": "lentos", "Muy lento": "muy_lentos"}[t]] = int((df.tipo == t).sum())
    for a in ACCIONES:
        for t in TIPOS:
            fila[f"{a.lower()}_{SLUG[t]}"] = int(((df.accion == a) & (df.tipo == t)).sum())
    fila["iniciaron_en_jornada"] = int(df.inicio_en_jornada.sum())
    fila["atendidos_jornada"] = int((df.estado == ESTADO_OK).sum())
    fila["en_servicio_cierre"] = int((df.estado == ESTADO_SERV).sum())
    fila["en_cola_cierre"] = int((df.estado == ESTADO_COLA).sum())
    fila["pendientes_cierre"] = fila["en_servicio_cierre"] + fila["en_cola_cierre"]
    fila["servicio_prom"] = df.servicio.mean()
    fila["espera_prom"] = df.espera.mean()
    fila["espera_max"] = df.espera.max()
    fila["sistema_prom"] = df.sistema.mean()
    fila["espera_retiro_prom"] = df.loc[df.accion == "Retiro", "espera"].mean()
    fila["espera_pago_prom"] = df.loc[df.accion == "Pago", "espera"].mean()
    fila["p_espera_mayor0"] = float((df.espera > 1e-9).mean())
    fila["p_espera_mayor5"] = float((df.espera > 5.0).mean())
    # longitud de cola y de sistema ponderadas en el tiempo (integral exacta sobre [0, T])
    lq_total = _solape(df.llegada.to_numpy(), df.inicio.to_numpy(), T).sum() / T
    l_total = _solape(df.llegada.to_numpy(), df.fin.to_numpy(), T).sum() / T
    fila["Lq_total"], fila["Lq_por_cola"], fila["L_total"] = lq_total, lq_total / k, l_total
    fila["fin_ultimo_servicio"] = df.fin.max()
    fila["tiempo_extra_cierre"] = max(0.0, df.fin.max() - T)

    filas_c, utils = [], []
    for c in range(1, k + 1):
        d = df[df.cajero == c]
        ocupado = _solape(d.inicio.to_numpy(), d.fin.to_numpy(), T).sum()
        util = ocupado / T
        utils.append(util)
        filas_c.append(dict(
            escenario=escenario, replica=replica, cajero=c, rol=ROLES[roles[c - 1]],
            asignados=len(d), iniciaron_en_jornada=int(d.inicio_en_jornada.sum()),
            atendidos_jornada=int((d.estado == ESTADO_OK).sum()),
            pendientes_cierre=int((d.estado != ESTADO_OK).sum()),
            servicio_prom=d.servicio.mean() if len(d) else np.nan,
            espera_prom=d.espera.mean() if len(d) else np.nan,
            espera_max=d.espera.max() if len(d) else np.nan,
            tiempo_ocupado_jornada=ocupado, utilizacion=util,
            Lq=_solape(d.llegada.to_numpy(), d.inicio.to_numpy(), T).sum() / T,
            fin_ultimo_servicio=d.fin.max() if len(d) else np.nan))
    fila["util_prom"], fila["util_max"], fila["util_min"] = float(np.mean(utils)), float(np.max(utils)), float(np.min(utils))
    return fila, filas_c


# ----------------------------------------------------------------------------
# 7. EJECUCIÓN DE RÉPLICAS
# ----------------------------------------------------------------------------
def ejecutar_todo(n_reps: int = N_REPLICAS, semilla_base: int = SEMILLA_BASE, escenarios: dict = ESCENARIOS):
    verificar_probabilidades()
    usuarios_all, reps, cajs = [], [], []
    for r in range(1, n_reps + 1):
        base = generar_usuarios(r, semilla_base)
        for clave, esc in escenarios.items():
            df = simular_jornada(base, esc["roles"], esc["politica"])
            df.insert(0, "escenario", clave)
            df.insert(1, "replica", r)
            fila, fc = resumir_replica(df, esc["roles"], clave, r)
            reps.append(fila)
            cajs.extend(fc)
            usuarios_all.append(df)
    usuarios = pd.concat(usuarios_all, ignore_index=True)
    usuarios["id_global"] = (usuarios.escenario + "-R" + usuarios.replica.map("{:02d}".format)
                             + "-U" + usuarios.usuario_id.map("{:04d}".format))
    return usuarios, pd.DataFrame(reps), pd.DataFrame(cajs)


# ----------------------------------------------------------------------------
# 9. ESTADÍSTICA Y COMPARACIÓN DE ESCENARIOS
# ----------------------------------------------------------------------------
def resumen_estadistico(x, conf: float = CONFIANZA) -> dict:
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n == 0:
        return dict(n=0, media=np.nan, mediana=np.nan, desv_est=np.nan, minimo=np.nan, maximo=np.nan,
                    ic95_inf=np.nan, ic95_sup=np.nan, semiancho=np.nan)
    m = x.mean()
    s = x.std(ddof=1) if n > 1 else np.nan
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, media=m, mediana=x.median(), desv_est=s, minimo=x.min(), maximo=x.max(),
                ic95_inf=m - h, ic95_sup=m + h, semiancho=h)


METRICAS_REPLICA = {
    "usuarios_llegados": "Usuarios que llegaron", "retiros": "Retiros", "pagos": "Pagos",
    "rapidos": "Usuarios rápidos", "normales": "Usuarios normales", "lentos": "Usuarios lentos",
    "muy_lentos": "Usuarios muy lentos", "iniciaron_en_jornada": "Iniciaron atención (≤480)",
    "atendidos_jornada": "Atendidos completamente (≤480)", "pendientes_cierre": "Pendientes al cierre",
    "servicio_prom": "Tiempo prom. de servicio (min)", "espera_prom": "Tiempo prom. de espera (min)",
    "espera_max": "Tiempo máx. de espera (min)", "sistema_prom": "Tiempo prom. en el sistema (min)",
    "espera_retiro_prom": "Espera prom. retiros (min)", "espera_pago_prom": "Espera prom. pagos (min)",
    "Lq_total": "Longitud prom. de cola, total (usuarios)", "Lq_por_cola": "Longitud prom. por cola (usuarios)",
    "L_total": "Usuarios prom. en el sistema", "util_prom": "Utilización promedio de cajeros",
    "util_max": "Utilización del cajero más cargado", "p_espera_mayor0": "Proporción que espera (>0)",
    "p_espera_mayor5": "Proporción que espera >5 min", "tiempo_extra_cierre": "Tiempo extra tras las 480 min (min)",
}
METRICAS_CAJERO = {
    "asignados": "Usuarios asignados", "atendidos_jornada": "Atendidos completamente (≤480)",
    "pendientes_cierre": "Pendientes al cierre", "servicio_prom": "Tiempo prom. de servicio (min)",
    "espera_prom": "Tiempo prom. de espera (min)", "espera_max": "Tiempo máx. de espera (min)",
    "utilizacion": "Utilización", "Lq": "Longitud prom. de su cola (usuarios)",
}


def estadisticas_replicas(reps: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas ENTRE réplicas (cada réplica aporta un valor) por escenario y métrica."""
    filas = []
    for esc, g in reps.groupby("escenario", sort=False):
        for m, nombre in METRICAS_REPLICA.items():
            filas.append(dict(escenario=esc, metrica=m, descripcion=nombre, **resumen_estadistico(g[m])))
    return pd.DataFrame(filas)


def estadisticas_cajeros(cajs: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (esc, c), g in cajs.groupby(["escenario", "cajero"], sort=False):
        rol = g.rol.iloc[0]
        for m, nombre in METRICAS_CAJERO.items():
            filas.append(dict(escenario=esc, cajero=c, rol=rol, metrica=m, descripcion=nombre, **resumen_estadistico(g[m])))
    return pd.DataFrame(filas)


def estadisticas_agregadas_usuarios(usuarios: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas AGREGADAS: todos los usuarios de todas las réplicas en un solo conjunto."""
    filas = []
    for esc, g in usuarios.groupby("escenario", sort=False):
        for col, nombre in [("servicio", "Servicio"), ("espera", "Espera"), ("sistema", "Tiempo en el sistema")]:
            x = g[col]
            filas.append(dict(escenario=esc, variable=nombre + " (min)", n_usuarios=len(x), media=x.mean(),
                              mediana=x.median(), desv_est=x.std(ddof=1), minimo=x.min(),
                              p90=x.quantile(0.90), p95=x.quantile(0.95), maximo=x.max()))
    return pd.DataFrame(filas)


def tabla_comparacion(stats_rep: pd.DataFrame, claves=None) -> pd.DataFrame:
    claves = claves or list(ESCENARIOS)
    cols = ["usuarios_llegados", "atendidos_jornada", "pendientes_cierre", "servicio_prom", "espera_prom",
            "espera_max", "sistema_prom", "espera_retiro_prom", "espera_pago_prom", "Lq_total", "Lq_por_cola",
            "util_prom", "util_max", "p_espera_mayor5", "tiempo_extra_cierre"]
    filas = []
    for esc in claves:
        s = stats_rep[stats_rep.escenario == esc].set_index("metrica")
        fila = {"escenario": esc, "nombre": ESCENARIOS[esc]["nombre"], "n_cajeros": len(ESCENARIOS[esc]["roles"])}
        for c in cols:
            fila[METRICAS_REPLICA[c]] = s.loc[c, "media"]
            fila[METRICAS_REPLICA[c] + " [IC95 ±]"] = s.loc[c, "semiancho"]
        filas.append(fila)
    return pd.DataFrame(filas)


def diferencias_pareadas(reps: pd.DataFrame, base: str, otros: list[str], metricas: list[str]) -> pd.DataFrame:
    """Diferencia (otro - base) por réplica con IC95 t pareado (mismos usuarios en cada réplica: CRN)."""
    filas = []
    b = reps[reps.escenario == base].set_index("replica")
    for o in otros:
        d = reps[reps.escenario == o].set_index("replica")
        for m in metricas:
            dif = (d[m] - b[m]).dropna()
            r = resumen_estadistico(dif)
            filas.append(dict(base=base, comparado=o, metrica=m, descripcion=METRICAS_REPLICA[m], diferencia_media=r["media"],
                              ic95_inf=r["ic95_inf"], ic95_sup=r["ic95_sup"],
                              significativa=bool(r["ic95_inf"] > 0 or r["ic95_sup"] < 0)))
    return pd.DataFrame(filas)


def tablas_puntos(usuarios, reps, cajs, stats_rep, stats_caj):
    """Tablas de los cinco puntos del enunciado."""
    P = {}
    # --- Punto 1: tiempo promedio de servicio por cajero (sin separar tipos de usuario)
    s = stats_caj[stats_caj.metrica == "servicio_prom"][["escenario", "cajero", "rol", "n", "media", "semiancho", "ic95_inf", "ic95_sup"]].copy()
    s = s.rename(columns={"media": "servicio_prom_min", "semiancho": "ic95_semiancho"})
    P["punto1_detalle"] = s
    res = []
    for esc, g in s.groupby("escenario", sort=False):
        gmin, gmax = g.loc[g.servicio_prom_min.idxmin()], g.loc[g.servicio_prom_min.idxmax()]
        res.append(dict(escenario=esc, cajero_menor=f"C{int(gmin.cajero)} ({gmin.rol})", menor_min=gmin.servicio_prom_min,
                        cajero_mayor=f"C{int(gmax.cajero)} ({gmax.rol})", mayor_min=gmax.servicio_prom_min,
                        diferencia_min=gmax.servicio_prom_min - gmin.servicio_prom_min))
    P["punto1_resumen"] = pd.DataFrame(res)
    # --- Punto 2: promedio de usuarios por tipo (por réplica, todos los cajeros); CRN => igual en todos los escenarios
    e1 = reps[reps.escenario == "E1"]
    filas = []
    for t in TIPOS:
        r = resumen_estadistico(e1[{"Rápido": "rapidos", "Normal": "normales", "Lento": "lentos", "Muy lento": "muy_lentos"}[t]])
        rr = resumen_estadistico(e1[f"retiro_{SLUG[t]}"])
        rp = resumen_estadistico(e1[f"pago_{SLUG[t]}"])
        filas.append(dict(tipo=t, prom_por_replica_total=r["media"], ic95_semiancho=r["semiancho"],
                          prom_retiros=rr["media"], prom_pagos=rp["media"],
                          total_todas_replicas=int(e1[{"Rápido": "rapidos", "Normal": "normales", "Lento": "lentos", "Muy lento": "muy_lentos"}[t]].sum()),
                          porcentaje=100 * e1[{"Rápido": "rapidos", "Normal": "normales", "Lento": "lentos", "Muy lento": "muy_lentos"}[t]].sum() / e1.usuarios_llegados.sum()))
    r = resumen_estadistico(e1.usuarios_llegados)
    filas.append(dict(tipo="Total", prom_por_replica_total=r["media"], ic95_semiancho=r["semiancho"],
                      prom_retiros=e1.retiros.mean(), prom_pagos=e1.pagos.mean(),
                      total_todas_replicas=int(e1.usuarios_llegados.sum()), porcentaje=100.0))
    P["punto2"] = pd.DataFrame(filas)
    # --- Punto 3: usuarios por réplica
    p3 = e1[["replica", "rapidos", "normales", "lentos", "muy_lentos", "retiros", "pagos", "usuarios_llegados"]].copy()
    p3.columns = ["Réplica", "Rápidos", "Normales", "Lentos", "Muy lentos", "Retiros", "Pagos", "Total"]
    P["punto3_replicas"] = p3
    mins = []
    for c in ["Rápidos", "Normales", "Lentos", "Muy lentos", "Retiros", "Pagos", "Total"]:
        v = p3[c].min()
        mins.append(dict(categoria=c, minimo=int(v), replicas_con_minimo=", ".join(str(int(x)) for x in p3.loc[p3[c] == v, "Réplica"])))
    P["punto3_minimos"] = pd.DataFrame(mins)
    # usuarios atendidos completamente en la jornada por tipo y escenario (aquí sí difieren entre escenarios)
    ok = usuarios[usuarios.estado == ESTADO_OK]
    t3 = ok.groupby(["escenario", "replica", "tipo"]).size().unstack(fill_value=0).reindex(columns=TIPOS, fill_value=0)
    t3["Total"] = t3.sum(axis=1)
    P["punto3_atendidos_por_escenario"] = t3.groupby("escenario").mean().reindex(list(ESCENARIOS))
    # --- Punto 4 y 5: tabla de comparación
    P["comparacion"] = tabla_comparacion(stats_rep)
    return P


# ----------------------------------------------------------------------------
# 11. VALIDACIÓN DEL MODELO
# ----------------------------------------------------------------------------
def _area_conteo(t_sube, t_baja, T):
    """Integral de un contador (+1 en t_sube, -1 en t_baja) sobre [0, T], dividida entre T. Método de barrido."""
    tiempos = np.concatenate([t_sube, t_baja])
    delta = np.concatenate([np.ones(len(t_sube)), -np.ones(len(t_baja))])
    o = np.argsort(tiempos, kind="stable")
    tc = np.minimum(tiempos[o], T)
    nivel = np.cumsum(delta[o])
    return float(np.sum(nivel[:-1] * np.diff(tc)) / T)


def validar_mm1(lam=0.5, mu=1.0, n=200_000, semilla=7):
    """Valida el motor de colas contra la teoría M/M/1: Wq=ρ/(μ-λ), W=1/(μ-λ), Lq=ρ²/(1-ρ), U=ρ."""
    rng = np.random.default_rng(semilla)
    llegada = np.cumsum(rng.exponential(1 / lam, n))
    df = pd.DataFrame(dict(usuario_id=np.arange(1, n + 1), accion="Retiro", tipo="Normal", llegada=llegada,
                           servicio=rng.exponential(1 / mu, n), media_servicio=1 / mu, u_asig=rng.random(n)))
    T = llegada[-1]
    out = simular_jornada(df, ["M"], t_cierre=T)
    rho = lam / mu
    teo = dict(Wq=rho / (mu - lam), W=1 / (mu - lam), Lq=rho ** 2 / (1 - rho), U=rho)
    sim = dict(Wq=out.espera.mean(), W=out.sistema.mean(),
               Lq=_solape(out.llegada.to_numpy(), out.inicio.to_numpy(), T).sum() / T,
               U=_solape(out.inicio.to_numpy(), out.fin.to_numpy(), T).sum() / T)
    return teo, sim


def validar_modelo(usuarios, reps, cajs, n_reps, semilla_base=SEMILLA_BASE) -> pd.DataFrame:
    """Ejecuta las pruebas de validación; devuelve tabla (prueba, resultado, detalle)."""
    R = []

    def reg(nombre, ok, detalle=""):
        R.append(dict(prueba=nombre, resultado="APROBADA" if ok else "FALLÓ", detalle=detalle))

    # 1. Probabilidades
    try:
        verificar_probabilidades()
        reg("Las probabilidades de cada grupo suman 1", True,
            f"Acción={sum(P_ACCION.values()):.2f}; Retiro={sum(P_TIPO['Retiro']):.2f}; Pago={sum(P_TIPO['Pago']):.2f}")
    except AssertionError as e:
        reg("Las probabilidades de cada grupo suman 1", False, str(e))
    # 2. Tiempos positivos
    base = usuarios[usuarios.escenario == "E1"]
    gaps_ok = all((g.llegada.diff().fillna(g.llegada) > 0).all() for _, g in base.groupby("replica"))
    reg("Tiempos de servicio y entre llegadas estrictamente positivos", bool((usuarios.servicio > 0).all() and gaps_ok),
        f"mín. servicio={usuarios.servicio.min():.6f} min")
    # 3. Sin solapamiento por cajero
    solap, fifo, compat = 0, 0, 0
    for (esc, r, c), g in usuarios.groupby(["escenario", "replica", "cajero"]):
        g = g.sort_values("inicio")
        if (g.inicio.to_numpy()[1:] < g.fin.to_numpy()[:-1] - 1e-9).any():
            solap += 1
        # FIFO: ordenados por llegada, los inicios y los id deben crecer
        h = g.sort_values("llegada")
        if not (np.all(np.diff(h.inicio.to_numpy()) >= -1e-9) and np.all(np.diff(h.usuario_id.to_numpy()) > 0)):
            fifo += 1
    reg("Ningún cajero atiende dos usuarios a la vez", solap == 0, f"cajeros-réplica con solapamiento: {solap}")
    reg("Orden de atención FIFO en cada cola", fifo == 0, f"colas con violación FIFO: {fifo}")
    # 4. Compatibilidad
    mal = usuarios[((usuarios.rol_cajero == "R") & (usuarios.accion != "Retiro")) |
                   ((usuarios.rol_cajero == "P") & (usuarios.accion != "Pago"))]
    reg("Ningún usuario atendido por cajero incompatible", len(mal) == 0, f"asignaciones incompatibles: {len(mal)}")
    # 5. Esperas no negativas
    reg("Tiempos de espera no negativos", bool((usuarios.espera >= -1e-12).all()), f"mín. espera={usuarios.espera.min():.6f}")
    # 6. Conservación de usuarios y mismos usuarios entre escenarios (CRN)
    cons = (reps.usuarios_llegados == reps.atendidos_jornada + reps.pendientes_cierre).all()
    reg("Llegados = atendidos en jornada + pendientes al cierre", bool(cons))
    crn = reps.pivot(index="replica", columns="escenario", values="usuarios_llegados").nunique(axis=1).max() == 1
    reg("Mismos usuarios en todos los escenarios de una réplica (CRN)", bool(crn))
    # 7. Reproducibilidad
    u1, u2 = generar_usuarios(3, semilla_base), generar_usuarios(3, semilla_base)
    s1 = simular_jornada(u1, ESCENARIOS["E1"]["roles"])
    s2 = simular_jornada(u2, ESCENARIOS["E1"]["roles"])
    reg("Reproducibilidad: misma semilla produce resultados idénticos", u1.equals(u2) and s1.equals(s2))
    reg("Réplicas distintas producen usuarios distintos", not generar_usuarios(1, semilla_base).equals(generar_usuarios(2, semilla_base)))
    # 8. Estadísticas vs registros individuales (método alterno: barrido de eventos y groupby)
    max_dif = 0.0
    for (esc, r), g in usuarios.groupby(["escenario", "replica"]):
        fila = reps[(reps.escenario == esc) & (reps.replica == r)].iloc[0]
        dif = [abs(fila.espera_prom - g.espera.mean()), abs(fila.servicio_prom - g.servicio.mean()),
               abs(fila.sistema_prom - (g.fin - g.llegada).mean()),
               abs(fila.Lq_total - _area_conteo(g.llegada.to_numpy(), g.inicio.to_numpy(), T_CIERRE)),
               abs(fila.usuarios_llegados - len(g))]
        for c, gc in g.groupby("cajero"):
            fc = cajs[(cajs.escenario == esc) & (cajs.replica == r) & (cajs.cajero == c)].iloc[0]
            dif.append(abs(fc.utilizacion - _area_conteo(gc.inicio.to_numpy(), gc.fin.to_numpy(), T_CIERRE)))
            dif.append(abs(fc.asignados - len(gc)))
        max_dif = max(max_dif, max(dif))
    reg("Estadísticas coinciden con registros individuales (recálculo independiente)", max_dif < 1e-8,
        f"máxima diferencia absoluta={max_dif:.2e}")
    # 9. Cajeros: suma de asignados = llegados
    sa = cajs.groupby(["escenario", "replica"]).asignados.sum().reset_index().merge(
        reps[["escenario", "replica", "usuarios_llegados"]])
    reg("Suma de usuarios por cajero = usuarios llegados", bool((sa.asignados == sa.usuarios_llegados).all()))
    # 10. Distribuciones generadas vs teóricas (4 errores estándar)
    n = len(base)
    ok_prop, peor = True, 0.0
    for a in ACCIONES:
        for i, t in enumerate(TIPOS):
            p = P_ACCION[a] * P_TIPO[a][i]
            obs = ((base.accion == a) & (base.tipo == t)).mean()
            z = abs(obs - p) / np.sqrt(p * (1 - p) / n)
            peor = max(peor, z)
            ok_prop &= z < 4
    reg("Frecuencias de (acción, tipo) coinciden con la Tabla 2", bool(ok_prop), f"mayor desvío = {peor:.2f} errores estándar")
    peor_s, peor_g = 0.0, 0.0
    for a in ACCIONES:
        for t in TIPOS:
            g = base[(base.accion == a) & (base.tipo == t)]
            peor_s = max(peor_s, abs(g.servicio.mean() - MEDIA_SERVICIO[(a, t)]) / (MEDIA_SERVICIO[(a, t)] / np.sqrt(len(g))))
            gaps = base.groupby("replica").llegada.diff().fillna(base.llegada)
            gg = gaps[(base.accion == a) & (base.tipo == t)]
            peor_g = max(peor_g, abs(gg.mean() - MEDIA_LLEGADA[(a, t)]) / (MEDIA_LLEGADA[(a, t)] / np.sqrt(len(gg))))
    reg("Medias de servicio generadas coinciden con la Tabla 1", peor_s < 4, f"mayor desvío = {peor_s:.2f} errores estándar")
    reg("Medias de tiempo entre llegadas coinciden con la Tabla 1", peor_g < 4, f"mayor desvío = {peor_g:.2f} errores estándar")
    # 11. Tamaño esperado de la jornada (renovación): E[N] ≈ T / E[intervalo]
    mu_gap = sum(P_ACCION[a] * P_TIPO[a][i] * MEDIA_LLEGADA[(a, t)] for a in ACCIONES for i, t in enumerate(TIPOS))
    reg_n = reps[reps.escenario == "E1"].usuarios_llegados
    r = resumen_estadistico(reg_n)
    esp = T_CIERRE / mu_gap
    reg("Usuarios por jornada ≈ T/E[intervalo]", bool(abs(r["media"] - esp) < 3 * r["desv_est"] / np.sqrt(r["n"]) + 1.0),
        f"esperado≈{esp:.1f}; simulado={r['media']:.1f} (IC95 {r['ic95_inf']:.1f}–{r['ic95_sup']:.1f})")
    # 12. Motor de colas vs teoría M/M/1
    teo, sim = validar_mm1()
    ok_mm1 = all(abs(sim[k] - teo[k]) / teo[k] < 0.06 for k in teo)
    reg("Motor de colas vs teoría M/M/1 (λ=0.5, μ=1)", bool(ok_mm1),
        "; ".join(f"{k}: teoría={teo[k]:.3f}, sim={sim[k]:.3f}" for k in teo))
    # 13. Utilización de E1 vs carga ofrecida (trabajo/capacidad)
    mu_s = sum(P_ACCION[a] * P_TIPO[a][i] * MEDIA_SERVICIO[(a, t)] for a in ACCIONES for i, t in enumerate(TIPOS))
    u_teo = (mu_s / mu_gap) / 3
    u_sim = resumen_estadistico(reps[reps.escenario == "E1"].util_prom)
    reg("Utilización de E1 ≈ carga ofrecida/3", bool(abs(u_sim["media"] - u_teo) < 0.03),
        f"teórica≈{u_teo:.3f}; simulada={u_sim['media']:.3f}")
    return pd.DataFrame(R)


# ----------------------------------------------------------------------------
# 10. GRÁFICAS Y EXPORTACIÓN
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


def _guardar(fig, carpeta, nombre):
    ruta = os.path.join(carpeta, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
    return ruta


def _barras_por_cajero(stats_caj, metrica, claves, titulo, ylabel, carpeta, nombre, escala=1.0):
    d = stats_caj[(stats_caj.metrica == metrica) & (stats_caj.escenario.isin(claves))]
    fig, ax = plt.subplots(figsize=(12, 5))
    x, ticks, labs, sep = 0, [], [], []
    for esc in claves:
        g = d[d.escenario == esc]
        inicio = x
        for _, f in g.iterrows():
            rol = {"Mixto": "M", "Retiros": "R", "Pagos": "P"}[f.rol]
            ax.bar(x, f.media * escala, yerr=f.semiancho * escala, color=COLORES_ROL[rol], capsize=3, edgecolor="white")
            ax.text(x, f.media * escala + f.semiancho * escala + 0.01 * (ax.get_ylim()[1] or 1), f"{f.media * escala:.2f}", ha="center", va="bottom", fontsize=8)
            ticks.append(x)
            labs.append(f"C{int(f.cajero)}")
            x += 1
        sep.append((inicio + x - 1) / 2)
        x += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    for cen, esc in zip(sep, claves):
        ax.text(cen, -0.13, ESCENARIOS[esc]["nombre"].split(" (")[0], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend(handles=[Patch(color=COLORES_ROL[k], label={"M": "Cajero mixto", "R": "Cajero de retiros", "P": "Cajero de pagos"}[k]) for k in "MRP"], loc="upper left", ncol=3, frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.subplots_adjust(bottom=0.2)
    return _guardar(fig, carpeta, nombre)


def generar_graficas(usuarios, reps, cajs, stats_rep, stats_caj, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    rutas = []
    claves = PRINCIPALES
    et = [ESCENARIOS[c]["etiqueta"] for c in claves]
    sr = stats_rep.set_index(["escenario", "metrica"])
    # 1. Tiempo promedio de servicio por cajero
    rutas.append(_barras_por_cajero(stats_caj, "servicio_prom", claves,
                 "Tiempo promedio de servicio por cajero (media de 30 réplicas, IC 95 %)",
                 "Tiempo promedio de servicio (min)", carpeta, "01_tiempo_servicio_por_cajero.png"))
    # 2. Espera promedio por escenario (total, retiros, pagos)
    fig, ax = plt.subplots(figsize=(11, 5))
    w, xs = 0.27, np.arange(len(claves))
    for j, (m, lab, col) in enumerate([("espera_prom", "Todos los usuarios", "#444444"),
                                       ("espera_retiro_prom", "Retiros", COLORES_ROL["R"]),
                                       ("espera_pago_prom", "Pagos", COLORES_ROL["P"])]):
        med = [sr.loc[(c, m), "media"] for c in claves]
        err = [sr.loc[(c, m), "semiancho"] for c in claves]
        b = ax.bar(xs + (j - 1) * w, med, w, yerr=err, capsize=3, color=col, label=lab, edgecolor="white")
        for rect, v in zip(b, med):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(et)
    ax.set_ylabel("Tiempo promedio de espera (min)")
    ax.set_title("Tiempo promedio de espera en cola por escenario (media de réplicas, IC 95 %)")
    ax.legend(frameon=False)
    rutas.append(_guardar(fig, carpeta, "02_tiempo_espera_por_escenario.png"))
    # 3. Utilización de los cajeros
    rutas.append(_barras_por_cajero(stats_caj, "utilizacion", claves,
                 "Utilización de cada cajero en la jornada de 480 min (media de réplicas, IC 95 %)",
                 "Utilización (%)", carpeta, "03_utilizacion_cajeros.png", escala=100))
    # 4. Usuarios por tipo (apilado retiro/pago)
    e1 = reps[reps.escenario == "E1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(4)
    ret = [e1[f"retiro_{SLUG[t]}"].mean() for t in TIPOS]
    pag = [e1[f"pago_{SLUG[t]}"].mean() for t in TIPOS]
    tot_ci = [resumen_estadistico(e1[f"retiro_{SLUG[t]}"] + e1[f"pago_{SLUG[t]}"])["semiancho"] for t in TIPOS]
    ax.bar(xs, ret, color=COLORES_ROL["R"], label="Retiros", edgecolor="white")
    ax.bar(xs, pag, bottom=ret, color=COLORES_ROL["P"], label="Pagos", edgecolor="white")
    ax.errorbar(xs, np.array(ret) + np.array(pag), yerr=tot_ci, fmt="none", ecolor="black", capsize=4)
    for i in range(4):
        ax.text(i, ret[i] + pag[i] + tot_ci[i] + 1, f"{ret[i] + pag[i]:.1f}", ha="center", fontsize=9)
        ax.text(i, ret[i] / 2, f"{ret[i]:.1f}", ha="center", color="white", fontsize=8)
        ax.text(i, ret[i] + pag[i] / 2, f"{pag[i]:.1f}", ha="center", color="white", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(TIPOS)
    ax.set_ylabel("Usuarios por jornada (promedio por réplica)")
    ax.set_xlabel("Tipo de usuario")
    ax.set_title("Cantidad de usuarios por tipo (promedio de 30 jornadas, IC 95 % del total)")
    ax.legend(frameon=False)
    rutas.append(_guardar(fig, carpeta, "04_usuarios_por_tipo.png"))
    # 5. Tres vs cuatro cajeros (4 paneles)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    paneles = [("espera_prom", "Espera promedio", "Tiempo (min)"), ("espera_max", "Espera máxima por jornada", "Tiempo (min)"),
               ("Lq_total", "Longitud promedio de cola total", "Usuarios en cola"), ("pendientes_cierre", "Usuarios pendientes al cierre (t = 480 min)", "Usuarios")]
    for ax, (m, tit, ylab) in zip(axs.ravel(), paneles):
        med = [sr.loc[(c, m), "media"] for c in claves]
        err = [sr.loc[(c, m), "semiancho"] for c in claves]
        cols = ["#2E86AB" if len(ESCENARIOS[c]["roles"]) == 3 else "#C0392B" for c in claves]
        ax.bar(range(len(claves)), med, yerr=err, color=cols, capsize=3, edgecolor="white")
        ax.set_xticks(range(len(claves)))
        ax.set_xticklabels([c for c in claves])
        ax.set_title(tit, fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Escenario")
    axs[0, 0].legend(handles=[Patch(color="#2E86AB", label="3 cajeros"), Patch(color="#C0392B", label="4 cajeros")], frameon=False)
    fig.suptitle("Comparación de configuraciones de 3 y 4 cajeros (media de 30 réplicas, IC 95 %)")
    fig.text(0.5, 0.005, "E1 3 mixtos · E2 1R+2P · E3 2R+1P · E4a 2R+2P · E4b 1R+3P · E4c 3R+1P · E4d 4 mixtos", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    rutas.append(os.path.join(carpeta, "05_comparacion_3_vs_4_cajeros.png"))
    fig.savefig(rutas[-1], dpi=160)
    plt.close(fig)
    # 6. Longitud promedio de las colas
    fig, ax = plt.subplots(figsize=(11, 5))
    w, xs = 0.38, np.arange(len(claves))
    for j, (m, lab, col) in enumerate([("Lq_total", "Total (suma de todas las colas)", "#34495E"), ("Lq_por_cola", "Promedio por cola", "#7FB3D5")]):
        med = [sr.loc[(c, m), "media"] for c in claves]
        err = [sr.loc[(c, m), "semiancho"] for c in claves]
        b = ax.bar(xs + (j - 0.5) * w, med, w, yerr=err, capsize=3, color=col, label=lab, edgecolor="white")
        for rect, v in zip(b, med):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(et)
    ax.set_ylabel("Longitud promedio de cola (usuarios)")
    ax.set_title("Longitud promedio de las colas ponderada en el tiempo, [0, 480] min (IC 95 %)")
    ax.legend(frameon=False)
    rutas.append(_guardar(fig, carpeta, "06_longitud_promedio_colas.png"))
    # 7. Dispersión entre réplicas (complementaria)
    fig, ax = plt.subplots(figsize=(11, 5))
    datos = [reps[reps.escenario == c].espera_prom.to_numpy() for c in claves]
    bp = ax.boxplot(datos, tick_labels=claves, patch_artist=True, showmeans=True)
    for p in bp["boxes"]:
        p.set_facecolor("#AED6F1")
    ax.set_ylabel("Espera promedio de la réplica (min)")
    ax.set_xlabel("Escenario")
    ax.set_title("Variabilidad del tiempo promedio de espera entre las 30 réplicas")
    rutas.append(_guardar(fig, carpeta, "07_variabilidad_espera_entre_replicas.png"))
    return rutas


def _formato_hoja(ws, ancho=18):
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    for fila in ws.iter_rows(min_row=1, max_row=ws.max_row):
        for c in fila:
            if c.font and c.font.bold is not True and isinstance(c.value, float):
                c.number_format = "0.0000"
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E79")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(i)].width = ancho
    ws.freeze_panes = "A2"


def exportar_excel(ruta, usuarios, reps, cajs, stats_rep, stats_caj, agg_usr, comp, dif, P, valid, cfg):
    from openpyxl.styles import Font
    with pd.ExcelWriter(ruta, engine="openpyxl") as xw:
        # Parámetros (varios bloques)
        t1 = pd.DataFrame([dict(Acción=a, Usuario=t, **{"Tiempo medio de servicio (min)": MEDIA_SERVICIO[(a, t)],
                                                         "Tiempo medio entre llegadas (min)": MEDIA_LLEGADA[(a, t)]})
                           for a in ACCIONES for t in TIPOS])
        t2 = pd.DataFrame([dict(Acción=a, **{"P(acción)": P_ACCION[a]}, **{f"P({t}|{a})": P_TIPO[a][i] for i, t in enumerate(TIPOS)},
                                **{"Suma condicionadas": sum(P_TIPO[a])}) for a in ACCIONES])
        escn = pd.DataFrame([dict(Clave=k, Nombre=v["nombre"], Cajeros=len(v["roles"]),
                                  Distribución=" | ".join(f"C{i + 1}:{ROLES[r]}" for i, r in enumerate(v["roles"])),
                                  Política={"carga_proyectada": "Menor carga proyectada", "uniforme": "Uniforme aleatoria", "menos_usuarios": "Menor número de usuarios"}[v["politica"]],
                                  Grupo=v["grupo"]) for k, v in ESCENARIOS.items()])
        gen = pd.DataFrame([("Duración de la jornada (min)", T_CIERRE), ("Réplicas por escenario", cfg["n_reps"]),
                            ("Semilla base", cfg["semilla"]), ("Semilla de la réplica r", "[semilla base, r] (numpy.random.default_rng)"),
                            ("Nivel de confianza", CONFIANZA), ("Disciplina de cola", "FIFO, una cola por cajero"),
                            ("Tratamiento del cierre", "Llegadas hasta 480 min; los usuarios llegados se atienden hasta terminar; utilización y longitud de cola se miden en [0, 480]"),
                            ("Números aleatorios comunes", "Mismos usuarios en todos los escenarios de una réplica")],
                           columns=["Parámetro", "Valor"])
        fila = 0
        for titulo, tabla in [("Tabla 1. Tiempos promedio (min)", t1), ("Tabla 2. Probabilidades", t2),
                              ("Escenarios", escn), ("Configuración general", gen)]:
            pd.DataFrame([[titulo]]).to_excel(xw, sheet_name="Parámetros", startrow=fila, index=False, header=False)
            tabla.to_excel(xw, sheet_name="Parámetros", startrow=fila + 1, index=False)
            fila += len(tabla) + 4
        # Usuarios generados (nombres según el enunciado)
        u = usuarios.copy()
        u["escenario_nombre"] = u.escenario.map(lambda k: ESCENARIOS[k]["nombre"])
        u = u[["id_global", "usuario_id", "replica", "escenario", "escenario_nombre", "accion", "tipo", "llegada", "inicio", "fin",
               "espera", "servicio", "sistema", "cajero", "rol_cajero", "estado"]]
        u.columns = ["Identificador", "N.º usuario en la réplica", "Réplica", "Escenario", "Nombre del escenario", "Tipo de acción",
                     "Tipo de usuario", "Tiempo de llegada (min)", "Inicio del servicio (min)", "Fin del servicio (min)",
                     "Tiempo de espera (min)", "Tiempo de servicio (min)", "Tiempo en el sistema (min)", "Cajero asignado",
                     "Rol del cajero", "Estado de atención"]
        u["Rol del cajero"] = u["Rol del cajero"].map(ROLES)
        u.to_excel(xw, sheet_name="Usuarios generados", index=False)
        reps.to_excel(xw, sheet_name="Resultados por réplica", index=False)
        cajs.to_excel(xw, sheet_name="Resultados por cajero", index=False)
        # Estadísticas generales: entre réplicas, por cajero, y agregadas de usuarios
        sr = stats_rep[["escenario", "descripcion", "n", "media", "mediana", "desv_est", "minimo", "maximo"]]
        sr.to_excel(xw, sheet_name="Estadísticas generales", index=False)
        r0 = len(sr) + 4
        pd.DataFrame([["Estadísticas por cajero (entre réplicas)"]]).to_excel(xw, sheet_name="Estadísticas generales", startrow=r0 - 2, index=False, header=False)
        sc = stats_caj[["escenario", "cajero", "rol", "descripcion", "n", "media", "mediana", "desv_est", "minimo", "maximo"]]
        sc.to_excel(xw, sheet_name="Estadísticas generales", startrow=r0 - 1, index=False)
        r1 = r0 + len(sc) + 3
        pd.DataFrame([["Estadísticas agregadas de todos los usuarios de todas las réplicas (nivel usuario)"]]).to_excel(
            xw, sheet_name="Estadísticas generales", startrow=r1 - 2, index=False, header=False)
        agg_usr.to_excel(xw, sheet_name="Estadísticas generales", startrow=r1 - 1, index=False)
        comp.to_excel(xw, sheet_name="Comparación de escenarios", index=False)
        ic = pd.concat([stats_rep[["escenario", "descripcion", "media", "ic95_inf", "ic95_sup", "semiancho"]].assign(nivel="Escenario", cajero=""),
                        stats_caj[["escenario", "cajero", "descripcion", "media", "ic95_inf", "ic95_sup", "semiancho"]].assign(nivel="Cajero")])
        ic["cajero"] = ic["cajero"].astype(str)
        ic = ic[["nivel", "escenario", "cajero", "descripcion", "media", "ic95_inf", "ic95_sup", "semiancho"]]
        ic.to_excel(xw, sheet_name="Intervalos de confianza", index=False)
        dif.to_excel(xw, sheet_name="Diferencias pareadas", index=False)
        P["punto1_resumen"].to_excel(xw, sheet_name="Punto 1", index=False)
        P["punto1_detalle"].to_excel(xw, sheet_name="Punto 1", startrow=len(P["punto1_resumen"]) + 3, index=False)
        P["punto2"].to_excel(xw, sheet_name="Punto 2", index=False)
        P["punto3_replicas"].to_excel(xw, sheet_name="Punto 3", index=False)
        P["punto3_minimos"].to_excel(xw, sheet_name="Punto 3", startrow=len(P["punto3_replicas"]) + 3, index=False)
        P["punto3_atendidos_por_escenario"].reset_index().to_excel(xw, sheet_name="Punto 3", startrow=len(P["punto3_replicas"]) + 14, index=False)
        valid.to_excel(xw, sheet_name="Validación", index=False)
        for ws in xw.book.worksheets:
            _formato_hoja(ws, ancho=16 if ws.title in ("Usuarios generados", "Resultados por réplica", "Resultados por cajero") else 22)
            if ws.title == "Parámetros":
                for row in ws.iter_rows():
                    if row[0].value and all(c.value is None for c in row[1:]):
                        row[0].font = Font(bold=True, size=12)
        xw.book["Validación"].column_dimensions["A"].width = 70
        xw.book["Validación"].column_dimensions["C"].width = 90
        xw.book["Estadísticas generales"].column_dimensions["B"].width = 45
        xw.book["Parámetros"].column_dimensions["B"].width = 40


# ----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def calcular_todo(n_reps=N_REPLICAS, semilla=SEMILLA_BASE):
    usuarios, reps, cajs = ejecutar_todo(n_reps, semilla)
    stats_rep = estadisticas_replicas(reps)
    stats_caj = estadisticas_cajeros(cajs)
    agg_usr = estadisticas_agregadas_usuarios(usuarios)
    comp = tabla_comparacion(stats_rep)
    P = tablas_puntos(usuarios, reps, cajs, stats_rep, stats_caj)
    claves3 = ["E2", "E3"]
    mets = ["espera_prom", "espera_max", "Lq_total", "util_prom", "pendientes_cierre", "tiempo_extra_cierre", "atendidos_jornada"]
    dif = pd.concat([diferencias_pareadas(reps, "E1", [k for k in PRINCIPALES if k != "E1"] + ["E1u", "E1q"], mets),
                     diferencias_pareadas(reps, "E2", ["E3"], mets),
                     diferencias_pareadas(reps, "E3", ["E4a", "E4b", "E4c"], mets),
                     diferencias_pareadas(reps, "E2", ["E4a", "E4b"], mets),
                     diferencias_pareadas(reps, "E1", ["E4d"], mets)], ignore_index=True).drop_duplicates()
    valid = validar_modelo(usuarios, reps, cajs, n_reps, semilla)
    return dict(usuarios=usuarios, reps=reps, cajs=cajs, stats_rep=stats_rep, stats_caj=stats_caj, agg_usr=agg_usr,
                comp=comp, P=P, dif=dif, valid=valid, cfg=dict(n_reps=n_reps, semilla=semilla))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulación bancaria M/M/1 con múltiples cajeros")
    ap.add_argument("--reps", type=int, default=N_REPLICAS)
    ap.add_argument("--seed", type=int, default=SEMILLA_BASE)
    ap.add_argument("--out", default=".")
    ap.add_argument("--cache", default=None, help="ruta de pickle para reutilizar resultados en otros scripts")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = calcular_todo(a.reps, a.seed)
    print(res["valid"].to_string(index=False))
    if (res["valid"].resultado != "APROBADA").any():
        print("\n*** Hay pruebas de validación fallidas; no se exportan resultados. ***")
        sys.exit(1)
    rutas = generar_graficas(res["usuarios"], res["reps"], res["cajs"], res["stats_rep"], res["stats_caj"], os.path.join(a.out, "graficas"))
    exportar_excel(os.path.join(a.out, "resultados_simulacion_banco.xlsx"), res["usuarios"], res["reps"], res["cajs"],
                   res["stats_rep"], res["stats_caj"], res["agg_usr"], res["comp"], res["dif"], res["P"], res["valid"], res["cfg"])
    res["graficas"] = rutas
    if a.cache:
        with open(a.cache, "wb") as f:
            pickle.dump(res, f)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\nComparación de escenarios (medias entre réplicas):")
    print(res["comp"][["escenario", "Tiempo prom. de espera (min)", "Tiempo máx. de espera (min)", "Longitud prom. de cola, total (usuarios)",
                       "Utilización promedio de cajeros", "Pendientes al cierre"]].round(3).to_string(index=False))
    print("\nArchivos generados en", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
