"""Funciones compartidas por las etapas del pipeline: logging, I/O y validacion."""

import sys

import pandas as pd


def banner(titulo):
    """Separador visual con el titulo de la etapa."""
    print("=" * 60)
    print(titulo.upper())
    print("=" * 60)


def afirmar(condicion, mensaje):
    """Validacion fail-fast: corta la ejecucion si la condicion no se cumple."""
    if condicion:
        print(f"[ok] {mensaje}")
    else:
        print(f"[FALLO] {mensaje}")
        sys.exit(1)


def cargar_df(ruta, **kwargs):
    """Lee un csv y reporta su forma."""
    df = pd.read_csv(ruta, **kwargs)
    print(f"[cargado]  {ruta.name} -> {len(df)} filas, {len(df.columns)} columnas")
    return df


def guardar_df(df, ruta):
    """Escribe un csv creando el directorio si hace falta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"[guardado] {ruta.name} -> {len(df)} filas, {len(df.columns)} columnas")


def guardar_figura(fig, nombre, dir_figuras):
    """Escribe una figura en png para reusarla en el informe."""
    dir_figuras.mkdir(parents=True, exist_ok=True)
    ruta = dir_figuras / f"{nombre}.png"
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    print(f"[guardado] {ruta.name}")
