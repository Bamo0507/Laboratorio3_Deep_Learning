"""
Etapa 0 del pipeline: INICIALIZACION DEL ENTORNO
------------------------------------------------------------
Una sola responsabilidad: dejar el venv listo con las dependencias instaladas.

  Lee : requirements.txt
  Hace: - crea .venv/ si no existe (con Python 3.12, TensorFlow no soporta 3.13+)
        - instala requirements.txt dentro del venv
  Escribe: .venv/

Ejecutar:  python3.12 src/00_init.py
"""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_VENV = RAIZ / ".venv"
RUTA_REQUIREMENTS = RAIZ / "requirements.txt"

# TensorFlow no publica wheels para 3.13 ni 3.14 todavia.
VERSION_MINIMA = (3, 9)
VERSION_MAXIMA = (3, 12)


def python_del_venv():
    """Ruta al interprete dentro del venv, segun sistema operativo."""
    if sys.platform == "win32":
        return DIR_VENV / "Scripts" / "python.exe"
    return DIR_VENV / "bin" / "python"


def validar_version():
    actual = sys.version_info[:2]
    if actual > VERSION_MAXIMA:
        print(f"[FALLO] Python {actual[0]}.{actual[1]} no sirve: TensorFlow soporta hasta 3.12.")
        print("        Corre este script con: python3.12 src/00_init.py")
        sys.exit(1)
    if actual < VERSION_MINIMA:
        print(f"[FALLO] Python {actual[0]}.{actual[1]} es muy viejo, se necesita 3.9 o superior.")
        sys.exit(1)
    print(f"[ok] Python {actual[0]}.{actual[1]} es compatible")


def main():
    print("=" * 60)
    print("ETAPA 0: INICIALIZACION DEL ENTORNO")
    print("=" * 60)

    validar_version()

    if DIR_VENV.exists():
        print(f"[ok] el venv ya existe en {DIR_VENV}")
    else:
        print(f"[creando] venv en {DIR_VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(DIR_VENV)], check=True)

    py = python_del_venv()
    print("[instalando] actualizando pip")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)

    print(f"[instalando] dependencias de {RUTA_REQUIREMENTS.name}")
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(RUTA_REQUIREMENTS), "-q"], check=True)

    print("[ok] entorno listo")
    print("")
    print("Siguiente paso:")
    if sys.platform == "win32":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")


main()
