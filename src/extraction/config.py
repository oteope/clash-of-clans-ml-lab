"""
Módulo de configuración para el proyecto Clash of Clans ML Lab.

Carga variables de entorno desde .env, verifica la IP pública actual
y compara con una IP esperada opcional para advertencias de bloqueo.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
import aiohttp

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Cargar variables de entorno desde .env
load_dotenv()

# Token de Supercell
COC_API_TOKEN: Optional[str] = os.getenv("COC_API_TOKEN")

# IP esperada (opcional)
EXPECTED_IP: Optional[str] = os.getenv("EXPECTED_IP")


async def get_public_ip() -> str:
    """
    Obtiene la IP pública actual usando api.ipify.org.

    Returns:
        str: IP pública en formato IPv4.

    Raises:
        RuntimeError: Si no se puede obtener la IP.
    """
    url = "https://api.ipify.org?format=json"
    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                ip = data.get("ip")
                if not ip:
                    raise RuntimeError("No se pudo obtener la IP de la respuesta.")
                logger.info("IP pública obtenida: %s", ip)
                return ip
    except aiohttp.ClientError as e:
        logger.error("Error al obtener IP pública: %s", e)
        raise RuntimeError(f"Error al obtener IP pública: {e}") from e
    except Exception as e:
        logger.error("Error inesperado al obtener IP pública: %s", e)
        raise RuntimeError(f"Error inesperado al obtener IP pública: {e}") from e


async def verify_ip() -> None:
    """
    Verifica la IP pública actual contra EXPECTED_IP si está definida.

    Si no coincide, registra una advertencia.
    """
    if not EXPECTED_IP:
        logger.info("EXPECTED_IP no definida en .env, omitiendo verificación de IP.")
        return

    try:
        current_ip = await get_public_ip()
    except RuntimeError as e:
        logger.error("No se pudo verificar la IP: %s", e)
        return

    if current_ip != EXPECTED_IP:
        logger.warning(
            "La IP pública actual (%s) no coincide con la IP esperada (%s). "
            "El token de Supercell puede estar bloqueado por IP.",
            current_ip,
            EXPECTED_IP,
        )
    else:
        logger.info("IP pública coincide con la esperada (%s).", current_ip)


async def main() -> None:
    """
    Función principal para ejecutar la verificación de IP.
    """
    if not COC_API_TOKEN:
        logger.error("COC_API_TOKEN no está definido en .env")
        return

    logger.info("COC_API_TOKEN cargado correctamente.")
    await verify_ip()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
