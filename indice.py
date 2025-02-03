import asyncio
import os
from deriv_api import DerivAPI
import aiohttp
from rich.console import Console

console = Console()
API_TOKEN = os.getenv("DERIV_TOKEN", "YOUR_DERIV_TOKEN")  # Reemplaza con tu token
APP_ID = int(os.getenv("APP_ID", 67991))  # Reemplaza con tu App ID

async def get_active_symbols(filters=None):
    """
    Obtiene los símbolos activos de la API de Deriv, con la opción de filtrar por parámetros.

    Args:
        filters (dict, optional): Un diccionario de filtros para aplicar a la consulta de símbolos.
            Ejemplo: {"product_type": "multipliers", "contract_type": ["MULTUP", "MULTDOWN"]}
    
    Returns:
        list: A list of symbols that match the provided filters, or None if an error occurs.
    """
    session = aiohttp.ClientSession()
    api = DerivAPI(app_id=APP_ID, session=session)
    try:
        auth = await api.authorize({"authorize": API_TOKEN})
        if "error" in auth:
            console.print(f"[bold red]Error de autenticación: {auth['error']}[/bold red]")
            return None

        params = {"active_symbols": "brief"}
        if filters:
            params.update(filters)
            
        response = await api.active_symbols(params)
        if "error" in response:
            console.print(f"[bold red]Error al obtener símbolos: {response['error']}[/bold red]")
            return None
        
        await session.close()
        return response["active_symbols"]

    except Exception as e:
        console.print(f"[bold red]Error inesperado: {e}[/bold red]")
        return None
    finally:
        await api.disconnect()
        await session.close()


async def find_symbol(display_name, filters=None):
  """
   Busca un símbolo por su display_name en la lista de símbolos activos
  """
  symbols = await get_active_symbols(filters)
  if symbols:
    for symbol in symbols:
      if display_name in symbol["display_name"]:
        return symbol["symbol"]
  return None
    

if __name__ == "__main__":
    async def main():

        # Ejemplo 1: Obtener todos los símbolos de tipo "multipliers"
        multipliers_symbols = await get_active_symbols({"product_type": "multipliers"})
        if multipliers_symbols:
          console.print(f"[bold green]Símbolos multipliers encontrados: {len(multipliers_symbols)}[/bold green]")
          for symbol in multipliers_symbols:
            console.print(f"[bold blue]  {symbol['display_name']}: {symbol['symbol']}[/bold blue]")
        else:
            console.print(f"[bold red]Error al obtener símbolos de tipo 'multipliers'[/bold red]")

        # Ejemplo 2: Obtener todos los símbolos  "multipliers" con contratos MULTUP y MULTDOWN
        multupdown_symbols = await get_active_symbols({"product_type": "multipliers", "contract_type": ["MULTUP", "MULTDOWN"]})
        if multupdown_symbols:
           console.print(f"[bold green]Símbolos multipliers MULTUP/MULTDOWN encontrados: {len(multupdown_symbols)}[/bold green]")
           for symbol in multupdown_symbols:
              console.print(f"[bold blue]  {symbol['display_name']}: {symbol['symbol']}[/bold blue]")
        else:
          console.print(f"[bold red]Error al obtener símbolos de tipo 'multipliers' con contratos MULTUP y MULTDOWN[/bold red]")
        
        # Ejemplo 3: Buscar un símbolo por display name
        symbol = await find_symbol("Crash 500 Index",  {"product_type": "multipliers", "contract_type": ["MULTUP", "MULTDOWN"]})
        if symbol:
           console.print(f"[bold green]Símbolo Crash 500 Index  encontrado: {symbol}[/bold green]")
        else:
          console.print(f"[bold red]Error al buscar símbolo de Crash 500 Index[/bold red]")

        symbol = await find_symbol("Volatility 50 Index",  {"product_type": "multipliers", "contract_type": ["MULTUP", "MULTDOWN"]})
        if symbol:
           console.print(f"[bold green]Símbolo Volatility 50 Index  encontrado: {symbol}[/bold green]")
        else:
           console.print(f"[bold red]Error al buscar símbolo de Volatility 50 Index[/bold red]")
        
        symbol = await find_symbol("Bitcoin / USD",  {"product_type": "multipliers", "contract_type": ["MULTUP", "MULTDOWN"]})
        if symbol:
           console.print(f"[bold green]Símbolo Bitcoin / USD  encontrado: {symbol}[/bold green]")
        else:
           console.print(f"[bold red]Error al buscar símbolo de Bitcoin / USD[/bold red]")

    asyncio.run(main())