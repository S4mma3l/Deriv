import asyncio
import os
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from deriv_api import DerivAPI
from rich.console import Console
from rich.text import Text
from rich.style import Style
import aiohttp
import pandas as pd
from tensorflow.keras.optimizers import Adam

# --- Configuración mejorada ---
API_TOKEN = os.getenv("DERIV_TOKEN", "YOUR_DERIV_TOKEN")  # Reemplaza con tu token
APP_ID = int(os.getenv("APP_ID", 67991))  # Reemplaza con tu App ID
SYMBOL = os.getenv("SYMBOL", "1HZ50V")  # Símbolo por defecto ahora es 1HZ50V
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
FIXED_TRADE_AMOUNT = 10.0  # Monto fijo de inversión
MIN_DIFF_THRESHOLD = 0.5  # Ajustamos el umbral para CRASH500, (Este valor puede necesitar ajustes)
MODEL_PATH = "trading_model_crash500.h5"  # Cambiamos el nombre del modelo para CRASH500
TIMEFRAME_MAP = {
    "1m": 60,
    "5m": 300,
    "1h": 3600,
    "1d": 86400,
}
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", 0.05))  # 5%
TAKE_PROFIT_PERCENTAGE = float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.1))  # 10%
TRADE_FREQUENCY = 300  # Ejecutar un trade cada 5 minutos en segundos
MAX_OPEN_TRADES = 3
CONFIDENCE_THRESHOLD = 0.6 # Nivel de confianza inicial
SLIPPAGE_PERCENTAGE = float(os.getenv("SLIPPAGE_PERCENTAGE", 0.005)) # 0.5% de tolerancia a slippage

# --- Inicialización de componentes ---
console = Console()
api = None
session = None  # Sesión aiohttp
open_trades = {}  # Diccionario para trackear trades abiertos
trade_history = []
confidence_level = 0.5  # Nivel de confianza inicial del bot
confidence_history = []
trade_counter = 0

# --- Funciones de Utilidad ---

def calculate_moving_average(data, window=20):
    """Calcula la media móvil simple usando pandas."""
    if len(data) < window:
        return np.array([])
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values

def calculate_rsi(data, window=14):
    """Calcula el RSI usando pandas."""
    if len(data) < window:
        return np.array([])

    delta = pd.Series(data).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.values

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
  """Calcula el MACD usando pandas."""
  if len(data) < slow_period:
      return np.array([]), np.array([]), np.array([])
  
  df = pd.Series(data)
  ema_fast = df.ewm(span=fast_period, adjust=False).mean()
  ema_slow = df.ewm(span=slow_period, adjust=False).mean()
  macd = ema_fast - ema_slow
  signal = macd.ewm(span=signal_period, adjust=False).mean()
  histogram = macd - signal
  
  return macd.values, signal.values, histogram.values

def preprocess_data(data):
    """Preprocesa los datos, incluyendo la adición de indicadores técnicos."""
    if len(data) < 60:
        raise ValueError("Datos insuficientes para preprocesar")
    df = pd.DataFrame({'Close': data})

    # Añadir indicadores técnicos usando rolling de pandas
    df['SMA'] = calculate_moving_average(data)
    df['RSI'] = calculate_rsi(data)
    macd, signal, histogram = calculate_macd(data)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = signal
    df['MACD_HIST'] = histogram
    # Rellenar NaN con la media de cada columna
    df.fillna(df.mean(), inplace=True)
    # Escalar características
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, df.index

def create_model():
    """Crea o carga el modelo LSTM mejorado."""
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        console.print("[bold blue]Modelo cargado desde disco.[/bold blue]")
        return model

    model = Sequential([
        Input(shape=(60, 6)),  # 6 características por entrada ahora
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    console.print("[bold blue]Nuevo modelo creado.[/bold blue]")
    return model

def train_model(model, data, epochs=20):
    """Entrena el modelo LSTM con los datos proporcionados."""
    if len(data) < 61:
        return model

    X_train, y_train = [], []
    for i in range(60, len(data)):
        X_train.append(data[i - 60:i, :])  # 6 Características
        y_train.append(data[i, 0])  # Precio de Cierre
    X_train, y_train = np.array(X_train), np.array(y_train)

    if X_train.size == 0:
        return model

    # Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Callbacks para Early Stopping y Model Checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

    model.fit(
        X_train, y_train,
        batch_size=64, epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint], verbose=0
    )
    console.print("[bold green]Modelo entrenado.[/bold green]")
    return model

def predict_price(model, data, scaler):
    """Predice el precio futuro utilizando el modelo LSTM."""
    try:
        last_60_days = data[-60:, :]
        X_test = np.array([last_60_days])
        predicted_price = model.predict(X_test, verbose=0)
        # Invertir la predicción al precio original usando solo el escalado del precio de cierre
        inverse_transformed_prediction = scaler.inverse_transform(
            np.concatenate(
                [predicted_price, np.zeros((predicted_price.shape[0], 5))],
                axis=1,
            )
        )[:, 0]
        console.print(f"[bold blue]Tipo de predicted_price: {type(predicted_price)}, Valor: {predicted_price}[/bold blue]")
        console.print(f"[bold blue]Tipo de inverse_transformed_prediction: {type(inverse_transformed_prediction)}, Valor: {inverse_transformed_prediction}[/bold blue]")
        return inverse_transformed_prediction[0]
    except Exception as e:
        console.print(f"[bold red]Error en predicción: {str(e)}[/bold red]")
        return None

async def retry_api_call(api_call, *args, max_retries=3, base_delay=1):
    """Realiza una llamada al API con reintentos y retroceso exponencial."""
    for attempt in range(max_retries):
        try:
            return await api_call(*args)
        except Exception as e:
            console.print(f"[bold red]Intento {attempt + 1} fallido: {e}[/bold red]")
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # Retroceso exponencial
            await asyncio.sleep(delay)

async def get_balance(account_id):
    """Obtiene el balance de la cuenta."""
    response = await retry_api_call(api.balance, {"balance": 1, "account": account_id})
    if response and "balance" in response:
      return response["balance"]["balance"]
    else:
      console.print("[bold red]No se pudo obtener el balance de la cuenta.[/bold red]")
      return None # Si hay error, retornamos None y evitamos el error.

async def get_market_data(symbol, timeframe, count=180):
    """Obtiene los datos del mercado usando DerivAPI con reintentos."""
    granularity = TIMEFRAME_MAP[timeframe]
    response = await retry_api_call(api.ticks_history, {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": granularity,
        "count": count,
        "end": "latest",
    })

    if response and "candles" in response:
      prices = np.array([candle["close"] for candle in response["candles"]], dtype=float)
      return prices
    
    console.print(f"[bold red]No se obtuvieron datos de velas válidos.[/bold red]")
    return None
    
async def execute_trade(direction, symbol, predicted_price, current_price, confidence):
    """Ejecuta una operación de compra o venta."""
    try:
        trade_amount = FIXED_TRADE_AMOUNT

        console.print(
            f"[green]⌛ Ejecutando {direction} - ${trade_amount:.2f} (Pred: {predicted_price:.5f}, Actual: {current_price:.5f}, Confianza: {confidence:.2f})[/green]"
        )

        # Parámetros del trade para vanilla options
        parameters = {
            "amount": trade_amount,
            "basis": "stake",
            "contract_type": direction, # "CALL" o "PUT"
            "currency": "USD",
            "symbol": symbol,
             "stop_loss": STOP_LOSS_PERCENTAGE,
             "take_profit": TAKE_PROFIT_PERCENTAGE,
        }
        
        console.print(f"[bold blue]Parámetros del trade: {parameters}[/bold blue]")

        # Control de deslizamiento (slippage)
        buy_response = await api.buy({"buy": 1, "price": 100, "parameters": parameters})

        if buy_response.get("error"):
            console.print(f"[bold red]Error en la compra: {buy_response['error']}[/bold red]")
            return None
        
        contract_id = buy_response["buy"]["contract_id"]
        console.print(f"[bold green]✅ Operación exitosa. ID: {contract_id}[/bold green]")
        
        open_trades[contract_id] = {  # Agregar trade al diccionario de trades abiertos
            "direction": direction,
            "start_time": time.time(),
            "trade_amount": trade_amount,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "confidence": confidence,
             "status": "open"
        }
        return contract_id  # Retornar el ID del contrato

    except Exception as e:
        console.print(f"[bold red]❌ Error en ejecución: {str(e)}[/bold red]")
        return None
    
async def monitor_and_close_trade(contract_id):
    """Monitorea un trade abierto y lo cierra cuando se alcanza el beneficio objetivo."""
    try:
      
      while open_trades.get(contract_id):
        portfolio = await retry_api_call(api.portfolio, {"portfolio": 1}, max_retries=5)
        if "error" in portfolio:
          console.print(f"[bold red]Error portfolio: {portfolio['error']}[/bold red]")
          await asyncio.sleep(10)
          continue
          
        contract_info = next((item for item in portfolio.get('portfolio',[]) if item['contract_id'] == contract_id), None)
          
        if contract_info:
            profit = contract_info.get('profit')
            if profit is not None:
              console.print(f"[bold blue]Beneficio/Pérdida: {profit}[/bold blue]")
              
              if profit >= (open_trades[contract_id]["trade_amount"] * TAKE_PROFIT_PERCENTAGE):
                  console.print(f"[bold green]Beneficio objetivo alcanzado. Cerrando trade.[/bold green]")
                  sell_response = await retry_api_call(api.sell, {"sell": 1, "contract_id": contract_id}, max_retries=5)
                  console.print(f"[bold green]Trade cerrado: {sell_response}[/bold green]")
                  
                  # Registrar trade exitoso
                  trade_history.append({
                      "contract_id": contract_id,
                      "direction": open_trades[contract_id]['direction'],
                      "predicted_price": open_trades[contract_id]['predicted_price'],
                      "current_price": open_trades[contract_id]['current_price'],
                      "start_time": open_trades[contract_id]['start_time'],
                      "end_time": time.time(),
                      "profit": profit,
                      "confidence":open_trades[contract_id]['confidence'],
                      "status": "win"
                  })
                  del open_trades[contract_id] # Remover el trade del diccionario de trades abiertos
                  return True  # Trade cerrado exitosamente
                  
              elif profit < 0 and abs(profit) >= (open_trades[contract_id]["trade_amount"] * STOP_LOSS_PERCENTAGE):
                  console.print(f"[bold yellow]Stop-loss alcanzado. Cerrando trade.[/bold yellow]")
                  sell_response = await retry_api_call(api.sell, {"sell": 1, "contract_id": contract_id}, max_retries=5)
                  console.print(f"[bold yellow]Trade cerrado (stop-loss): {sell_response}[/bold yellow]")
                   # Registrar trade fallido
                  trade_history.append({
                      "contract_id": contract_id,
                      "direction": open_trades[contract_id]['direction'],
                      "predicted_price": open_trades[contract_id]['predicted_price'],
                      "current_price": open_trades[contract_id]['current_price'],
                      "start_time": open_trades[contract_id]['start_time'],
                      "end_time": time.time(),
                      "profit": profit,
                      "confidence":open_trades[contract_id]['confidence'],
                      "status": "loss"
                  })
                  del open_trades[contract_id]
                  return True
              else:
                  await asyncio.sleep(5) # check every 5 seconds

            else:
                 console.print(f"[bold yellow]Datos de beneficio no disponibles. Esperando...[/bold yellow]")
                 await asyncio.sleep(10)  # Esperar antes de reintentar
        else:
          # Si por alguna razón el contrato no está en el portfolio, lo removemos de open_trades y logueamos el error
          console.print(f"[bold red]Trade {contract_id} no encontrado en el portfolio, cerrando...[/bold red]")
          trade_history.append({
              "contract_id": contract_id,
                "direction": open_trades[contract_id]['direction'],
              "predicted_price": open_trades[contract_id]['predicted_price'],
              "current_price": open_trades[contract_id]['current_price'],
              "start_time": open_trades[contract_id]['start_time'],
              "end_time": time.time(),
              "profit": 0,
              "confidence":open_trades[contract_id]['confidence'],
              "status": "not_found"
          })
          del open_trades[contract_id]
          return False


    except Exception as e:
      console.print(f"[bold red]Error monitoreando trade: {e}[/bold red]")
      return False  # Falló el cierre del trade
    
async def analyze_trend(prices, window=30):
    """Analiza la tendencia de los precios. Retorna 'up', 'down' o 'neutral'."""
    if len(prices) < window:
        return 'neutral'
    
    sma = calculate_moving_average(prices, window)
    if len(sma) < 2:
      return 'neutral'
    
    if sma[-1] > sma[-2]:
        return 'up'
    elif sma[-1] < sma[-2]:
        return 'down'
    else:
        return 'neutral'

def update_confidence(trade_history):
    """Actualiza el nivel de confianza del bot basado en el historial de trades."""
    global confidence_level, confidence_history
    if not trade_history:
        return
    
    wins = sum(1 for trade in trade_history if trade['status'] == 'win')
    losses = sum(1 for trade in trade_history if trade['status'] in ['loss', 'not_found', 'timeout'])
    
    if (wins + losses) > 0:
      win_ratio = wins / (wins + losses)
      confidence_change = (win_ratio - confidence_level) * 0.1 # Ajustar la tasa de aprendizaje
      confidence_level = min(max(0.1, confidence_level + confidence_change), 1) # Rango de 0.1 a 1
    
    confidence_history.append({"time": time.time(), "confidence": confidence_level})
    console.print(f"[bold cyan]Confianza del bot actualizada: {confidence_level:.2f}[/bold cyan]")
    return confidence_level

def adjust_trading_parameters():
    """Ajusta los parámetros de trading de forma dinámica."""
    global TRADE_FREQUENCY, MIN_DIFF_THRESHOLD
    # Si hay un histórico, revisamos los ultimos 10 trades.
    if len(trade_history) >= 10:
        last_10_trades = trade_history[-10:]
        # Si hubo pocas ganancias, ajustamos los paramentros para ser mas cautelosos
        wins = sum(1 for trade in last_10_trades if trade["status"] == "win")
        if wins <= 3:
            console.print("[bold yellow]Ajustando parámetros para ser más cauteloso...[/bold yellow]")
            TRADE_FREQUENCY = int(TRADE_FREQUENCY * 1.1) # Reducimos la frecuencia
            MIN_DIFF_THRESHOLD = MIN_DIFF_THRESHOLD * 1.2 # Reducimos la frecuencia
        else:
            # Caso contrario, aumentamos la frecuencia de trading y reducimos el umbral
           console.print("[bold green]Ajustando parámetros para ser más agresivo...[/bold green]")
           TRADE_FREQUENCY = int(TRADE_FREQUENCY * 0.9) # Reducimos la frecuencia
           MIN_DIFF_THRESHOLD = MIN_DIFF_THRESHOLD * 0.8 # Reducimos la frecuencia

        # Aseguramos que no sean valores extremos
        TRADE_FREQUENCY = max(60, min(TRADE_FREQUENCY, 600))
        MIN_DIFF_THRESHOLD = max(0.0001, min(MIN_DIFF_THRESHOLD, 10))

        console.print(f"[bold cyan]Parámetros ajustados: Frecuencia: {TRADE_FREQUENCY}, Umbral: {MIN_DIFF_THRESHOLD}[/bold cyan]")

async def trading_strategy():
    """Función principal que ejecuta la estrategia de trading."""
    global open_trades, confidence_level, trade_counter  # Mantener estos globales para una limpieza adecuada

    last_trade_time = 0
    session = None
    api = None

    try:
        session = aiohttp.ClientSession()
        api = DerivAPI(app_id=APP_ID, session=session)

        auth = await retry_api_call(api.authorize, {"authorize": API_TOKEN})
        account_id = auth["authorize"]["loginid"]
        console.print(f"[bold blue]🔑 Autenticado en la cuenta: {account_id}[/bold blue]")

        balance = await get_balance(account_id)
        if balance is None: # En caso de error al obtener el balance, terminamos
            console.print(f"[bold red]No se pudo obtener el balance inicial, terminando...[/bold red]")
            return

        console.print(f"[bold green]💰 Balance inicial: {balance:.2f} USD[/bold green]")

        model = create_model()
        scaler = None

        while True:
            current_time = time.time()

            prices = await get_market_data(SYMBOL, TIMEFRAME)

            if prices is None or len(prices) < 60:
                console.print("[yellow]📊 Esperando datos...[/yellow]")
                await asyncio.sleep(30)
                continue

            try:
                scaled_data, scaler, _ = preprocess_data(prices)  # _ para timestamps si no se usan
            except Exception as e:
                console.print(f"[bold red]⚠ Error en preprocesamiento: {str(e)}[/bold red]")
                await asyncio.sleep(10)
                continue

            if len(prices) >= 120:  # Entrenar si hay suficientes datos
                model = train_model(model, scaled_data)

            current_price = float(prices[-1])  # Definir current_price AQUÍ
            predicted_price = predict_price(model, scaled_data, scaler)  # Definir predicted_price AQUÍ

            if predicted_price is not None:
                price_diff = abs(predicted_price - current_price)  # Definir price_diff AQUÍ

                # Realizar análisis de tendencia
                trend = await analyze_trend(prices)

                if price_diff >= MIN_DIFF_THRESHOLD and (
                    current_time - last_trade_time >= TRADE_FREQUENCY
                ):
                    if len(open_trades) < MAX_OPEN_TRADES:
                        direction = None
                        if trend == "up" and predicted_price > current_price:
                            direction = "CALL"
                        elif trend == "down" and predicted_price < current_price:
                            direction = "PUT"

                        if direction:
                            contract_id = await execute_trade(
                                direction, SYMBOL, predicted_price, current_price, confidence_level
                            )
                            if contract_id:
                                console.print(
                                    f"[bold magenta]💰 Balance (antes del trade): {balance:.2f} USD[/bold magenta]"
                                )  # Imprimir antes del trade
                                trade_closed = await monitor_and_close_trade(contract_id)
                                if trade_closed:
                                    balance = await get_balance(account_id) or balance
                                    console.print(
                                        f"[bold magenta]💰 Balance (después del trade): {balance:.2f} USD[/bold magenta]"
                                    )  # Imprimir después del trade
                                else:
                                    console.print(f"[bold red]Fallo en el monitoreo/cierre del trade.")
                                last_trade_time = current_time
                                trade_counter += 1
                        else:
                            console.print(
                                f"[bold yellow]Tendencia no confirmada o predicción no favorable, trade omitido[/bold yellow]"
                            )
                    else:
                        console.print(f"[bold yellow]Máximo de trades simultáneos alcanzado, espera...[/bold yellow]")

            if trade_counter % 5 == 0:
                update_confidence(trade_history)  # Actualizamos la confianza cada 5 trades
                adjust_trading_parameters()  # Ajustamos los parametros cada 5 trades

            await asyncio.sleep(10)  # Verificar el precio cada 10 segundos

    except Exception as e:
        console.print(f"[bold red]🔥 Error crítico: {str(e)}[/bold red]")
        if "The truth value of an array" in str(e):
            console.print(f"[bold red]❌ Detalle del error: {e}[/bold red]")
            if "current_price" in locals():
                console.print(f"[bold red]❌ current_price: {current_price}[/bold red]")
            else:
                console.print(f"[bold red]❌ current_price no está definido.[/bold red]")
            if "predicted_price" in locals():
                console.print(f"[bold red]❌ predicted_price: {predicted_price}[/bold red]")
            else:
                console.print(f"[bold red]❌ predicted_price no está definido.[/bold red]")

    finally:
        if session:
            await session.close()
            console.print("[bold blue]🔌 Sesión HTTP cerrada[/bold blue]")

if __name__ == "__main__":
    asyncio.run(trading_strategy())