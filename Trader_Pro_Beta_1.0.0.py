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
import json
import random

# --- Configuración mejorada ---
API_TOKEN = os.getenv("DERIV_TOKEN", "YOUR_DERIV_TOKEN")  # Reemplaza con tu token
APP_ID = int(os.getenv("APP_ID", 67991))  # Reemplaza con tu App ID
SYMBOL = os.getenv("SYMBOL", "frxEURUSD")  # Símbolo por defecto: EUR/USD
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
TRADE_AMOUNT_PERCENTAGE = float(os.getenv("TRADE_AMOUNT_PERCENTAGE", 0.01))
MIN_TRADE_AMOUNT = 2.0
MAX_TRADE_AMOUNT = 1.0
MIN_DIFF_THRESHOLD = 0.0002
MODEL_PATH = "trading_model_v3.h5"
TRADE_DURATION = 900  # Duración del contrato en segundos (15 minutos)
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
INITIAL_LEARNING_RATE = 0.001
DATA_FILE = "market_data.json"
MAX_HISTORICAL_DATA = 100

# --- Inicialización de componentes ---
console = Console()
api = None
session = None  # Sesión aiohttp
open_trades = {}  # Diccionario para trackear trades abiertos
trade_history = []
confidence_level = 0.8  # Nivel de confianza inicial del bot
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
        model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE), loss='mse')
        console.print("[bold blue]Modelo cargado desde disco.[/bold blue]")
        return model

    model = Sequential([
        Input(shape=(60, 6)),  # 6 características por entrada ahora
        LSTM(128, return_sequences=True, activation='tanh', kernel_initializer='glorot_uniform'),
        Dropout(0.3),
        LSTM(64, return_sequences=True, activation='tanh', kernel_initializer='glorot_uniform'),
        Dropout(0.3),
        LSTM(32, activation='tanh', kernel_initializer='glorot_uniform'),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_initializer='glorot_uniform'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE), loss='mse')
    console.print("[bold blue]Nuevo modelo creado.[/bold blue]")
    return model

def train_model(model, data, epochs=30):
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
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

async def get_balance(account_id):
    """Obtiene el balance de la cuenta."""
    for _ in range(3):
        try:
            response = await api.balance({"balance": 1, "account": account_id})
            return response["balance"]["balance"]
        except Exception as e:
            console.print(f"[bold red]Error obteniendo balance: {str(e)}[/bold red]")
            await asyncio.sleep(5)
    return None

async def get_market_data(symbol, timeframe, count=180):
    """Obtiene los datos del mercado usando DerivAPI."""
    for _ in range(3):
        try:
            granularity = TIMEFRAME_MAP[timeframe]
            response = await api.ticks_history({
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": count,
                "end": "latest",
            })

            if "error" in response:
                console.print(
                    f"[bold red]Error en la respuesta de la API: {response['error']['code']} - {response['error']['message']}[/bold red]"
                )
                await asyncio.sleep(10)
                continue

            # Verificar si la respuesta contiene 'candles'
            if "candles" in response:
                prices = np.array([candle["close"] for candle in response["candles"]], dtype=float)
                return prices  # Retorna los precios si la respuesta es válida
            else:
                console.print(f"[bold red]Respuesta de API no contiene datos de velas.[/bold red]")
                return None  # Retorna None si no hay datos de velas

        except Exception as e:
            console.print(f"[bold red]Error en datos: {str(e)}[/bold red]")
            await asyncio.sleep(10)
    return None  # Retorna None después de varios intentos fallidos

async def execute_trade(direction, balance, symbol, predicted_price, current_price, confidence):
    """Ejecuta una operación de compra o venta."""
    try:
        raw_amount = balance * TRADE_AMOUNT_PERCENTAGE * confidence # Ajusta el tamaño del trade según la confianza
        trade_amount = np.clip(raw_amount, MIN_TRADE_AMOUNT, MAX_TRADE_AMOUNT)
        trade_amount = round(float(trade_amount), 2)

        console.print(
            f"[green]⌛ Ejecutando {direction} - ${trade_amount:.2f} (Pred: {predicted_price:.5f}, Actual: {current_price:.5f}, Confianza: {confidence:.2f})[/green]"
        )

        # Parámetros del trade con duración
        parameters = {
            "amount": trade_amount,
            "basis": "stake",
            "contract_type": direction,
            "currency": "USD",
            "symbol": symbol,
            "duration": TRADE_DURATION,  # Duración en segundos
            "duration_unit": "s",  # Unidad de duración (segundos)
        }

        console.print(f"[bold blue]Parámetros del trade: {parameters}[/bold blue]")
        response = await api.buy({"buy": 1, "price": 100, "parameters": parameters})
        console.print("[bold green]✅ Operación exitosa[/bold green]")
        contract_id = response["buy"]["contract_id"]  # Obtener el ID del contrato
        open_trades[contract_id] = {  # Agregar trade al diccionario de trades abiertos
            "direction": direction,
            "start_time": time.time(),
            "trade_amount": trade_amount,
            "predicted_price": predicted_price,
            "current_price": current_price,
             "confidence": confidence
        }
        return contract_id  # Retornar el ID del contrato

    except Exception as e:
        console.print(f"[bold red]❌ Error en ejecución: {str(e)}[/bold red]")
        return None

async def monitor_and_close_trade(contract_id, target_profit):
    """Monitorea un trade abierto y lo cierra cuando se alcanza el beneficio objetivo."""
    try:
        # Suscribirse al contrato solo una vez
        contract_status = await api.proposal_open_contract(
            {"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}
        )

        while True:
            # Obtener el estado actual del contrato
            contract_status = await api.proposal_open_contract(
                {"proposal_open_contract": 1, "contract_id": contract_id}
            )

            profit = contract_status["proposal_open_contract"].get("profit")
            
            if profit is not None:  # Verificar si hay datos de beneficio
                console.print(f"[bold blue]Beneficio/Pérdida: {profit}[/bold blue]")
                if profit >= target_profit:
                    console.print(f"[bold green]Beneficio objetivo de {target_profit} alcanzado. Cerrando trade.[/bold green]")
                    sell_response = await api.sell({"sell": 1, "contract_id": contract_id})
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
                    sell_response = await api.sell({"sell": 1, "contract_id": contract_id})
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
                console.print(f"[bold yellow]Datos de beneficio no disponibles. Esperando...[/bold yellow]")
                await asyncio.sleep(10)  # Esperar antes de reintentar

            await asyncio.sleep(5)  # Verificar cada 5 segundos

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
    losses = sum(1 for trade in trade_history if trade['status'] == 'loss')
    
    if (wins + losses) > 0:
      win_ratio = wins / (wins + losses)
      confidence_change = (win_ratio - confidence_level) * 0.15 # Ajustar la tasa de aprendizaje
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
        MIN_DIFF_THRESHOLD = max(0.0001, min(MIN_DIFF_THRESHOLD, 0.001))

        console.print(f"[bold cyan]Parámetros ajustados: Frecuencia: {TRADE_FREQUENCY}, Umbral: {MIN_DIFF_THRESHOLD}[/bold cyan]")

def save_market_analysis(data, file_path=DATA_FILE):
    """Guarda los datos de análisis del mercado en un archivo."""
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
def load_market_analysis(file_path=DATA_FILE):
    """Carga los datos de análisis del mercado desde un archivo."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def update_and_save_analysis(prices, scaled_data, scaler, current_price, predicted_price, trend, confidence_level):
    """Actualiza y guarda el análisis del mercado."""
    try:
      # Crear un diccionario con la información a guardar
      data = {
          "prices": prices.tolist(),
          "scaled_data": scaled_data.tolist(),
          "current_price": current_price,
          "predicted_price": predicted_price,
          "trend": trend,
          "confidence_level": confidence_level,
          "timestamp": time.time()
      }
      
      historic_data = load_market_analysis()
      historic_data.append(data)
      # Limitar la cantidad de información guardada
      if len(historic_data) > MAX_HISTORICAL_DATA:
        historic_data = historic_data[-MAX_HISTORICAL_DATA:] # Mantener los mas recientes
      save_market_analysis(historic_data)
      console.print("[bold green]✅ Análisis del mercado guardado.[/bold green]")
      return True
    except Exception as e:
      console.print(f"[bold red]⚠ Error al guardar el análisis del mercado: {str(e)}[/bold red]")
      return False

def load_and_adjust_from_history(trade_history):
    """Carga y utiliza la historia de trades para ajustar parámetros."""
    global confidence_level
    
    historic_data = load_market_analysis()
    if historic_data:
        # Ajustar la confianza inicial a partir de los datos históricos.
        last_confidence_value = historic_data[-1]['confidence_level'] if historic_data else 0.5
        confidence_level = last_confidence_value
        console.print(f"[bold cyan]Confianza inicial del bot cargada: {confidence_level:.2f}[/bold cyan]")
    else:
        console.print(f"[bold yellow]No se encontró información histórica, se usará configuración inicial.[/bold yellow]")
        # Si no hay datos historicos, crear una entrada inicial
        initial_data = {
            "prices": [],
            "scaled_data": [],
            "current_price": 0,
            "predicted_price": 0,
            "trend": 'neutral',
            "confidence_level": confidence_level,
            "timestamp": time.time()
        }
        save_market_analysis([initial_data])
        console.print(f"[bold yellow]Datos iniciales para el aprendizaje guardados.[/bold yellow]")

    return
    
def learn_from_trade(contract_id, trade_status):
    """Aprende de una operación exitosa o fallida."""
    global confidence_level
    try:
      historic_data = load_market_analysis()
      # Buscar los datos de mercado por timestamp o contract_id
      if not historic_data:
        console.print(f"[bold yellow]No hay datos historicos para aprender del trade[/bold yellow]")
        return
        
      trade_info = open_trades.get(contract_id)
      if not trade_info:
          console.print(f"[bold yellow]No hay información de trade con ese contract_id: {contract_id}[/bold yellow]")
          return
      
      trade_time = trade_info['start_time'] # Obtener el timestamp del trade para buscar los datos
      
      for idx, data in enumerate(historic_data):
        if data['timestamp'] >= trade_time:
            # Obtener la data de mercado
            if trade_status == "win":
                confidence_level = min(1, confidence_level + 0.1*data['confidence_level']) # Aumentar la confianza
                console.print(f"[bold green]✅ Aprendiendo de trade exitoso. Nueva confianza: {confidence_level}[/bold green]")
                
            elif trade_status == "loss":
                confidence_level = max(0.1, confidence_level - 0.1*data['confidence_level']) # Reducir la confianza
                console.print(f"[bold red]❌ Aprendiendo de trade fallido. Nueva confianza: {confidence_level}[/bold red]")
            
            # Eliminar los datos utilizados (opcional)
            del historic_data[idx]
            save_market_analysis(historic_data)
            
            return # Detener la búsqueda
            
      console.print(f"[bold yellow]No se encontraron datos para aprender del trade: {contract_id}[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]⚠ Error aprendiendo del trade: {str(e)}[/bold red]")

async def trading_strategy():
    """Función principal que ejecuta la estrategia de trading."""
    global api, session, open_trades, confidence_level, trade_counter # Mantener estos globales para una limpieza adecuada
    session = aiohttp.ClientSession()
    api = DerivAPI(app_id=APP_ID, session=session)
    last_trade_time = 0

    try:
        auth = await api.authorize({"authorize": API_TOKEN})
        account_id = auth["authorize"]["loginid"]
        console.print(f"[bold blue]🔑 Autenticado en la cuenta: {account_id}[/bold blue]")

        balance = await get_balance(account_id) or 0.0
        console.print(f"[bold green]💰 Balance inicial: {balance:.2f} USD[/bold green]")

        model = create_model()
        scaler = None

         # Cargar datos historicos al inicio.
        load_and_adjust_from_history(trade_history)

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

                if price_diff >= MIN_DIFF_THRESHOLD and (current_time - last_trade_time >= TRADE_FREQUENCY):
                  if len(open_trades) < MAX_OPEN_TRADES:
                    
                    direction = None
                    if trend == 'up' and predicted_price > current_price:
                        direction = "CALL"
                    elif trend == 'down' and predicted_price < current_price:
                        direction = "PUT"
                    
                    if direction:
                        # Guardar los datos antes de ejecutar el trade
                        update_and_save_analysis(prices, scaled_data, scaler, current_price, predicted_price, trend, confidence_level)
                        contract_id = await execute_trade(direction, balance, SYMBOL, predicted_price, current_price, confidence_level)
                        if contract_id:
                            console.print(f"[bold magenta]💰 Balance (antes del trade): {balance:.2f} USD[/bold magenta]")  # Imprimir antes del trade
                            target_profit = balance * TRADE_AMOUNT_PERCENTAGE * TAKE_PROFIT_PERCENTAGE
                            trade_closed = await monitor_and_close_trade(contract_id, target_profit)
                            if trade_closed:
                                balance = await get_balance(account_id) or balance
                                console.print(f"[bold magenta]💰 Balance (después del trade): {balance:.2f} USD[/bold magenta]")  # Imprimir después del trade
                                # Aprender del trade después de cerrar
                                if trade_history[-1]['status'] == 'win':
                                    learn_from_trade(contract_id, 'win')
                                elif trade_history[-1]['status'] == 'loss':
                                    learn_from_trade(contract_id, 'loss')
                            else:
                                console.print(f"[bold red]Fallo en el monitoreo/cierre del trade.")
                            last_trade_time = current_time
                            trade_counter += 1
                    else:
                        console.print(f"[bold yellow]Tendencia no confirmada o predicción no favorable, trade omitido[/bold yellow]")
                  else:
                      console.print(f"[bold yellow]Máximo de trades simultáneos alcanzado, espera...[/bold yellow]")
                
            # Limpieza de trades antiguos
            trades_to_remove = []
            for contract_id, trade_info in open_trades.items():
              if time.time() - trade_info["start_time"] > TRADE_DURATION * 1.2: # 1.2 para tener un pequeño margen
                trades_to_remove.append(contract_id)
            
            for contract_id in trades_to_remove:
                console.print(f"[bold yellow]Trade con contrato {contract_id} expirado, cerrando...[/bold yellow]")
                try:
                  sell_response = await api.sell({"sell": 1, "contract_id": contract_id})
                  console.print(f"[bold green]Trade cerrado: {sell_response}[/bold green]")
                   # Registrar trade fallido
                  trade_history.append({
                    "contract_id": contract_id,
                    "direction": open_trades[contract_id]['direction'],
                    "predicted_price": open_trades[contract_id]['predicted_price'],
                    "current_price": open_trades[contract_id]['current_price'],
                    "start_time": open_trades[contract_id]['start_time'],
                    "end_time": time.time(),
                    "profit": 0,
                    "confidence":open_trades[contract_id]['confidence'],
                    "status": "timeout"
                    })
                except Exception as e:
                  console.print(f"[bold red]Error cerrando trade expirado: {e}[/bold red]")
                del open_trades[contract_id]
            
            if trade_counter % 5 == 0:
              update_confidence(trade_history) # Actualizamos la confianza cada 5 trades
              adjust_trading_parameters() # Ajustamos los parametros cada 5 trades

            await asyncio.sleep(10)  # Verificar el precio cada 10 segundos

    except Exception as e:
        console.print(f"[bold red]🔥 Error crítico: {str(e)}[/bold red]")
        if "The truth value of an array" in str(e):
            console.print(f"[bold red]❌ Detalle del error: {e}[/bold red]")
            if 'current_price' in locals():
                console.print(f"[bold red]❌ current_price: {current_price}[/bold red]")
            else:
                console.print(f"[bold red]❌ current_price no está definido.[/bold red]")
            if 'predicted_price' in locals():
                console.print(f"[bold red]❌ predicted_price: {predicted_price}[/bold red]")
            else:
                console.print(f"[bold red]❌ predicted_price no está definido.[/bold red]")

    finally:
        if api:
            await api.disconnect()
            console.print("[bold blue]🔌 Conexión finalizada[/bold blue]")
        if session:
            await session.close()
            console.print("[bold blue]🔌 Sesión HTTP cerrada[/bold blue]")

if __name__ == "__main__":
    asyncio.run(trading_strategy())