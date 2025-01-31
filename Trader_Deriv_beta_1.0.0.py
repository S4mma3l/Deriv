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
SYMBOL = os.getenv("SYMBOL", "frxEURUSD")  # Símbolo por defecto: EUR/USD
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
TRADE_AMOUNT_PERCENTAGE = float(os.getenv("TRADE_AMOUNT_PERCENTAGE", 0.01))
MIN_TRADE_AMOUNT = 5.0
MAX_TRADE_AMOUNT = 10.0
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
# --- Inicialización de componentes ---
console = Console()
api = None
session = None  # Sesión aiohttp

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

def preprocess_data(data):
    """Preprocesa los datos, incluyendo la adición de indicadores técnicos."""
    if len(data) < 60:
        raise ValueError("Datos insuficientes para preprocesar")
    df = pd.DataFrame({'Close': data})

    # Añadir indicadores técnicos usando rolling de pandas
    df['SMA'] = calculate_moving_average(data)
    df['RSI'] = calculate_rsi(data)
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
        Input(shape=(60, 3)),  # 3 características por entrada ahora
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
        X_train.append(data[i - 60:i, :])  # 3 Características
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
                [predicted_price, np.zeros((predicted_price.shape[0], 2))],
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

async def execute_trade(direction, balance, symbol, predicted_price, current_price):
    """Ejecuta una operación de compra o venta."""
    try:
        raw_amount = balance * TRADE_AMOUNT_PERCENTAGE
        trade_amount = np.clip(raw_amount, MIN_TRADE_AMOUNT, MAX_TRADE_AMOUNT)
        trade_amount = round(float(trade_amount), 2)

        console.print(
            f"[green]⌛ Ejecutando {direction} - ${trade_amount:.2f} (Pred: {predicted_price:.5f}, Actual: {current_price:.5f})[/green]"
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
                    return True  # Trade cerrado exitosamente
            else:
                console.print(f"[bold yellow]Datos de beneficio no disponibles. Esperando...[/bold yellow]")
                await asyncio.sleep(10)  # Esperar antes de reintentar

            await asyncio.sleep(5)  # Verificar cada 5 segundos

    except Exception as e:
        console.print(f"[bold red]Error monitoreando trade: {e}[/bold red]")
        return False  # Falló el cierre del trade

async def trading_strategy():
    """Función principal que ejecuta la estrategia de trading."""
    global api, session  # Mantener estos globales para una limpieza adecuada

    session = aiohttp.ClientSession()
    api = DerivAPI(app_id=APP_ID, session=session)

    try:
        auth = await api.authorize({"authorize": API_TOKEN})
        account_id = auth["authorize"]["loginid"]
        console.print(f"[bold blue]🔑 Autenticado en la cuenta: {account_id}[/bold blue]")

        balance = await get_balance(account_id) or 0.0
        console.print(f"[bold green]💰 Balance inicial: {balance:.2f} USD[/bold green]")

        model = create_model()
        scaler = None
        last_trade = 0

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

                if price_diff >= MIN_DIFF_THRESHOLD:
                    direction = "CALL" if predicted_price > current_price else "PUT"
                    contract_id = await execute_trade(direction, balance, SYMBOL, predicted_price, current_price)

                    if contract_id:
                        console.print(f"[bold magenta]💰 Balance (antes del trade): {balance:.2f} USD[/bold magenta]")  # Imprimir antes del trade
                        target_profit = balance * TRADE_AMOUNT_PERCENTAGE * TAKE_PROFIT_PERCENTAGE
                        trade_closed = await monitor_and_close_trade(contract_id, target_profit)

                        if trade_closed:
                            balance = await get_balance(account_id) or balance
                            console.print(f"[bold magenta]💰 Balance (después del trade): {balance:.2f} USD[/bold magenta]")  # Imprimir después del trade
                        else:
                            console.print(f"[bold red]Fallo en el monitoreo/cierre del trade.")

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