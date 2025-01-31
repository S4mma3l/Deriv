# Trading Bot con Inteligencia Adaptativa

Este proyecto es un bot de trading automatizado que utiliza una red neuronal LSTM para predecir movimientos de precios y ejecutar operaciones en el mercado de divisas. El bot está diseñado para aprender y adaptarse a las condiciones cambiantes del mercado, mejorando su rendimiento con el tiempo.

## Características Principales

- **Predicción de Precios:** Utiliza una red neuronal LSTM para predecir los movimientos de precios, basada en datos históricos y análisis técnico.
- **Análisis Técnico:** Incorpora indicadores técnicos como la Media Móvil Simple (SMA), el Índice de Fuerza Relativa (RSI) y el MACD para un análisis más profundo del mercado.
- **Nivel de Confianza Dinámico:** Ajusta el tamaño de las operaciones y la frecuencia de trading basándose en un nivel de confianza que evoluciona con el rendimiento del bot.
- **Aprendizaje Continuo:** Aprende de su historial de trades, mejorando sus decisiones de compra y venta.
- **Ajuste Dinámico de Parámetros:** Adapta los parámetros de trading como la frecuencia y el umbral de diferencia de precios en función de su rendimiento reciente.
- **Gestión de Riesgo:** Utiliza un stop-loss para limitar las pérdidas en las operaciones.
- **Límite de Órdenes Simultáneas:** No abre más de 3 operaciones al mismo tiempo.
- **Ejecución de Trades Programada:** Realiza operaciones cada 5 minutos aproximadamente, con ajustes según la confianza.
- **Monitoreo de Operaciones:** Monitorea las operaciones abiertas y las cierra cuando alcanzan el beneficio objetivo o un stop-loss.
- **Registro de Historial:** Guarda un registro de todas las operaciones realizadas, incluyendo el estado, ganancias y pérdidas, para mejorar el aprendizaje del bot.

## Cómo Funciona

1.  **Obtención de Datos:** El bot obtiene datos de precios del mercado utilizando la API de Deriv.
2.  **Preprocesamiento:** Los datos son preprocesados y se calculan indicadores técnicos.
3.  **Entrenamiento del Modelo:** El modelo LSTM es entrenado con datos preprocesados.
4.  **Predicción de Precios:** El modelo predice el precio futuro.
5.  **Análisis de Tendencia:** Se analiza la tendencia del mercado (alcista, bajista o neutral).
6.  **Toma de Decisión:** Basado en la predicción de precios, la tendencia y el nivel de confianza, el bot decide si debe comprar o vender.
7.  **Ejecución de Trades:** Si se cumplen las condiciones, el bot ejecuta una operación.
8.  **Monitoreo y Cierre:** Las operaciones se monitorean hasta que alcanzan el beneficio objetivo, un stop-loss o la expiración.
9.  **Aprendizaje:** El bot actualiza su nivel de confianza y ajusta sus parámetros de trading basándose en el resultado de las operaciones y su historial.

## Requisitos

- Python 3.7 o superior
- Bibliotecas listadas en `requirements.txt`
- Una cuenta en Deriv con token de API

## Instalación

1.  Clona el repositorio.
2.  Crea un entorno virtual: `python -m venv venv`
3.  Activa el entorno virtual:
    -   En Linux/macOS: `source venv/bin/activate`
    -   En Windows: `venv\Scripts\activate`
4.  Instala las dependencias: `pip install -r requirements.txt`
5.  Configura las variables de entorno:
    -  `DERIV_TOKEN`: Tu token de API de Deriv.
    -  `APP_ID`: Tu App ID de Deriv.
    -  `SYMBOL`: El símbolo de la divisa a operar (por defecto, `frxEURUSD`).
    -  `TIMEFRAME`: El timeframe de las velas (`1m`, `5m`, `1h`, `1d`).
    -  `TRADE_AMOUNT_PERCENTAGE`: El porcentaje de balance que arriesgarás en cada trade.
    -  `STOP_LOSS_PERCENTAGE`: El porcentaje de stop loss
    -  `TAKE_PROFIT_PERCENTAGE`: El porcentaje de take profit
6.  Ejecuta el bot: `python trading_bot_v5.py`

## Configuración Adicional

Puedes personalizar el comportamiento del bot modificando las variables en la sección de configuración al inicio del script.

## Notas Importantes

- **Riesgo:** El trading conlleva riesgos. Este bot no garantiza ganancias y es posible perder dinero.
- **Monitoreo:** Es importante monitorear el comportamiento del bot y ajustar la configuración según sea necesario.
- **Aprendizaje:** El bot mejora con el tiempo, por lo que es crucial que se ejecute durante periodos prolongados para que alcance su máximo potencial.

## Contribución

Las contribuciones son bienvenidas. Si encuentras algún error o tienes sugerencias, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.