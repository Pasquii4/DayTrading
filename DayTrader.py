import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
from pathlib import Path
import json
from colorama import Fore, Back, Style, init

# Inicializar colorama para colores en consola
init(autoreset=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES - CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScannerConfig:
    """Configuración del scanner de day trading."""
    min_price: float = 5.0
    max_price: float = 500.0
    min_volume: int = 2_000_000
    min_relative_volume: float = 1.5
    min_atr_percent: float = 2.0
    min_score: int = 60
    lookback_days: int = 60
    max_workers: int = 10
    
@dataclass
class DayTradingOpportunity:
    """Representa una oportunidad de day trading identificada."""
    ticker: str
    precio_actual: float
    cambio_percent: float
    volumen: int
    volumen_promedio: int
    volumen_relativo: float
    atr: float
    atr_percent: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    tendencia: str
    señal: str
    puntuacion: int
    nivel_entrada: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    breakout_confirmado: bool
    momentum_score: int
    volatilidad_score: int
    volumen_score: int
    detalles: str = ""
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para exportación."""
        return {
            'Ticker': self.ticker,
            'Precio': f'${self.precio_actual:.2f}',
            'Cambio%': f'{self.cambio_percent:+.2f}%',
            'Volumen_M': f'{self.volumen/1e6:.2f}',
            'Vol_Relativo': f'{self.volumen_relativo:.2f}x',
            'ATR': f'${self.atr:.2f}',
            'ATR%': f'{self.atr_percent:.2f}%',
            'RSI': f'{self.rsi:.1f}',
            'Tendencia': self.tendencia,
            'Señal': self.señal,
            'Puntuación': self.puntuacion,
            'Entrada': f'${self.nivel_entrada:.2f}',
            'Stop_Loss': f'${self.stop_loss:.2f}',
            'TP1': f'${self.take_profit_1:.2f}',
            'TP2': f'${self.take_profit_2:.2f}',
            'R:R': f'1:{self.risk_reward:.1f}',
            'Breakout': 'SÍ' if self.breakout_confirmado else 'NO',
            'Detalles': self.detalles
        }

# ═══════════════════════════════════════════════════════════════════════════
# CLASE: ANALIZADOR DE INDICADORES TÉCNICOS
# ═══════════════════════════════════════════════════════════════════════════

class TechnicalAnalyzer:
    """Calculadora avanzada de indicadores técnicos para day trading."""
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calcula EMA (Exponential Moving Average)."""
        if len(prices) < period:
            return np.array([np.nan] * len(prices))
        
        ema = np.zeros(len(prices))
        ema[period-1] = np.mean(prices[:period])
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calcula ATR (Average True Range)."""
        if len(high) < period + 1:
            return 0.0
        
        tr_list = []
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        
        atr = np.mean(tr_list[-period:])
        return atr
    
    @staticmethod
    def calculate_macd(prices: np.ndarray) -> Tuple[float, float, float]:
        """Calcula MACD (Moving Average Convergence Divergence)."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = TechnicalAnalyzer.calculate_ema(prices, 12)
        ema_26 = TechnicalAnalyzer.calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        signal_line = TechnicalAnalyzer.calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])
    
    @staticmethod
    def detect_breakout(prices: np.ndarray, volumes: np.ndarray, lookback: int = 20) -> bool:
        """Detecta si hay un breakout confirmado."""
        if len(prices) < lookback + 5:
            return False
        
        current_price = prices[-1]
        recent_high = np.max(prices[-(lookback+1):-1])
        avg_volume = np.mean(volumes[-20:-1])
        current_volume = volumes[-1]
        
        # Breakout: precio supera máximo reciente con volumen alto
        price_breakout = current_price > recent_high * 1.01
        volume_confirmation = current_volume > avg_volume * 1.3
        
        return price_breakout and volume_confirmation

# ═══════════════════════════════════════════════════════════════════════════
# CLASE: SCANNER DE OPORTUNIDADES
# ═══════════════════════════════════════════════════════════════════════════

class DayTradingScanner:
    """Scanner profesional de oportunidades de day trading."""
    
    def __init__(self, config: ScannerConfig = ScannerConfig()):
        """Inicializa el scanner con configuración."""
        self.config = config
        self.analyzer = TechnicalAnalyzer()
        self.opportunities: List[DayTradingOpportunity] = []
        
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Descarga datos históricos del ticker."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days + 30)
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df is None or df.empty or len(df) < 30:
                return None
            
            df = df.dropna(subset=['Close', 'Volume'])
            df = df[df['Close'] > 0]
            
            if len(df) < 30:
                return None
                
            return df
            
        except Exception as e:
            logger.debug(f"Error descargando {ticker}: {e}")
            return None
    
    def analyze_ticker(self, ticker: str) -> Optional[DayTradingOpportunity]:
        """Analiza un ticker y determina si es oportunidad de day trading."""
        try:
            df = self.download_data(ticker)
            if df is None:
                return None
            
            # Extraer datos
            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values
            volumes = df['Volume'].values
            
            precio_actual = float(closes[-1])
            precio_anterior = float(closes[-2])
            cambio_percent = ((precio_actual - precio_anterior) / precio_anterior) * 100
            
            # FILTRO 1: Precio en rango óptimo
            if not (self.config.min_price <= precio_actual <= self.config.max_price):
                return None
            
            # FILTRO 2: Volumen mínimo
            volumen_actual = int(volumes[-1])
            volumen_promedio = int(np.mean(volumes[-20:]))
            
            if volumen_promedio < self.config.min_volume:
                return None
            
            # FILTRO 3: Volumen relativo
            volumen_relativo = volumen_actual / volumen_promedio if volumen_promedio > 0 else 0
            
            if volumen_relativo < self.config.min_relative_volume:
                return None
            
            # Calcular indicadores
            atr = self.analyzer.calculate_atr(highs, lows, closes, 14)
            atr_percent = (atr / precio_actual) * 100
            
            # FILTRO 4: ATR mínimo
            if atr_percent < self.config.min_atr_percent:
                return None
            
            rsi = self.analyzer.calculate_rsi(closes, 14)
            macd, macd_signal, macd_hist = self.analyzer.calculate_macd(closes)
            
            # EMA para tendencia
            ema_12 = self.analyzer.calculate_ema(closes, 12)
            ema_26 = self.analyzer.calculate_ema(closes, 26)
            
            # Determinar tendencia
            if precio_actual > ema_12[-1] > ema_26[-1]:
                tendencia = "ALCISTA"
            elif precio_actual < ema_12[-1] < ema_26[-1]:
                tendencia = "BAJISTA"
            else:
                tendencia = "NEUTRAL"
            
            # Detectar breakout
            breakout_confirmado = self.analyzer.detect_breakout(closes, volumes, 20)
            
            # ═══════════════════════════════════════════════════════════════
            # SISTEMA DE PUNTUACIÓN PONDERADA (0-100)
            # ═══════════════════════════════════════════════════════════════
            
            puntuacion = 0
            detalles = []
            
            # 1. PUNTUACIÓN POR VOLUMEN (25 puntos máx)
            volumen_score = 0
            if volumen_relativo >= 3.0:
                volumen_score = 25
                detalles.append("Vol Extremo 3x+")
            elif volumen_relativo >= 2.5:
                volumen_score = 20
                detalles.append("Vol Muy Alto 2.5x+")
            elif volumen_relativo >= 2.0:
                volumen_score = 15
                detalles.append("Vol Alto 2x+")
            elif volumen_relativo >= 1.5:
                volumen_score = 10
                detalles.append("Vol Bueno 1.5x+")
            
            puntuacion += volumen_score
            
            # 2. PUNTUACIÓN POR VOLATILIDAD ATR (20 puntos máx)
            volatilidad_score = 0
            if atr_percent >= 5.0:
                volatilidad_score = 20
                detalles.append("Volatilidad Extrema 5%+")
            elif atr_percent >= 4.0:
                volatilidad_score = 17
                detalles.append("Volatilidad Muy Alta 4%+")
            elif atr_percent >= 3.0:
                volatilidad_score = 14
                detalles.append("Volatilidad Alta 3%+")
            elif atr_percent >= 2.0:
                volatilidad_score = 10
                detalles.append("Volatilidad Media 2%+")
            
            puntuacion += volatilidad_score
            
            # 3. PUNTUACIÓN POR RSI (20 puntos máx)
            if 45 <= rsi <= 55:
                puntuacion += 20
                detalles.append("RSI Neutral Perfecto")
            elif 40 <= rsi <= 60:
                puntuacion += 15
                detalles.append("RSI Zona Operativa")
            elif 30 <= rsi < 40:
                puntuacion += 12
                detalles.append("RSI Sobreventa Leve")
            elif 60 < rsi <= 70:
                puntuacion += 10
                detalles.append("RSI Sobrecompra Leve")
            elif rsi < 30:
                puntuacion += 8
                detalles.append("RSI Sobreventa Extrema")
            elif rsi > 70:
                puntuacion += 5
                detalles.append("RSI Sobrecompra Extrema")
            
            # 4. PUNTUACIÓN POR MOMENTUM MACD (15 puntos máx)
            momentum_score = 0
            if macd_hist > 0 and macd > 0 and macd > macd_signal:
                momentum_score = 15
                detalles.append("MACD Alcista Fuerte")
            elif macd_hist > 0 and macd > macd_signal:
                momentum_score = 12
                detalles.append("MACD Momentum +")
            elif macd_hist < 0 and macd < 0 and macd < macd_signal:
                momentum_score = 10
                detalles.append("MACD Bajista Fuerte")
            elif macd_hist < 0:
                momentum_score = 7
                detalles.append("MACD Momentum -")
            else:
                momentum_score = 5
                detalles.append("MACD Neutral")
            
            puntuacion += momentum_score
            
            # 5. PUNTUACIÓN POR TENDENCIA (10 puntos máx)
            if tendencia == "ALCISTA":
                puntuacion += 10
                detalles.append("Tendencia Alcista")
            elif tendencia == "BAJISTA":
                puntuacion += 7
                detalles.append("Tendencia Bajista")
            else:
                puntuacion += 4
                detalles.append("Sin Tendencia Clara")
            
            # 6. BONUS POR BREAKOUT (10 puntos)
            if breakout_confirmado:
                puntuacion += 10
                detalles.append("🚀 BREAKOUT CONFIRMADO")
            
            # FILTRO FINAL: Puntuación mínima
            if puntuacion < self.config.min_score:
                return None
            
            # ═══════════════════════════════════════════════════════════════
            # CÁLCULO DE NIVELES DE TRADING
            # ═══════════════════════════════════════════════════════════════
            
            # Nivel de entrada (precio actual o ligeramente por encima)
            nivel_entrada = precio_actual
            
            # Stop loss basado en ATR
            stop_loss = precio_actual - (atr * 1.5)
            
            # Take profits escalonados
            take_profit_1 = precio_actual + (atr * 1.5)  # R:R 1:1
            take_profit_2 = precio_actual + (atr * 2.5)  # R:R 1:1.67
            
            # Risk/Reward ratio
            riesgo = precio_actual - stop_loss
            recompensa = take_profit_2 - precio_actual
            risk_reward = recompensa / riesgo if riesgo > 0 else 0
            
            # Señal de trading
            if puntuacion >= 85 and tendencia == "ALCISTA":
                señal = "🔥 COMPRA FUERTE"
            elif puntuacion >= 75:
                señal = "✅ COMPRA" if tendencia == "ALCISTA" else "🔻 VENTA"
            elif puntuacion >= 65:
                señal = "👁️ MONITOREAR"
            else:
                señal = "⚠️ PRECAUCIÓN"
            
            return DayTradingOpportunity(
                ticker=ticker,
                precio_actual=precio_actual,
                cambio_percent=cambio_percent,
                volumen=volumen_actual,
                volumen_promedio=volumen_promedio,
                volumen_relativo=volumen_relativo,
                atr=atr,
                atr_percent=atr_percent,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_hist=macd_hist,
                tendencia=tendencia,
                señal=señal,
                puntuacion=puntuacion,
                nivel_entrada=nivel_entrada,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                risk_reward=risk_reward,
                breakout_confirmado=breakout_confirmado,
                momentum_score=momentum_score,
                volatilidad_score=volatilidad_score,
                volumen_score=volumen_score,
                detalles=" | ".join(detalles)
            )
            
        except Exception as e:
            logger.debug(f"Error analizando {ticker}: {e}")
            return None
    
    def scan_market(self, tickers: List[str]) -> List[DayTradingOpportunity]:
        """Escanea múltiples tickers en paralelo."""
        logger.info(f"🔍 Iniciando escaneo de {len(tickers)} símbolos...")
        
        self.opportunities = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_ticker, ticker): ticker 
                               for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                completed += 1
                ticker = future_to_ticker[future]
                
                try:
                    opportunity = future.result()
                    if opportunity:
                        self.opportunities.append(opportunity)
                        logger.info(f"✅ {ticker}: Puntuación {opportunity.puntuacion} - {opportunity.señal}")
                except Exception as e:
                    logger.debug(f"❌ {ticker}: Error - {e}")
                
                # Mostrar progreso
                if completed % 5 == 0:
                    logger.info(f"📊 Progreso: {completed}/{len(tickers)} ({completed/len(tickers)*100:.0f}%)")
        
        # Ordenar por puntuación
        self.opportunities.sort(key=lambda x: x.puntuacion, reverse=True)
        
        logger.info(f"✨ Escaneo completado: {len(self.opportunities)} oportunidades encontradas")
        return self.opportunities

# ═══════════════════════════════════════════════════════════════════════════
# CLASE: VISUALIZADOR DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

class ResultsVisualizer:
    """Visualizador profesional de resultados."""
    
    @staticmethod
    def print_header():
        """Imprime encabezado profesional."""
        print("\n" + "═" * 100)
        print(Fore.CYAN + Style.BRIGHT + 
              " " * 25 + "🚀 SCANNER PROFESIONAL DE DAY TRADING 🚀")
        print(Fore.CYAN + " " * 30 + "Búsqueda Activa de Oportunidades")
        print("═" * 100 + Style.RESET_ALL)
    
    @staticmethod
    def print_summary(opportunities: List[DayTradingOpportunity], total_scanned: int):
        """Imprime resumen de estadísticas."""
        if not opportunities:
            print(Fore.YELLOW + "\n⚠️  No se encontraron oportunidades que cumplan los criterios.")
            return
        
        best = opportunities[0]
        avg_score = np.mean([o.puntuacion for o in opportunities])
        avg_atr = np.mean([o.atr_percent for o in opportunities])
        avg_vol_rel = np.mean([o.volumen_relativo for o in opportunities])
        
        breakouts = sum(1 for o in opportunities if o.breakout_confirmado)
        
        print(Fore.GREEN + "\n📊 RESUMEN DE ESCANEO:")
        print("─" * 100)
        print(f"  Total Escaneados:        {total_scanned}")
        print(f"  Oportunidades Encontradas: {Fore.YELLOW}{len(opportunities)}{Fore.GREEN}")
        print(f"  Mejor Puntuación:        {Fore.YELLOW}{best.puntuacion}{Fore.GREEN} ({best.ticker})")
        print(f"  Puntuación Promedio:     {avg_score:.1f}")
        print(f"  ATR Promedio:            {avg_atr:.2f}%")
        print(f"  Vol Relativo Promedio:   {avg_vol_rel:.2f}x")
        print(f"  Breakouts Confirmados:   {Fore.YELLOW}{breakouts}{Fore.GREEN}")
        print("─" * 100 + Style.RESET_ALL)
    
    @staticmethod
    def print_opportunities(opportunities: List[DayTradingOpportunity], top_n: int = 10):
        """Imprime las mejores oportunidades en formato tabla."""
        if not opportunities:
            return
        
        print(Fore.CYAN + f"\n🏆 TOP {min(top_n, len(opportunities))} OPORTUNIDADES DE DAY TRADING:")
        print("=" * 100 + Style.RESET_ALL)
        
        # Encabezado
        header = (
            f"{'#':<3} {'TICKER':<8} {'PRECIO':<10} {'CAMBIO%':<10} "
            f"{'VOL_REL':<9} {'ATR%':<8} {'RSI':<7} {'SEÑAL':<20} {'SCORE':<6}"
        )
        print(Fore.WHITE + Style.BRIGHT + header)
        print("─" * 100 + Style.RESET_ALL)
        
        # Filas
        for i, opp in enumerate(opportunities[:top_n], 1):
            # Colores según puntuación
            if opp.puntuacion >= 85:
                color = Fore.GREEN + Style.BRIGHT
            elif opp.puntuacion >= 75:
                color = Fore.GREEN
            elif opp.puntuacion >= 65:
                color = Fore.YELLOW
            else:
                color = Fore.WHITE
            
            cambio_color = Fore.GREEN if opp.cambio_percent >= 0 else Fore.RED
            
            row = (
                f"{i:<3} "
                f"{opp.ticker:<8} "
                f"${opp.precio_actual:<9.2f} "
                f"{cambio_color}{opp.cambio_percent:+9.2f}%{color} "
                f"{opp.volumen_relativo:<8.2f}x "
                f"{opp.atr_percent:<7.2f}% "
                f"{opp.rsi:<7.1f} "
                f"{opp.señal:<20} "
                f"{opp.puntuacion:<6}"
            )
            print(color + row + Style.RESET_ALL)
        
        print("=" * 100)
    
    @staticmethod
    def print_detailed_opportunity(opp: DayTradingOpportunity):
        """Imprime detalles completos de una oportunidad."""
        print("\n" + "╔" + "═" * 98 + "╗")
        print("║" + Fore.CYAN + Style.BRIGHT + f" ANÁLISIS DETALLADO: {opp.ticker}".center(98) + Style.RESET_ALL + "║")
        print("╠" + "═" * 98 + "╣")
        
        # Precio y cambio
        cambio_color = Fore.GREEN if opp.cambio_percent >= 0 else Fore.RED
        print(f"║  💰 Precio Actual:    ${opp.precio_actual:.2f}  "
              f"({cambio_color}{opp.cambio_percent:+.2f}%{Style.RESET_ALL})" + " " * 50 + "║")
        
        # Volumen
        print(f"║  📊 Volumen:          {opp.volumen:,} acciones  "
              f"({Fore.YELLOW}{opp.volumen_relativo:.2f}x promedio{Style.RESET_ALL})" + " " * 30 + "║")
        
        # Volatilidad
        print(f"║  📈 ATR:              ${opp.atr:.2f}  "
              f"({Fore.YELLOW}{opp.atr_percent:.2f}%{Style.RESET_ALL} del precio)" + " " * 40 + "║")
        
        # Indicadores
        print(f"║  🎯 RSI:              {opp.rsi:.1f}" + " " * 82 + "║")
        print(f"║  📉 MACD:             {opp.macd:.4f}  (Señal: {opp.macd_signal:.4f})" + " " * 50 + "║")
        print(f"║  📊 Tendencia:        {Fore.GREEN if opp.tendencia == 'ALCISTA' else Fore.RED}{opp.tendencia}{Style.RESET_ALL}" + " " * 80 + "║")
        
        # Breakout
        if opp.breakout_confirmado:
            print(f"║  🚀 Breakout:         {Fore.GREEN}CONFIRMADO ✓{Style.RESET_ALL}" + " " * 76 + "║")
        
        print("╠" + "═" * 98 + "╣")
        
        # Niveles de trading
        print(f"║  {Fore.CYAN}NIVELES DE TRADING:{Style.RESET_ALL}" + " " * 76 + "║")
        print(f"║    🎯 Entrada:        ${opp.nivel_entrada:.2f}" + " " * 73 + "║")
        print(f"║    🛑 Stop Loss:      {Fore.RED}${opp.stop_loss:.2f}{Style.RESET_ALL}  "
              f"(Riesgo: ${opp.nivel_entrada - opp.stop_loss:.2f})" + " " * 40 + "║")
        print(f"║    ✅ Take Profit 1:  {Fore.GREEN}${opp.take_profit_1:.2f}{Style.RESET_ALL}" + " " * 72 + "║")
        print(f"║    🎁 Take Profit 2:  {Fore.GREEN}${opp.take_profit_2:.2f}{Style.RESET_ALL}" + " " * 72 + "║")
        print(f"║    ⚖️  Risk/Reward:    1:{opp.risk_reward:.2f}" + " " * 75 + "║")
        
        print("╠" + "═" * 98 + "╣")
        
        # Señal y puntuación
        print(f"║  🎯 SEÑAL:            {opp.señal}" + " " * 77 + "║")
        
        score_color = (Fore.GREEN + Style.BRIGHT if opp.puntuacion >= 85 
                      else Fore.GREEN if opp.puntuacion >= 75 
                      else Fore.YELLOW)
        print(f"║  ⭐ PUNTUACIÓN:       {score_color}{opp.puntuacion}/100{Style.RESET_ALL}" + " " * 78 + "║")
        
        # Detalles
        if opp.detalles:
            print("╠" + "═" * 98 + "╣")
            print(f"║  📝 Detalles: {opp.detalles[:85]}" + " " * (13 + 85 - len(opp.detalles[:85])) + "║")
        
        print("╚" + "═" * 98 + "╝")

# ═══════════════════════════════════════════════════════════════════════════
# CLASE: EXPORTADOR DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

class ResultsExporter:
    """Exportador de resultados a diferentes formatos."""
    
    @staticmethod
    def export_to_csv(opportunities: List[DayTradingOpportunity], filename: str = None):
        """Exporta resultados a CSV."""
        if not opportunities:
            logger.warning("No hay oportunidades para exportar")
            return
        
        if filename is None:
            filename = f'day_trading_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        df = pd.DataFrame([opp.to_dict() for opp in opportunities])
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"💾 Resultados exportados a: {filename}")
    
    @staticmethod
    def export_to_html(opportunities: List[DayTradingOpportunity], filename: str = None):
        """Exporta resultados a HTML."""
        if not opportunities:
            return
        
        if filename is None:
            filename = f'day_trading_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte Day Trading - {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #0a0e27; color: #e8eaf6; }}
                h1 {{ color: #00d4ff; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ background: #151b3d; color: #00d4ff; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #333; }}
                tr:hover {{ background: rgba(0, 212, 255, 0.1); }}
                .score-high {{ color: #00ff88; font-weight: bold; }}
                .score-medium {{ color: #ffd700; }}
                .positive {{ color: #00ff88; }}
                .negative {{ color: #ff3366; }}
            </style>
        </head>
        <body>
            <h1>🚀 Reporte de Oportunidades Day Trading</h1>
            <p style="text-align: center;">Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <table>
                <tr>
                    <th>#</th>
                    <th>Ticker</th>
                    <th>Precio</th>
                    <th>Cambio %</th>
                    <th>Vol Relativo</th>
                    <th>ATR %</th>
                    <th>RSI</th>
                    <th>Señal</th>
                    <th>Puntuación</th>
                    <th>Entrada</th>
                    <th>Stop Loss</th>
                    <th>TP1</th>
                </tr>
        """
        
        for i, opp in enumerate(opportunities, 1):
            score_class = 'score-high' if opp.puntuacion >= 75 else 'score-medium'
            cambio_class = 'positive' if opp.cambio_percent >= 0 else 'negative'
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{opp.ticker}</strong></td>
                    <td>${opp.precio_actual:.2f}</td>
                    <td class="{cambio_class}">{opp.cambio_percent:+.2f}%</td>
                    <td>{opp.volumen_relativo:.2f}x</td>
                    <td>{opp.atr_percent:.2f}%</td>
                    <td>{opp.rsi:.1f}</td>
                    <td>{opp.señal}</td>
                    <td class="{score_class}">{opp.puntuacion}</td>
                    <td>${opp.nivel_entrada:.2f}</td>
                    <td>${opp.stop_loss:.2f}</td>
                    <td>${opp.take_profit_1:.2f}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"📄 Reporte HTML exportado a: {filename}")

# ═══════════════════════════════════════════════════════════════════════════
# LISTAS DE TICKERS PREDEFINIDAS
# ═══════════════════════════════════════════════════════════════════════════

TICKERS_POPULARES_DAY_TRADING = [
    # Mega caps con alta liquidez
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META',
    
    # Tech alta volatilidad
    'AMD', 'INTC', 'NFLX', 'AVGO', 'QCOM', 'ADBE', 'CRM',
    
    # Momentum stocks
    'PLTR', 'COIN', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    
    # Meme stocks / Alta actividad retail
    'GME', 'AMC', 'BBBY', 'BB', 'NOK',
    
    # Cripto-relacionadas
    'MARA', 'RIOT', 'MSTR', 'CLSK', 'CIFR',
    
    # Semiconductores
    'SMCI', 'MU', 'AMAT', 'LRCX', 'KLAC',
    
    # EVs
    'F', 'GM', 'LUCID',
    
    # Bio/Pharma volátil
    'MRNA', 'BNTX', 'PFE', 'JNJ',
    
    # Financieras
    'JPM', 'BAC', 'GS', 'MS', 'C',
    
    # Energía
    'XOM', 'CVX', 'COP', 'SLB',
    
    # Retail/Consumer
    'WMT', 'TGT', 'COST', 'HD', 'LOW',
    
    # Streaming/Media
    'DIS', 'PARA', 'WBD',
    
    # Software/Cloud
    'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS',
    
    # ETFs populares day trading
    'SPY', 'QQQ', 'IWM', 'DIA', 'SQQQ', 'TQQQ', 'UVXY'
]

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Función principal del scanner."""
    
    visualizer = ResultsVisualizer()
    visualizer.print_header()
    
    print(Fore.YELLOW + "\n⚙️  CONFIGURACIÓN DEL SCANNER:")
    print("─" * 100 + Style.RESET_ALL)
    
    # Configuración
    config = ScannerConfig()
    
    print(f"  Precio mínimo:        ${config.min_price}")
    print(f"  Precio máximo:        ${config.max_price}")
    print(f"  Volumen mínimo:       {config.min_volume:,} acciones")
    print(f"  Vol. relativo mín:    {config.min_relative_volume}x")
    print(f"  ATR% mínimo:          {config.min_atr_percent}%")
    print(f"  Puntuación mínima:    {config.min_score}/100")
    
    # Selección de tickers
    print(Fore.CYAN + "\n📋 OPCIONES DE ESCANEO:")
    print("  1. Lista predefinida (75 símbolos populares day trading)")
    print("  2. Ingresar símbolos personalizados")
    print("  3. Top 20 más activos" + Style.RESET_ALL)
    
    opcion = input("\nSelecciona opción (1/2/3) [1]: ").strip() or "1"
    
    if opcion == "1":
        tickers = TICKERS_POPULARES_DAY_TRADING
        print(Fore.GREEN + f"✓ Usando lista predefinida: {len(tickers)} símbolos")
    elif opcion == "2":
        tickers_input = input("\nIngresa símbolos separados por comas: ").strip()
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        print(Fore.GREEN + f"✓ {len(tickers)} símbolos personalizados")
    else:
        tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'PLTR', 'META', 'GOOGL', 'MSFT',
                  'COIN', 'MARA', 'RIOT', 'SMCI', 'RIVN', 'LCID', 'NIO', 'SPY',
                  'QQQ', 'SQQQ', 'TQQQ', 'UVXY']
        print(Fore.GREEN + f"✓ Top 20 más activos seleccionados")
    
    print(Style.RESET_ALL)
    
    # Crear scanner y escanear
    scanner = DayTradingScanner(config)
    opportunities = scanner.scan_market(tickers)
    
    # Mostrar resultados
    visualizer.print_summary(opportunities, len(tickers))
    visualizer.print_opportunities(opportunities, top_n=15)
    
    # Mostrar detalles de las mejores
    if opportunities:
        print(Fore.CYAN + "\n📊 ANÁLISIS DETALLADO DE LAS MEJORES OPORTUNIDADES:" + Style.RESET_ALL)
        for opp in opportunities[:3]:
            visualizer.print_detailed_opportunity(opp)
    
    # Exportar resultados
    if opportunities:
        print(Fore.YELLOW + "\n💾 EXPORTAR RESULTADOS:")
        exportar = input("¿Deseas exportar los resultados? (s/n) [s]: ").strip().lower() or 's'
        
        if exportar == 's':
            ResultsExporter.export_to_csv(opportunities)
            ResultsExporter.export_to_html(opportunities)
    
    # Footer
    print("\n" + "═" * 100)
    print(Fore.CYAN + Style.BRIGHT + " " * 35 + "✅ ESCANEO COMPLETADO")
    print("═" * 100 + Style.RESET_ALL)
    print(Fore.YELLOW + "\n⚠️  DISCLAIMER: Este sistema es para fines educativos.")
    print("   Los resultados no constituyen asesoría financiera.")
    print("   Opere bajo su propio riesgo y responsabilidad.\n" + Style.RESET_ALL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\n\n⚠️  Escaneo interrumpido por el usuario" + Style.RESET_ALL)
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
