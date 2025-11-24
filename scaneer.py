"""
Day Trading Stock Screener - VERSIÓN MEJORADA
Analiza las mejores acciones para day trading con manejo robusto de errores
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import argparse
import sys
import time

def get_top_stocks(num_stocks=50):
    """
    Obtiene las mejores acciones para day trading
    """

    # Lista expandida de acciones populares
    popular_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
        'CRM', 'ORCL', 'ADBE', 'INTC', 'QCOM', 'TXN', 'AVGO', 'CSCO',
        'IBM', 'SNOW', 'PLTR', 'NET', 'CRWD', 'ZS', 'DDOG', 'MDB',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP',
        'V', 'MA', 'PYPL', 'SQ', 'COIN', 'SOFI', 'HOOD',
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'AMGN',
        'GILD', 'BIIB', 'VRTX', 'REGN', 'MRNA', 'BNTX',
        'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'DIS', 'NFLX',
        'COST', 'TGT', 'LOW', 'TJX', 'ROST', 'ULTA',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX', 'DE',
        'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV',
        'TSM', 'ASML', 'AMAT', 'LRCX', 'KLAC', 'MU', 'NXPI', 'MCHP',
        'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR',
        'EBAY', 'ETSY', 'W', 'CHWY', 'DASH', 'UBER', 'LYFT',
        'OKTA', 'PANW', 'FTNT', 'DBX', 'BOX',
        'ON', 'SWKS', 'MPWR', 'MRVL', 'ADI',
        'EA', 'TTWO', 'RBLX', 'DKNG', 'PENN',
        'ENPH', 'SEDG', 'RUN', 'PLUG', 'FCEL', 'BE', 'BLNK', 'CHPT',
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'SPG',
        'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'STLD',
        'PG', 'KO', 'PEP', 'MDLZ', 'PM', 'MO', 'CL', 'KMB',
        'AAL', 'DAL', 'UAL', 'LUV', 'JBLU', 'ALK', 'CCL', 'RCL', 'NCLH',
        'USB', 'PNC', 'TFC', 'COF', 'FITB', 'KEY', 'RF', 'CFG',
        'GME', 'AMC', 'BB', 'CLOV', 'SPCE',
        'HAL', 'BKR', 'NOV', 'FTI', 'RIG',
        'BMY', 'NVO', 'AZN', 'SNY', 'TEVA',
        'ANET', 'WDAY', 'NOW', 'TEAM', 'ZM', 'DOCU', 'TWLO',
        'MAR', 'HLT', 'EXPE', 'BKNG', 'YUM', 'CMG', 'DPZ',
        'PGR', 'ALL', 'TRV', 'MET', 'PRU', 'AIG',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EEM', 'GLD', 'SLV',
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU',
    ]

    print(f"\n🔍 Analizando {len(popular_tickers)} acciones...")
    print("⏳ Obteniendo datos del mercado...\n")

    stock_data = []
    errors = 0
    total = len(popular_tickers)

    # Descargar datos en lotes para mejor rendimiento
    for i in range(0, len(popular_tickers), 10):
        batch = popular_tickers[i:i+10]
        try:
            # Descargar batch completo
            data = yf.download(batch, period='5d', interval='1d', 
                             progress=False, group_by='ticker', threads=True)

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        hist = data
                    else:
                        if ticker not in data.columns.levels[0]:
                            continue
                        hist = data[ticker]

                    if hist.empty or len(hist) < 2:
                        continue

                    # Obtener info básica
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    volume = info.get('volume', 0) or info.get('averageVolume', 0) or hist['Volume'].iloc[-1]
                    price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or hist['Close'].iloc[-1]

                    if volume == 0 or price == 0:
                        continue

                    # Calcular volatilidad
                    if 'High' in hist.columns and 'Low' in hist.columns:
                        daily_range = ((hist['High'] - hist['Low']) / hist['Low'] * 100).mean()
                    else:
                        daily_range = 0

                    # Cambio porcentual
                    pct_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

                    # Score: volumen * volatilidad
                    score = (volume / 1_000_000) * abs(daily_range) * 0.5

                    stock_data.append({
                        'Ticker': ticker,
                        'Precio': round(float(price), 2),
                        'Volumen': int(volume),
                        'Volatilidad_%': round(float(daily_range), 2),
                        'Cambio_5d_%': round(float(pct_change), 2),
                        'Score': round(float(score), 2)
                    })

                except Exception as e:
                    errors += 1
                    continue

            # Mostrar progreso
            progress = min(i + 10, total)
            print(f"   Progreso: {progress}/{total} ({progress*100//total}%) - Válidas: {len(stock_data)}")

            # Pequeña pausa para evitar rate limiting
            time.sleep(0.5)

        except Exception as e:
            errors += 1
            continue

    print(f"\n✅ Análisis completado")
    print(f"   • Acciones válidas: {len(stock_data)}")
    print(f"   • Errores: {errors}\n")

    if len(stock_data) == 0:
        print("❌ ERROR: No se pudieron obtener datos. Verifica tu conexión a internet.")
        sys.exit(1)

    # Crear DataFrame y ordenar
    df = pd.DataFrame(stock_data)
    df = df.sort_values('Score', ascending=False).head(num_stocks)
    df = df.reset_index(drop=True)
    df.index += 1

    return df

def main():
    parser = argparse.ArgumentParser(description='Day Trading Stock Screener')
    parser.add_argument('-n', '--num', type=int, default=50,
                       help='Número de acciones (default: 50)')
    parser.add_argument('-o', '--output', type=str, default='top_stocks.csv',
                       help='Archivo CSV (default: top_stocks.csv)')

    args = parser.parse_args()

    print("=" * 70)
    print("📊 DAY TRADING STOCK SCREENER")
    print("=" * 70)

    # Obtener datos
    df = get_top_stocks(args.num)

    # Crear lista separada por comas
    tickers_list = ','.join(df['Ticker'].tolist())

    # Guardar CSV solo con tickers separados por comas (sin headers ni datos extra)
    with open(args.output, 'w') as f:
        f.write(tickers_list)

    # Mostrar tabla en terminal
    print(f"🏆 TOP {len(df)} ACCIONES PARA DAY TRADING\n")
    print(df.to_string())

    # Mostrar lista separada por comas
    print(f"\n\n📋 LISTA SEPARADA POR COMAS:\n")
    print(tickers_list)

    print(f"\n\n💾 Archivo guardado: {args.output}")
    print(f"   Contenido: {tickers_list[:50]}...")
    print("=" * 70)

if __name__ == "__main__":
    main()
