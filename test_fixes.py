#!/usr/bin/env python3
"""
Test script: R:R fix ve hayalet sinyal kapatma doğrulaması
"""
import asyncio
import sys
from core.types import OracleState, PipelineStatus
from core.graph import compile_oracle_graph
from core.console import agent_print, system_print
from core.config import load_oracle_config

async def main():
    conf = await load_oracle_config()
    graph = compile_oracle_graph()
    
    print("\n" + "="*80)
    print("TEST 1: BTC Sinyal Oluşturma (Skor ve R:R Doğrulaması)")
    print("="*80)
    
    state = OracleState(
        symbol="BTC/USDT",
        timeframe="4h",
        direction="",
        status=PipelineStatus.IDLE,
    )
    
    try:
        result = await graph.ainvoke({"state": state})
        final_state = result.get("state")
        
        print(f"\n[SONUÇ] Symbol: {final_state.symbol}")
        print(f"[SONUÇ] Status: {final_state.status}")
        print(f"[SONUÇ] Signal Label: {final_state.signal_label}")
        print(f"[SONUÇ] Signal Direction: {final_state.signal_direction}")
        print(f"[SONUÇ] Composite Score: {final_state.composite_score}")
        print(f"[SONUÇ] CEO Approved: {final_state.ceo_approved}")
        
        if final_state.signal_label:
            print(f"\n✓ SINYAL BAŞARILI: {final_state.signal_label}")
            print(f"  - Base R:R: {final_state.base_rr}")
            print(f"  - Entry: {final_state.entry}")
            print(f"  - SL: {final_state.sl}")
            print(f"  - T1: {final_state.t1}")
            print(f"  - Confidence: {final_state.confidence}")
            
            # R:R kontrol
            if final_state.base_rr >= 3.0:
                print(f"  ✓ R:R BAŞARILI: {final_state.base_rr} >= 3.0")
            else:
                print(f"  ✗ R:R BAŞARILI DEĞİL: {final_state.base_rr} < 3.0")
        else:
            print(f"\n✗ SINYAL BAŞARILI DEĞİL (İptal edilen)")
            print(f"  Fatal Error: {final_state.fatal_error}")
            
            # Hayalet sinyal testi
            if final_state.status == PipelineStatus.ABORTED:
                if final_state.signal_label is None:
                    print(f"  ✓ HAYALET SINYAL KAPATILDI: signal_label = None")
                else:
                    print(f"  ✗ HAYALET SINYAL: signal_label = {final_state.signal_label}")
    
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST TAMAMLANDI")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
