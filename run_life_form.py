# run_life_form.py
# デジタル生命体 起動スクリプト
# 概要：DigitalLifeFormインスタンスを生成し、その活動を開始・停止する。
import time
from snn_research.agent.digital_life_form import DigitalLifeForm

def main():
    """
    デジタル生命体を起動し、指定時間（または無限に）活動させる。
    """
    life_form = DigitalLifeForm()
    
    try:
        life_form.start()
        # ここではデモのために60秒後に停止する
        # 実際の運用では、外部からのシャットダウン信号を受け取るまでループさせる
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down.")
    finally:
        life_form.stop()
        print("DigitalLifeForm has been deactivated.")

if __name__ == "__main__":
    main()