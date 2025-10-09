# matsushibadenki/snn3/SNN3-main/run_life_form.py
# デジタル生命体 起動スクリプト
# 概要：DigitalLifeFormインスタンスを生成し、その活動を開始・停止する。
# 修正点: DIコンテナを利用して依存関係を構築するように修正。

import time
import argparse
from app.containers import AgentContainer
from snn_research.agent.digital_life_form import DigitalLifeForm

def main():
    """
    デジタル生命体を起動し、指定時間（または無限に）活動させる。
    """
    parser = argparse.ArgumentParser(description="Digital Life Form Orchestrator")
    parser.add_argument("--duration", type=int, default=60, help="実行時間（秒）。0を指定すると無限に実行します。")
    args = parser.parse_args()

    # --- ◾️◾️◾️◾️◾️↓修正↓◾️◾️◾️◾️◾️ ---
    # DIコンテナを初期化し、依存関係が注入済みのDigitalLifeFormを取得する
    # DigitalLifeFormの__init__をリファクタリングして、コンテナから注入できるようにする必要がある
    # ここでは、DigitalLifeFormの内部でコンテナを使う形に修正するのが現実的
    
    print("Initializing Digital Life Form with dependencies...")
    life_form = DigitalLifeForm() # DigitalLifeForm内部でコンテナが使われる想定
    # --- ◾️◾️◾️◾️◾️↑修正↑◾️◾️◾️◾️◾️ ---
    
    try:
        life_form.start()
        
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down.")
    finally:
        life_form.stop()
        print("DigitalLifeForm has been deactivated.")

if __name__ == "__main__":
    main()
