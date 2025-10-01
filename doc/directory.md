# **ディレクトリ構成**

snn2/
├── configs/
│   └── base_config.yaml          # 🖍️ 変更: 学習パラダイムを追加
├── snn_research/
│   ├── core/
│   │   └── snn_core.py
│   ├── training/
│   │   ├── trainers.py
│   │   └── bio_trainer.py          # ✨ 新規: 生物学的学習則用のトレーナー
│   ├── learning_rules/             # ✨ 新規: 学習ルールを格納するディレクトリ
│   │   ├── __init__.py
│   │   ├── base_rule.py
│   │   ├── stdp.py