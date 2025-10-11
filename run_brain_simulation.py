# ファイルパス: run_brain_simulation.py
# (更新)
# 修正: ArtificialBrainの新しい依存関係（HierarchicalPlannerなど）を
#       すべてインスタンス化して正しく注入するように修正。

import sys
from pathlib import Path
import time

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent))

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
# IO and encoding
from snn_research.io.sensory_receptor import SensoryReceptor
from snn_research.io.spike_encoder import SpikeEncoder
from snn_research.io.actuator import Actuator
# Core cognitive modules
from snn_research.cognitive_architecture.perception_cortex import PerceptionCortex
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
# Memory systems
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.rag_snn import RAGSystem
from snn_research.distillation.model_registry import SimpleModelRegistry
# Value and action selection
from snn_research.cognitive_architecture.amygdala import Amygdala
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
# Motor control
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.motor_cortex import MotorCortex


def main():
    """
    人工脳の全コンポーネントを初期化し、シミュレーションを実行する。
    """
    # 依存関係の深いコンポーネントを先に初期化
    num_neurons = 256
    model_registry = SimpleModelRegistry()
    rag_system = RAGSystem()

    # 各コンポーネントの初期化
    receptor = SensoryReceptor()
    encoder = SpikeEncoder(num_neurons=num_neurons)
    perception = PerceptionCortex(num_neurons=num_neurons, feature_dim=64)
    hippocampus = Hippocampus(capacity=20)
    cortex = Cortex()
    amygdala = Amygdala()
    pfc = PrefrontalCortex()
    # PlannerはModelRegistryとRAGSystemに依存
    planner = HierarchicalPlanner(model_registry=model_registry, rag_system=rag_system)
    basal_ganglia = BasalGanglia()
    cerebellum = Cerebellum()
    motor_cortex = MotorCortex(actuators=['voice_synthesizer'])
    actuator = Actuator(actuator_name='voice_synthesizer')

    # 人工脳の組み立て
    brain = ArtificialBrain(
        sensory_receptor=receptor,
        spike_encoder=encoder,
        actuator=actuator,
        perception_cortex=perception,
        prefrontal_cortex=pfc,
        hierarchical_planner=planner,
        hippocampus=hippocampus,
        cortex=cortex,
        amygdala=amygdala,
        basal_ganglia=basal_ganglia,
        cerebellum=cerebellum,
        motor_cortex=motor_cortex
    )

    # シミュレーションの実行
    inputs = [
        "素晴らしい発見だ！これは成功に繋がるだろう。",
        "エラーが発生しました。システムに問題があるようです。",
        "今日は穏やかな一日だ。"
    ]

    for text_input in inputs:
        brain.run_cognitive_cycle(text_input)
        time.sleep(1) # 各サイクルの間に少し待機

if __name__ == "__main__":
    main()
