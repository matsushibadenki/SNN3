# ファイルパス: snn_research/hardware/compiler.py
# (新規作成)
#
# Title: Neuromorphic Hardware Compiler
#
# Description:
# - ロードマップ「ニューロモーフィックハードウェアへの最適化」を実装するコンポーネント。
# - 学習済みのSNNモデル（特にBioSNN）のアーキテクチャを解析し、
#   Intel Loihiのようなニューロモーフィックチップで実行するための
#   ハードウェア構成を生成する。
# - この実装では、実際のSDKの代わりに、ハードウェアへのマッピングを記述した
#   設定辞書を生成することで、コンパイルプロセスをシミュレートする。

from typing import Dict, Any, List
import yaml

from snn_research.bio_models.simple_network import BioSNN
from snn_research.hardware.profiles import get_hardware_profile

class NeuromorphicCompiler:
    """
    SNNモデルをニューロモーフィックハードウェア用の構成にコンパイルする。
    """
    def __init__(self, hardware_profile_name: str = "default"):
        """
        Args:
            hardware_profile_name (str): 'profiles.py'で定義されたハードウェアプロファイル名。
        """
        self.hardware_profile = get_hardware_profile(hardware_profile_name)
        print(f"🔩 ニューロモーフィック・コンパイラが初期化されました (ターゲット: {self.hardware_profile['name']})。")

    def compile(self, model: BioSNN, output_path: str):
        """
        BioSNNモデルを解析し、ハードウェア構成ファイルを生成する。

        Args:
            model (BioSNN): コンパイル対象の学習済みSNNモデル。
            output_path (str): 生成されたハードウェア構成ファイルの保存先 (YAML形式)。
        """
        print(f"⚙️ モデル '{type(model).__name__}' のコンパイルを開始...")

        hardware_config: Dict[str, Any] = {
            "target_hardware": self.hardware_profile['name'],
            "neuron_cores": [],
            "synaptic_connectivity": []
        }

        # 1. ニューロンのマッピング (Neuron Core Mapping)
        #    各層のニューロンを、ハードウェア上の計算コアに割り当てる。
        neuron_offset = 0
        for i, layer in enumerate(model.layers):
            core_config = {
                "core_id": i,
                "neuron_type": type(layer).__name__,
                "num_neurons": layer.n_neurons,
                "neuron_ids": list(range(neuron_offset, neuron_offset + layer.n_neurons)),
                # ここではLIFニューロンのパラメータを例としてマッピング
                "parameters": {
                    "tau_mem": layer.tau_mem,
                    "v_threshold": layer.v_thresh,
                }
            }
            hardware_config["neuron_cores"].append(core_config)
            neuron_offset += layer.n_neurons
        
        print(f"  - {len(model.layers)}個のニューロン層を{len(model.layers)}個のコアにマッピングしました。")

        # 2. シナプスのマッピング (Synaptic Connectivity)
        #    層間の結合重みを、ハードウェアの接続情報に変換する。
        for i, weight_matrix in enumerate(model.weights):
            pre_core = hardware_config["neuron_cores"][i-1] if i > 0 else {"neuron_ids": list(range(model.layer_sizes[0]))}
            post_core = hardware_config["neuron_cores"][i]

            # ゼロでない重みのみを接続として記録 (スパース表現)
            connections = []
            for pre_id_local, pre_neuron_id in enumerate(pre_core["neuron_ids"]):
                for post_id_local, post_neuron_id in enumerate(post_core["neuron_ids"]):
                    weight = weight_matrix[post_id_local, pre_id_local].item()
                    if weight > 0:
                        connections.append({
                            "source_neuron": pre_neuron_id,
                            "target_neuron": post_neuron_id,
                            "weight": round(weight, 4),
                            "delay": 1 # ここでは遅延を1ステップに固定
                        })
            
            hardware_config["synaptic_connectivity"].append({
                "source_core": i - 1 if i > 0 else "input",
                "target_core": i,
                "num_connections": len(connections),
                "connections": connections
            })
        
        print(f"  - {len(model.weights)}個のシナプス接続をマッピングしました。")

        # 3. 設定ファイルの保存
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(hardware_config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ コンパイル完了。ハードウェア構成を '{output_path}' に保存しました。")