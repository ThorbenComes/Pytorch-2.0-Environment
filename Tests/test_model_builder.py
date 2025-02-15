import utils.model_builder
import utils.visualization.model_layout_visualizer


"""manual test class for model_builder"""


config_transformer = [
    ("embedding_encoder", ["observation"], ["encoder_embedded"], None),
    ("embedding_decoder", ["observation"], ["decoder_embedded"], None),
    ("encoder", ["encoder_embedded"], ["encoder_output"], None),
    ("decoder", ["encoder_output", "decoder_embedded"], ["decoder_out"], None),
    ("Linear_Softmax", ["decoder_out"], ["out"], None)
]

config_mts3_4lev_simple = [
    ("embedding_encoder", ["observation"], ["manager_embedded"], None),
    ("embedding_encoder", ["observation"], ["submanager1_embedded"], None),
    ("embedding_encoder", ["observation"], ["submanager2_embedded"], None),
    ("embedding_encoder", ["observation"], ["worker_embedded"], None),
    ("manager", ["manager_embedded"], ["manager_out"], None),
    ("submanager", ["submanager1_embedded", "manager_out"], ["submanager1_out"], None),
    ("submanager", ["submanager2_embedded", "submanager1_out"], ["submanager2_out"], None),
    ("worker", ["submanager2_out", "worker_embedded"], ["worker_out"], None),
    ("decoder", ["worker_out"], ["out"], None),
]

config_mts3_4lev_act = [
    ("embedding_encoder", ["observation", "action"], ["manager_embedded"], None),
    ("embedding_encoder", ["observation", "action"], ["submanager1_embedded"], None),
    ("embedding_encoder", ["observation", "action"], ["submanager2_embedded"], None),
    ("embedding_encoder", ["observation", "action"], ["worker_embedded"], None),
    ("manager", ["manager_embedded"], ["manager_out", "out_top"], None),
    ("submanager", ["submanager1_embedded", "manager_out"], ["submanager1_out"], None),
    ("submanager", ["submanager2_embedded", "submanager1_out"], ["submanager2_out"], None),
    ("worker", ["submanager2_out", "worker_embedded"], ["worker_out"], None),
    ("decoder", ["worker_out"], ["out"], None),
]

source = ["observation", "action"]
target = ["out", "out_top"]

blocks, graph = utils.model_builder.build_model_graph(source, config_mts3_4lev_act, target)
utils.visualization.model_layout_visualizer.create_view(graph, blocks)
