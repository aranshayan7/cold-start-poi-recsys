import torch

CONFIG = {
    # JAPAN
    # 'processed_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\processed_japan_v2',
    'processed_dir': r'/Volumes/Extreme SSD/Model/Tourism_ProtoNet/processed_japan_v2',
    # 'checkpoint_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\checkpoints_protonet',
    'checkpoint_dir' :r'/Volumes/Extreme SSD/Model/Tourism_ProtoNet/checkpoints_protonet',
    # 'log_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\logs_protonet',
    'log_dir':r'/Volumes/Extreme SSD/Model/Tourism_ProtoNet/logs_protonet',
    # ///// ASIA
    # 'processed_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\processed_asia_v1',
    # 'processed_dir': r'/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet/processed_asia_v1',
    # 'checkpoint_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\checkpoints_asia',
    # 'checkpoint_dir' :r'/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet/checkpoints_asia',
    # 'log_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\logs_asia',
    # 'log_dir':r'/Users/aran/Downloads/Proposal/Model/Tourism_ProtoNet/logs_asia',

    # # ///// FULL
    # 'processed_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\processed',
    # 'processed_dir': r'/Volumes/Extreme SSD/Model/Tourism_ProtoNet/processed',
    # 'checkpoint_dir': r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\checkpoints_full',
    # 'checkpoint_dir': r'/Volumes/Extreme SSD/Model/Tourism_ProtoNet/checkpoints_full',
    # 'log_dir':        r'C:\Users\niloz\PycharmProjects\Tourism_ProtoNet\logs_full',
    # 'log_dir': r'/Volumes/Extreme SSD/Model/Tourism_ProtoNet/logs_full',

    # Model
    'embed_dim': 64,
    'dropout': 0.5,
    'max_vocab_size': 100_000,      # ← این خط اضافه شد
    'temperature': 0.05,

    # Data
    'support_k': 15,
    'query_size': 15,

    # Training
    'lr': 0.001,
    'weight_decay': 2e-4,
    'clip_norm': 1.0,
    'batch_size': 4,

    'n_epochs': 30,
    'patience': 6,
    'eval_every': 1,
    'top_k_list': [10, 20],

    'seed': 42,
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'

}