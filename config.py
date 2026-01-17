from dataclasses import dataclass

@dataclass
class cfg:

    run_readme: str = "decreases distance to energy minimum"
    # Datagen
    N: int = 32
    dim: int = 2
    box_size: float = 10.0
    dt: float = 1e-3
    num_steps: int = 15_000
    train_fname: str = "train_states.npz"
    val_fname: str = "val_states.npz"

    seed_train: int = 0
    seed_val: int = 68


    mass: float = 1.0
    kT: float = 10.0
    sigma: float = 0.5
    epsilon: float = 1.0

    # Training
    lr: float = 5e-5
    batch_size: int = 1000
    mask_ratio: float = 0.25
    ema_decay: float = 0.996
    
    num_epochs: int = 30

    save_every: int = 10

    load_epoch: int = 20


@dataclass
class contextcfg:
    in_feat_dim: int = 4
    h_dim: int = 96
    msg_dim: int = 96
    hidden_dim: int = 192
    n_layers: int = 2
    mlp_depth: int = 2
    out_dim: int = 96

@dataclass
class predcfg:
    in_feat_dim: int = 96 #same as outdim
    h_dim: int = 96 #same as teacher/stu
    msg_dim: int = 96
    hidden_dim: int = 192
    n_layers: int = 1
    mlp_depth: int = 2
    out_dim: int = 96