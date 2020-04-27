from easydict import EasyDict as edict

RL_CONFIGS = {
    "pendulum":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/pendulum/1nd/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/pendulum/1nd/policies/",
        "hidden_size": [128, 128],

        "gamma":0.99,
        "lamda":0.98,
        "actor_lr":1e-4,
        "critic_lr":1e-3,
        "clip_param":0.2,

        "model_update_num":10,
        "max_iter":1000,
        "batch_size":400,
        "total_sample_size":4000
    }),
    "acrobat":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/acrobat/1st/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/pendulum/1st/policies/",
        "hidden_size": [128, 128],

        "gamma":0.99,
        "lamda":0.98,
        "actor_lr":1e-4,
        "critic_lr":1e-3,
        "clip_param":0.2,

        "model_update_num":10,
        "max_iter":1000,
        "batch_size":400,
        "total_sample_size":4000
    })
}
