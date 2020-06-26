from easydict import EasyDict as edict

PPO = {
    "pendulum":
    edict({
        "gpu":False,
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
        "total_sample_size":4000,
        "test_iter":1000
    }),
    "quadruped":
    edict({
        "gpu":True,
        "render":True,
        "log_dir":"./expData/quadruped/1nd/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/quadruped/1nd/policies/",
        "hidden_size": [128, 128],

        "gamma":0.99,
        "lamda":0.98,
        "actor_lr":1e-4,
        "critic_lr":1e-3,
        "clip_param":0.2,

        "model_update_num":10,
        "max_iter":1000,
        "batch_size":400,
        "total_sample_size":4000,
        "test_iter":1000
    }),

    "Pendulum-v0":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/Pendulum-v0/2nd/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/Pendulum-v0/2nd/policies/",
        "hidden_size": [128, 128],

        "gamma":0.99,
        "lamda":0.98,
        "actor_lr":1e-4,
        "critic_lr":1e-3,
        "clip_param":0.2,

        "model_update_num":10,
        "max_iter":1000,
        "batch_size":200,
        "total_sample_size":2000,
        "test_iter":1000
    }),
    "acrobot":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/acrobat/1st/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/acrobat/1st/policies/",
        "hidden_size": [256, 128],

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
AWR = {
    "Pendulum-v0":
    edict({
        "gpu":True,
        "render":False,
        "log_dir":"./expData/Pendulum-v0/awr/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/Pendulum-v0/awr/policies/",
        "hidden_size": [128, 128],

        "gamma":0.99,
        "lamda":0.95,
        "actor_lr":1e-4,
        "critic_lr":1e-3,
        "beta":0.05,
        "buffer_size":2000,
        "max_weight":20,

        "model_update_num_critic":20,
        "model_update_num_actor":100,

        "max_iter":1000,
        "batch_size":100,
        "total_sample_size":500,
        "test_iter":1000
    }),
    "Walker2d-v2":
    edict({
       "gpu":True,
        "render":False,
        "log_dir":"./expData/Walker2d-v2/awr/logs/",
        "log_interval":1,
        "save_interval":30,
        "model_dir":"./expData/Walker2d-v2/awr/policies/",
        "hidden_size": [128, 64],

        "gamma":0.99,
        "lamda":0.95,
        "actor_lr":0.000025,
        "critic_lr":0.01,
        "beta":1.0,
        "buffer_size":50000,
        "max_weight":20,

        "model_update_num_critic":200,
        "model_update_num_actor":1000,

        "max_iter":1000,
        "batch_size":256,
        "total_sample_size":500,
        "test_iter":1000  
    })
}
