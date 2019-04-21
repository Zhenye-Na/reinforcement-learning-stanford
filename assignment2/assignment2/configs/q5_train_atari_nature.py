class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = "results/q5_train_atari_nature/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000
