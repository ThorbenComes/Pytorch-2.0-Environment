import experiment_data.timeseriesExperiment
import wandb
import utils.wandb_utils
import time


def conduct_experiment(name, project_name, mode):

    start_time = time.time()
    wandb_run = utils.wandb_utils.wandb_init(name=name, project_name=project_name, mode=mode)

    # choose experiment
    experiment_type = choose_experiment()

    # initialize experiment
    init_time = time.time()
    print_time(start_time, init_time, "Duration of pre-initialization:")
    experiment = experiment_type.Experiment()

    # train model
    train_time = time.time()
    print_time(init_time, train_time, "Duration of initialization:")
    experiment.train()

    # evaluate model
    eval_time = time.time()
    print_time(train_time, eval_time, "Duration of training:")
    experiment.eval()

    # finish experiment
    finish_time = time.time()
    print_time(eval_time, finish_time, "Duration of evaluation:")
    print_time(start_time, finish_time, "Total duration:")
    # profit.
    # maybe cleanup


def print_time(t_start, t_end, message):
    """calculates duration and prints it"""

    duration = t_end - t_start
    print(message, duration)


def choose_experiment():
    """chooses experiment based on configuration"""

    # TODO
    return experiment_data.timeseriesExperiment
