import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice"
    )

    parser.add_argument(
        "--env_name",
        default="BipedalWalker-v3",
        type=str,
        help="Name of the environment.",
    )
    parser.add_argument(
        "--interval",
        default=20,
        type=int,
        help="The interval specifies how often different parameters should be saved and printed,"
        " counted by episodes.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="The flag determines whether to train the agent or play with it.",
    )
    parser.add_argument(
        "--train_from_scratch",
        action="store_false",
        help="The flag determines whether to train from scratch or continue previous tries.",
    )
    parser.add_argument(
        "--mem_size", default=int(1e6), type=int, help="The memory size."
    )
    parser.add_argument(
        "--n_skills", default=50, type=int, help="The number of skills to learn."
    )
    parser.add_argument(
        "--reward_scale",
        default=1,
        type=float,
        help="The reward scaling factor introduced in SAC.",
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="The randomness' seed for torch, numpy, random & gym[env].",
    )
    parser.add_argument(
        "--n_evals",
        default=1000,
        type=int,
        help="How many iterations should an agent go through during evaluation",
    )
    parser.add_argument("--save_path", default="./results", type=str, help="Save path")
    parser.add_argument(
        "--neurons_list",
        default="128 128",
        type=str,
        help="Actor NN: [neurons_list + [action dim]]",
    )
    parser.add_argument(
        "--n_prior",
        default=0,
        type=int,
        help="Size of prior, set to 0 for no prior.",
    )

    parser_params = parser.parse_args()

    # Parameters based on the DIAYN and SAC papers.
    # region default parameters
    default_params = {
        "lr": 3e-4,
        "batch_size": 256,
        "max_n_episodes": 5000,
        "max_episode_len": 1000,
        "gamma": 0.99,
        "alpha": 0.1,
        "tau": 0.005,
        "n_hiddens": 300,
    }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    total_params["neurons_list"] = [int(x) for x in total_params["neurons_list"].split()]
    return total_params
