"""Command-line entry point for training, evaluation, and visualization."""

import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

from path_planner.LSTM_agent import LSTMAgent
from path_planner.config import DEFAULT_SEED, MAP_SAVE_DIR, MODEL_SAVE_DIR
from path_planner.gif_generator import generate_cleaning_gif
from path_planner.utils.utils import is_jupyter
from path_planner.utils.visualizer import visualize_saved_map


def _add_seed_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=str, default=MODEL_SAVE_DIR, help=f"Model directory (default: {MODEL_SAVE_DIR})")
    parser.add_argument("--map-save-dir", type=str, default=MAP_SAVE_DIR, help=f"Map directory (default: {MAP_SAVE_DIR})")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pre-model-name", type=str, default=None, help="Pre-trained model name used only when training")
    parser.add_argument("--model-name", type=str, default="model.pth", help="Model/checkpoint file name")
    parser.add_argument("--best-traj-img-name", type=str, default="best_coverage_path.png", help="Best trajectory image file name")
    parser.add_argument("--action-enc-dim", type=int, default=128, help="Action encoding dimension")
    parser.add_argument("--lstm-in-prj-dim", type=int, default=32, help="LSTM input projection dimension")
    parser.add_argument("--lstm-hid-dim", type=int, default=512, help="LSTM hidden-state dimension")


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("environment")
    group.add_argument("--max-steps", type=int, help="Maximum steps per episode")
    group.add_argument("--max-no-progress-steps", type=int, help="Steps allowed without coverage progress")
    group.add_argument("--max-no-progress-steps-final", type=int, help="Steps allowed without progress near completion")
    group.add_argument("--final-coverage-thres", type=float, help="Coverage threshold for final no-progress handling")
    group.add_argument("--target-coverage", type=float, help="Target coverage")
    group.add_argument("--local-view", type=float, help="Local observation size in cm")
    group.add_argument("--max-forward", type=float, help="Maximum forward distance in cm")
    group.add_argument("--robot-size", type=float, help="Robot diameter in cm")
    group.add_argument("--stack-steps", type=int, help="Number of stacked local-map observations")
    group.add_argument("--uncleaned-reward", type=float, help="Reward for newly cleaned area")
    group.add_argument("--cleaned-penalty", type=float, help="Penalty for revisiting cleaned area")
    group.add_argument("--obstacle-penalty", type=float, help="Collision penalty")
    group.add_argument("--turn-penalty", type=float, help="Turning penalty")
    group.add_argument("--step-penalty", type=float, help="Per-step penalty")
    group.add_argument("--complete-reward", type=float, help="Completion reward")
    group.add_argument("--intrinsic-reward", type=float, help="Intrinsic reward scale")


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    _add_seed_argument(parser)
    _add_storage_arguments(parser)
    _add_model_arguments(parser)
    _add_environment_arguments(parser)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--use-train-maps", action="store_true", help="Use pre-generated train maps")
    group.add_argument("--max-step-per-eps", type=int, help="Maximum episodes per map")
    group.add_argument("--max-eps-per-map-size", type=int, help="Maximum episodes at one map size")
    group.add_argument("--path-reward-thres", type=float, help="Validation reward threshold for map scale-up")
    group.add_argument("--max-episodes", type=int, help="Total training episodes")
    group.add_argument("--batch-size", type=int, help="Number of parallel environments")
    group.add_argument("--optimizer", choices=["sgd", "adam"], help="Optimizer")
    group.add_argument("--lr", type=float, help="Learning rate")
    group.add_argument("--momentum", type=float, help="SGD momentum")
    group.add_argument("--min-lr", type=float, help="Minimum scheduler learning rate")
    group.add_argument("--scheduler-max-step", type=int, help="Cosine scheduler duration")
    group.add_argument("--detach-period", type=int, help="LSTM detach period")
    group.add_argument("--valid-freq", type=int, help="Validation interval in episodes")
    group.add_argument("--ckp-freq", type=int, help="Checkpoint interval in episodes")
    group.add_argument("--valid-map-num", type=int, default=5, help="Validation maps per map size")
    group.add_argument("--valid-start-point-num", type=int, help="Validation start points per map")
    group.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    group.add_argument("--use-tb", action="store_true", help="Enable TensorBoard logging")
    group.add_argument("--use-vessl", action="store_true", help="Enable Vessl integration")


def _add_test_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("test")
    group.add_argument("--test-map-folder-name", type=str, default="test", help="Test-map subdirectory")
    group.add_argument("--test-map-num-per-level", type=int, default=250, help="Maps evaluated per level")
    group.add_argument("--test-start-point-num", type=int, default=1, help="Start points evaluated per map")
    group.add_argument("--vis-test-map-num", type=int, default=3, help="Best/median/worst paths shown per level")
    group.add_argument("--debug", action="store_true", help="Show interactive step-by-step test debugging")
    group.add_argument("--min-coverable-area-rate", type=float, default=0.1, help="Minimum coverable area rate for debugging")


def _add_map_path_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--map-rel-path", type=str, default="test/test_map_L1_0000.npy", help="Path relative to --map-save-dir")


def _normalize_legacy_arguments(argv: list[str]) -> list[str]:
    """Translate the former ``--mode`` CLI form to the subcommand form.

    Existing notebooks still use underscore-style options.  Keep them working
    during the CLI migration while presenting only the new form in ``-h``.
    """
    if "--mode" not in argv:
        return argv

    mode_index = argv.index("--mode")
    if mode_index + 1 >= len(argv):
        return argv

    legacy_modes = {
        "see_map": "map",
        "see_weight": "weights",
        "test_for_debug": "debug",
    }
    command = legacy_modes.get(argv[mode_index + 1], argv[mode_index + 1])
    remaining_args = argv[:mode_index] + argv[mode_index + 2:]
    normalized_args = [command]
    for argument in remaining_args:
        if argument.startswith("--"):
            option, separator, value = argument.partition("=")
            normalized_args.append(option.replace("_", "-") + separator + value)
        else:
            normalized_args.append(argument)

    print("Warning: '--mode ...' is deprecated; use a subcommand instead.", file=sys.stderr)
    return normalized_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot vacuum path-planning tools")
    commands = parser.add_subparsers(dest="mode", required=True, title="commands")

    train_parser = commands.add_parser("train", help="Train the LSTM policy")
    _add_agent_arguments(train_parser)
    _add_training_arguments(train_parser)

    test_parser = commands.add_parser("test", help="Evaluate a model over a test-map set")
    _add_agent_arguments(test_parser)
    _add_test_arguments(test_parser)

    gif_parser = commands.add_parser("gif", help="Save a single map rollout as an animated GIF")
    _add_agent_arguments(gif_parser)
    _add_map_path_argument(gif_parser)
    gif_parser.add_argument("--gif-frame-duration", type=float, default=0.08, help="GIF frame duration in seconds (default: 0.08)")

    inference_parser = commands.add_parser("inference", help="Run one map for inference and get performance metrics")
    _add_agent_arguments(inference_parser)
    _add_map_path_argument(inference_parser)
    inference_parser.add_argument("--start-mode", type=str, default="corner", choices=["corner", "edge"], help="Starting position mode for inference")
    inference_parser.add_argument("--min-coverable-area-rate", type=float, default=0.1, help="Minimum coverable area rate for inference")
    inference_parser.add_argument("--debug", action="store_true", help="Enable interactive step-by-step debugging mode")

    weights_parser = commands.add_parser("weights", help="Visualize selected model weights")
    _add_agent_arguments(weights_parser)

    map_parser = commands.add_parser("map", help="Display one saved map and its initial trajectory")
    _add_storage_arguments(map_parser)
    _add_map_path_argument(map_parser)

    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_legacy_arguments(raw_args))
    model_name = getattr(args, "model_name", None)
    if model_name:
        name, extension = os.path.splitext(model_name)
        if not extension:
            args.model_name = f"{name}.pth"
    return args


def _show_map(map_save_dir: str, map_rel_path: str) -> None:
    map_path = os.path.join(map_save_dir, map_rel_path)
    visualize_saved_map(map_file_path=map_path)

    from path_planner.environment import CoverageEnv
    from path_planner.map_layer import MapConfigSchema

    env = CoverageEnv()
    reset_result = env.reset(map_config=MapConfigSchema(file_path=map_path))
    if reset_result is not None:
        env.show_visualized_img(img_choice="traj", window_name=f"{os.path.basename(map_path)} image")


def main() -> None:
    args = parse_args()

    if args.mode == "map":
        _show_map(args.map_save_dir, args.map_rel_path)
    else:
        agent = LSTMAgent(args)
        if args.mode == "train":
            agent.train()
        elif args.mode == "test":
            agent.test()
        elif args.mode == "gif":
            generate_cleaning_gif(
                agent,
                map_file_path=os.path.join(args.map_save_dir, args.map_rel_path),
                frame_duration=args.gif_frame_duration,
            )
        elif args.mode == "inference":
            agent.run_inference_on_single_map(map_rel_path=args.map_rel_path, start_mode=args.start_mode, min_coverable_area_rate=args.min_coverable_area_rate, debug=args.debug)
        elif args.mode == "weights":
            agent.see_weight()

    if args.mode != "gif" and not is_jupyter():
        print("완료. 창을 확인하고 마우스로 닫으세요.")
        import matplotlib.pyplot as plt
        plt.show(block=True)


if __name__ == "__main__":
    main()
