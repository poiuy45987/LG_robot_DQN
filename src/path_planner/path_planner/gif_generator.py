"""Create an animated cleaning trajectory for one saved test map."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from path_planner.config import RESULT_SAVE_DIR
from path_planner.map_layer import MapConfigSchema

if TYPE_CHECKING:
    from path_planner.LSTM_agent import LSTMAgent


def generate_cleaning_gif(
    agent: LSTMAgent,
    map_file_path: str,
    frame_duration: float = 0.08,
    start_mode: str = "corner",
) -> Path:
    """Save every cleaning step on one map as an animated trajectory GIF.

    Frames are written incrementally rather than being held in memory.  The
    rollout itself is delegated to ``LSTMAgent._test_one_map`` so it always
    uses the exact same policy and action-mask logic as normal testing.
    """
    if frame_duration <= 0:
        raise ValueError("frame_duration must be greater than zero.")

    map_path = Path(map_file_path)
    if not map_path.is_file():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "GIF export requires imageio. Install project requirements again "
            "after adding the imageio dependency."
        ) from exc

    model_stem = Path(agent.args.model_name).stem
    output_dir = Path(RESULT_SAVE_DIR) / model_stem / "gif_result"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{map_path.stem}.gif"

    reset_result = agent.test_env.reset(
        seed=None,
        start_mode=start_mode,
        map_config=MapConfigSchema(file_path=str(map_path)),
    )
    if reset_result is None:
        raise RuntimeError(f"Could not reset the environment for map: {map_path.name}")

    obs, _ = reset_result
    policy_was_training = agent.policy_net.training
    agent.policy_net.eval()

    frame_count = 0

    try:
        with imageio.get_writer(output_path, mode="I", duration=frame_duration, loop=0) as writer:
            def append_trajectory_frame(env, _step: int) -> None:
                """Write the initial frame and every completed cleaning step."""
                nonlocal frame_count
                writer.append_data(env.get_visualized_img(img_choice="traj"))
                frame_count += 1

            agent._test_one_map(
                agent.test_env,
                obs,
                debug=False,
                on_step=append_trajectory_frame,
            )
    finally:
        agent.policy_net.train(policy_was_training)

    print(
        f"GIF saved: {output_path} "
        f"({frame_count} frames, {frame_duration:.3f}s per frame)"
    )
    return output_path
