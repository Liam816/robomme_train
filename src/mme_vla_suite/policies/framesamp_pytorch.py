"""PyTorch inference callbacks for the existing RoboMME policy/history buffer."""

import jax
import numpy as np
import torch

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.framesamp_pytorch import FrameSampPolicy
from mme_vla_suite.training.framesamp_weights import load_orbax


def create_model(config, checkpoint, device="cuda"):
    memory = get_history_config(config.history_config)
    if config.history_config != "perceptual-framesamp-modul.yaml" or not config.pi05:
        raise ValueError("The PyTorch evaluator supports the π0.5 FrameSamp+Modul checkpoint")
    model = FrameSampPolicy(config, memory, activation_checkpointing=False)
    load_orbax(model, checkpoint / "params")
    # The released evaluation loader restores all tensors with dtype=bfloat16.
    # Keep exactly those values; action/time linear layers compute with FP32 inputs.
    model.bfloat16().float().to(device).eval()

    def tensor(value, floating=False):
        array = np.array(value, dtype=np.float32 if floating else None)
        return torch.from_numpy(array).to(device)

    @torch.inference_mode()
    def vision(images):
        t, v, h, w, c = images.shape
        x = tensor(images, True).reshape(t * v, h, w, c).permute(0, 3, 1, 2)
        result = model.encode_images(x).reshape(t, v, -1, 2048)
        # Preserve bfloat16 during the reference buffer's spatial mean pooling.
        return jax.numpy.asarray(result.float().cpu().numpy(), dtype=jax.numpy.bfloat16)

    @torch.inference_mode()
    def sample(key, obs, *, num_steps=10):
        batch = {
            name: tensor(getattr(obs, name), floating=name in ("state", "static_image_emb", "static_pos_emb"))
            for name in (
                "state",
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "static_image_emb",
                "static_pos_emb",
                "static_mask",
            )
        }
        batch["images"] = torch.stack(
            [tensor(obs.images[name], True).permute(0, 3, 1, 2) for name in ("base_0_rgb", "left_wrist_0_rgb")], dim=1
        )
        shape = (batch["state"].shape[0], model.action_horizon, model.action_dim)
        return model.sample(batch, tensor(jax.random.normal(key, shape), True), num_steps).cpu().numpy()

    return model, sample, vision


def create_eval_policy(train_config, checkpoint, seed, steps, vision_batch, device="cuda"):
    import dataclasses

    from mme_vla_suite.policies.policy import MME_VLA_Policy
    from openpi import transforms
    from openpi.shared.normalize import load

    history = (checkpoint.parent / "history_config.txt").read_text().strip()
    model_config = dataclasses.replace(train_config.model, use_history=True, history_config=history)
    model, sample, vision = create_model(model_config, checkpoint, device)
    data = train_config.data.create(checkpoint / "assets", model_config)
    stats = load(checkpoint / "assets" / data.asset_id)

    class TorchPolicy(MME_VLA_Policy):
        """Keep the released reset/infer/history/normalization methods and supply Torch callbacks."""

        def __init__(self):
            self._model, self.config = model, model.history_config
            self._seed, self._metadata = seed, {}
            self._sample_kwargs = {"num_steps": steps}
            self._sample_actions = sample
            self._vision_encode = self.encode
            self.state_norm_stats, self.use_quantiles = stats["state"], data.use_quantile_norm
            self._input_transform = transforms.compose(
                [
                    transforms.InjectDefaultPrompt(None),
                    *data.data_transforms.inputs,
                    transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
                    *data.model_transforms.inputs,
                ]
            )
            self._output_transform = transforms.compose(
                [
                    *data.model_transforms.outputs,
                    transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
                    *data.data_transforms.outputs,
                ]
            )
            self.mem_buffer = None
            self.reset()

        def encode(self, images):
            return jax.numpy.concatenate(
                [vision(images[i : i + vision_batch]) for i in range(0, len(images), vision_batch)], axis=0
            )

    return TorchPolicy()
