import argparse

from ..diffusion import gaussian_diffusion as gd
from ..diffusion.respace import SpacedDiffusion, space_timesteps
# from .unet import UNetWithStyEncoderModel
from ..models import DiT_models

def diffusion_defaults():

    return dict(
        learn_sigma=False,  # learn VLB term when True
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False, # learn VLB term when True
    )

def model_and_diffusion_defaults():

    res = dict(
        model_name='DiT-S/8',
        image_size=64,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        channel_mult="",
        dropout=0.0,
        stroke_path=None,
        chara_nums=6625,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res

def create_model_and_diffusion(
    model_name,
    image_size,
    chara_nums,
    learn_sigma,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_fp16,
    use_new_attention_order,
    stroke_path,
):
    model = create_model(
        model_name,
        image_size,
        chara_nums,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        stroke_path=stroke_path,
    )
    diffusion = create_gaussian_diffusion(  ### get a noise schedule
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def create_model(
    model_name,
    image_size,
    chara_nums,
    channel_mult="",
    learn_sigma=False,
    use_checkpoint=False,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    use_fp16=False,
    use_new_attention_order=False,
    stroke_path=None,
):
    if stroke_path is not None:
        use_stroke = True
    else:
        use_stroke = False

    # Create model:
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size # // 8
    model = DiT_models[model_name](
        input_size=latent_size,
        in_channels=3,
        learn_sigma=learn_sigma,
        num_classes=chara_nums,
        use_stroke=use_stroke,
    )
    return model



def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
