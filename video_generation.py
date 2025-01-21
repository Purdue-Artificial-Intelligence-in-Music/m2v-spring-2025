"""
This script includes code to generate videos from features.

Todos:
TODO 0: Add imports for the existing code
TODO 1: Implement segmentation, adapt code to use it (i.e., create video prompt per music segment)
"""

SCHEDULER_PARAMS = {
    "clip_sample": False, # Having this as True can make video become very fuzzy
    "timestep_spacing": "linspace", # "linspace", "log?"
    "beta_schedule": "linear",
    "steps_offset": 5
}

def load_default_pipe():
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=SCHEDULER_PARAMS["clip_sample"],
        timestep_spacing=SCHEDULER_PARAMS["timestep_spacing"],
        beta_schedule=SCHEDULER_PARAMS["beta_schedule"],
        steps_offset=SCHEDULER_PARAMS["steps_offset"],
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe, adapter



