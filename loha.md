LoHaConfig
class peft.LoHaConfig
<
source
>
( task_type: Optional[Union[str, TaskType]] = Nonepeft_type: Optional[Union[str, PeftType]] = Noneauto_mapping: Optional[dict] = Nonepeft_version: Optional[str] = Nonebase_model_name_or_path: Optional[str] = Nonerevision: Optional[str] = Noneinference_mode: bool = Falserank_pattern: Optional[dict] = <factory>alpha_pattern: Optional[dict] = <factory>r: int = 8alpha: int = 8rank_dropout: float = 0.0module_dropout: float = 0.0use_effective_conv2d: bool = Falsetarget_modules: Optional[Union[list[str], str]] = Noneexclude_modules: Optional[Union[list[str], str]] = Noneinit_weights: bool = Truelayers_to_transform: Optional[Union[list[int], int]] = Nonelayers_pattern: Optional[Union[list[str], str]] = Nonemodules_to_save: Optional[list[str]] = None )

Parameters

r (int) — LoHa rank.
alpha (int) — The alpha parameter for LoHa scaling.
rank_dropout (float) — The dropout probability for rank dimension during training.
module_dropout (float) — The dropout probability for disabling LoHa modules during training.
use_effective_conv2d (bool) — Use parameter effective decomposition for Conv2d (and Conv1d) with ksize > 1 (“Proposition 3” from FedPara paper).
target_modules (Optional[Union[List[str], str]]) — The names of the modules to apply the adapter to. If this is specified, only the modules with the specified names will be replaced. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as ‘all-linear’, then all linear/Conv1D modules are chosen, excluding the output layer. If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules manually.
exclude_modules (Optional[Union[List[str], str]]) — The names of the modules to not apply the adapter. When passing a string, a regex match will be performed. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings.
init_weights (bool) — Whether to perform initialization of adapter weights. This defaults to True, passing False is discouraged.
layers_to_transform (Union[List[int], int]) — The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices that are specified in this list. If a single integer is passed, it will apply the transformations on the layer at this index.
layers_pattern (Optional[Union[List[str], str]]) — The layer pattern name, used only if layers_to_transform is different from None. This should target the nn.ModuleList of the model, which is often called 'layers' or 'h'.
rank_pattern (dict) — The mapping from layer names or regexp expression to ranks which are different from the default rank specified by r. For example, {'^model.decoder.layers.0.encoder_attn.k_proj': 16}.
alpha_pattern (dict) — The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by alpha. For example, {'^model.decoder.layers.0.encoder_attn.k_proj': 16}.
modules_to_save (Optional[List[str]]) — List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
This is the configuration class to store the configuration of a LoHaModel.
LoHaModel
class peft.LoHaModel
<
source
>
( modelpeft_config: Union[PeftConfig, dict[str, PeftConfig]]adapter_name: strlow_cpu_mem_usage: bool = Falsestate_dict: Optional[dict[str, torch.Tensor]] = None ) → torch.nn.Module

Parameters

model (torch.nn.Module) — The model to which the adapter tuner layers will be attached.
config (LoHaConfig) — The configuration of the LoHa model.
adapter_name (str) — The name of the adapter, defaults to "default".
low_cpu_mem_usage (bool, optional, defaults to False) — Create empty adapter weights on meta device. Useful to speed up the loading process.
Returns

torch.nn.Module

The LoHa model.


Creates Low-Rank Hadamard Product model from a pretrained model. The method is partially described in https://huggingface.co/papers/2108.06098 Current implementation heavily borrows from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py

Example:

Copied
from diffusers import StableDiffusionPipeline
from peft import LoHaModel, LoHaConfig

config_te = LoHaConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)
config_unet = LoHaConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "proj_in",
        "proj_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,
)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.text_encoder = LoHaModel(model.text_encoder, config_te, "default")
model.unet = LoHaModel(model.unet, config_unet, "default")
Attributes:

model (~torch.nn.Module) — The model to be adapted.
peft_config (LoHaConfig): The configuration of the LoHa model.