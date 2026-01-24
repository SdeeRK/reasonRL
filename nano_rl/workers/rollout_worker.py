
import ray
from vllm import LLM, SamplingParams

from ..core.config import GenerationConfig, ModelConfig
from ..core.types import (
    GroupRollout,
    ModelCheckpoint,
    PromptBatch,
    RolloutBatch,
    SingleRollout,
)


@ray.remote(num_gpus=1)
class RolloutWorker:
    """
    vLLM-based rollout generation worker implementing RolloutWorkerProtocol.

    Responsible for generating responses from prompts using the current policy.
    Runs as a Ray remote actor with GPU access for efficient batched inference.
    """

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.model_config = model_config
        self.generation_config = generation_config

        self.engine = LLM(
            model=model_config.model_name_or_path,
            dtype=model_config.dtype,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            enforce_eager=True,
        )

    def generate(
        self,
        requests: PromptBatch,
        generation_config: GenerationConfig | None = None,
    ) -> RolloutBatch:
        generation_config = (
            generation_config if generation_config else self.generation_config
        )
        sampling_params = SamplingParams(
            n=generation_config.group_size,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            min_tokens=generation_config.min_tokens,
            max_tokens=generation_config.max_tokens,
            logprobs=1 if generation_config.include_logprobs else None,
        )

        # 优先走token id的形式，如果不行再走字符串形式
        prompt_token_ids: list[list[int]] | None = (
            requests.prompts_token_ids if requests.prompts_token_ids else None
        )
        prompts: list[str] | None = (
            None
            if prompt_token_ids is not None
            else [sample.prompt for sample in requests.samples]
        )

        engine_outputs = self.engine.generate(
            prompts=prompts,  # type: ignore
            prompt_token_ids=prompt_token_ids,  # type: ignore
            sampling_params=sampling_params,
        )

        groups: list[GroupRollout] = []
        for sample, request_output in zip(requests.samples, engine_outputs):
            items: list[SingleRollout] = []
            for completion_output in request_output.outputs:
                response_ids = list(completion_output.token_ids)
                response_text = completion_output.text

                old_log_probs = []
                if generation_config.include_logprobs and completion_output.logprobs:
                    for i, token_id in enumerate(response_ids):
                        logprob_obj = completion_output.logprobs[i][token_id]
                        old_log_probs.append(logprob_obj.logprob)

                items.append(
                    SingleRollout(
                        response_ids=response_ids,
                        old_log_probs=old_log_probs,
                        response_text=response_text,
                    )
                )

            groups.append(GroupRollout(sample=sample, items=items))

        return RolloutBatch(groups=groups)

    def update_weights(self, checkpoint: ModelCheckpoint) -> None:
        state_dict_cpu = {k: v.cpu() for k, v in checkpoint.state_dict.items()}
        engine = self.engine.llm_engine.model_executor.driver_worker.model_runner.model
        engine.load_weights(state_dict_cpu.items())
