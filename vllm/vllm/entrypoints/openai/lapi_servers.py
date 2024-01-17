# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import codecs
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
from PIL import Image
from io import BytesIO
import pdb

import requests
import base64

from transformers import TextIteratorStreamer

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from vllm.engine.arg_utils import AsyncEngineArgs

# from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.async_llava_engine import AsyncLLaVAEngine

from vllm.engine.metrics import add_global_metrics_labels
from vllm.entrypoints.openai.openai_protocol import (ChatCompletionRequest,
                             ChatCompletionResponseStreamChoice,
                             ChatCompletionStreamResponse, DeltaMessage,
                             ChatCompletionResponseChoice,
                             ChatCompletionResponse,
                             random_uuid)

from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    # ChatCompletionRequest,
    # ChatCompletionResponse,
    # ChatCompletionResponseChoice,
    # ChatCompletionResponseStreamChoice,
    # ChatCompletionStreamResponse,
    ChatMessage,
    # DeltaMessage,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()
engine = None
response_role = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The file path to the chat template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--response-role",
        type=str,
        default="assistant",
        help="The role name to return if " "`request.add_generation_prompt=true`.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="The file path to the SSL key file",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="The file path to the SSL cert file",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


app.add_middleware(MetricsMiddleware)  # Trace HTTP server metrics
app.add_route("/metrics", metrics)  # Exposes HTTP metrics


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None,
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert not (prompt is None and prompt_ids is None) and not (
        prompt is not None and prompt_ids is not None
    ), "Either prompt or prompt_ids should be provided."
    input_ids = prompt_ids if prompt_ids is not None else tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model, root=served_model, permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)




def convert_to_desired_format(messages):
    new_messages = []
    img_enc = None

    for msg in messages:
        new_msg = {"role": msg["role"]}

        for content_item in msg["content"]:
            if content_item["type"] == "image_url":
                img_enc = content_item["image_url"]["url"]
            else:
                new_msg["content"] = content_item["text"]

        new_messages.append(new_msg)

    return new_messages, img_enc


def read_image(input_string):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    if input_string.startswith("http"):
        # Case: URL
        response = requests.get(input_string, headers=headers)
        img = Image.open(BytesIO(response.content))
    elif input_string.startswith("data:image"):
        # Case: base64-encoded string
        _, encoded_data = input_string.split(",", 1)
        img_data = base64.b64decode(encoded_data)
        img = Image.open(BytesIO(img_data))
    else:
        raise ValueError("Unsupported input format")

    img = img.convert("RGB")
    img.thumbnail((512, 512))

    return img


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
        )
    
    # pdb.set_trace()
    # print(request)
    # logger.warning(request)
    prompt = ""
    images = []
    
    for message in request.messages:
        if message["role"] == "user":
            prompt += "USER:\n"
            for content in message["content"]:
                if content["type"] == "text":
                    prompt += f"{content['text']}\n"
                if content["type"] == "image_url":
                    # read the image
                    url = content["image_url"]["url"]
                    image = read_image(url)
                    images.append(image)
                    prompt += f"<image>\n"
        if message["role"] == "assistant":
            prompt += "ASSISTANT:\n"
            for content in message["content"]:
                if content["type"] == "text":
                    prompt += f"{content['text']}\n"

    prompt += "ASSISTANT:\n"
    

    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    chunk_object_type = "chat.completion.chunk"
    # request.n = 1
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=1,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    # result_generator = engine.generate(prompt, sampling_params, request_id,
    #                                    token_ids)

    result_generator = engine.generate(
        prompt, sampling_params, request_id, images=images
    )

        
    def get_role() -> str:
        return "assistant"
    

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # Send first response for each request.n (index) with the role
        role = get_role()
        for i in range(1):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name,
            )
            # data = chunk.json(exclude_unset=True, ensure_ascii=False)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if (
                request.messages
                and isinstance(request.messages, list)
                and request.messages[-1].get("content")
                and request.messages[-1].get("role") == role
            ):
                last_msg_content = request.messages[-1]["content"]
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index

                if finish_reason_sent[i]:
                    continue

                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
                    delta_text = output.text[len(previous_texts[i]) :]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                else:
                    # Send the finish response for each request.n only once
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=previous_num_tokens[i],
                        total_tokens=prompt_tokens + previous_num_tokens[i],
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i, delta=[], finish_reason=output.finish_reason
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    if final_usage is not None:
                        chunk.usage = final_usage
                    data = chunk.json(
                        exclude_unset=True, exclude_none=True, ensure_ascii=False
                    )
                    yield f"data: {data}\n\n"
                    finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def completion_full_generator():
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, "Client disconnected"
                )
            final_res = res
        assert final_res is not None

        choices = []
        role = get_role()
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if (
                request.messages
                and isinstance(request.messages, list)
                and request.messages[-1].get("content")
                and request.messages[-1].get("role") == role
            ):
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    # Streaming response
    if request.stream:
        return StreamingResponse(
            completion_stream_generator(), media_type="text/event-stream"
        )
    else:
        return await completion_full_generator()


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    response_role = args.response_role

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine = AsyncLLaVAEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code,
    )
    # load_chat_template(args, tokenizer)

    # Register labels for metrics
    add_global_metrics_labels(model_name=engine_args.model)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )

