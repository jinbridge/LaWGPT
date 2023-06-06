import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer, \
    AutoModelForCausalLM
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pyngrok import ngrok
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        except:
            print("*" * 50, "\n Attention! No Lora Weights \n", "*" * 50)
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        except:
            print("*" * 50, "\n Attention! No Lora Weights \n", "*" * 50)
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
        except:
            print("*" * 50, "\n Attention! No Lora Weights \n", "*" * 50)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            # input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        input = None
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            print(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)
        yield prompter.get_response(output)

    class MyHttpHander(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            data = {'code': 0, 'success': False, 'message': 'Not support get method'}
            response = json.dumps(data)
            self.wfile.write(response.encode())

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            path = self.path
            if path != "/law-gpt/chat":
                data = {'code': 0, 'success': False, 'message': 'No find this url'}
                response = json.dumps(data)
                self.wfile.write(response.encode())
                return
            content_length = int(self.headers["content-length"])
            data = self.rfile.read(content_length)
            data = data.decode("utf-8").replace("\r", "")
            data = data.replace("\n", "")
            data = data.replace("\t", "")
            data = data.replace(" ", "")
            data = json.loads(data)
            message = data.get("message")
            res = evaluate(message)
            data = {'code': 1, 'success': True, 'message': '操作成功', 'data': res}
            response = json.dumps(data)
            self.wfile.write(response.encode())
            return

    os.environ["https_proxy"] = "http://192.168.2.69:7890"
    httpserver = ThreadingHTTPServer(("", 8000), MyHttpHander)
    httpserver.serve_forever()
    public_url = ngrok.connect(8000).public_url
    print(f'Serving HTTP on {public_url}')

# 注册nacos
# registerServer()



if __name__ == "__main__":
    fire.Fire(main)
