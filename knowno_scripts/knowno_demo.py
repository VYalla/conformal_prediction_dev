#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KnowNo Demo: Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners

This script demonstrates the basics of constructing the prediction set (possible actions in a scenario)
in the Mobile Manipulation setting using the KnowNo framework.
"""

import openai
import signal
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Timeout handler for API calls
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only considering the five option
# tokens: A, B, C, D, E
def lm(prompt,
       max_tokens=256,
       temperature=0,
       logprobs=None,
       stop_seq=None,
       logit_bias={
          317: 100.0,   #  A (with space at front)
          347: 100.0,   #  B (with space at front)
          327: 100.0,   #  C (with space at front)
          360: 100.0,   #  D (with space at front)
          412: 100.0,   #  E (with space at front)
      },
       timeout_seconds=20):
    max_attempts = 5
    response = None
    for attempt in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                # Using the new OpenAI API format
                response = client.completions.create(
                    model='gpt-3.5-turbo-instruct',  # Replacement for text-davinci-003
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    logit_bias=logit_bias,
                    stop=list(stop_seq) if stop_seq is not None else None,
                )
            break
        except Exception as e:
            print(f'Attempt {attempt+1}/{max_attempts} failed: {str(e)}')
            if attempt == max_attempts - 1:
                raise Exception(f"All {max_attempts} attempts to call OpenAI API failed. Last error: {str(e)}")
    
    if response is None:
        raise Exception(f"Failed to get a response from OpenAI API after {max_attempts} attempts")
    
    # Extract text and logprobs from the new API response format
    text = response.choices[0].text.strip()
    
    # Return in a format compatible with the rest of the code
    response_dict = {
        "choices": [{
            "text": text,
            "logprobs": response.choices[0].logprobs if hasattr(response.choices[0], 'logprobs') else None
        }]
    }
        
    return response_dict, text

def process_mc_raw(mc_raw, add_mc='an option not listed here'):
    mc_all = mc_raw.split('\n')

    mc_processed_all = []
    for mc in mc_all:
        mc = mc.strip()

        # skip nonsense
        if len(mc) < 5 or mc[0] not in [
            'a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', '1', '2', '3', '4'
        ]:
            continue
        mc = mc[2:]  # remove a), b), ...
        mc = mc.strip().lower().split('.')[0]
        mc_processed_all.append(mc)
    if len(mc_processed_all) < 4:
        raise Exception('Cannot extract four options from the raw output.')

    # Check if any repeated option - use do nothing as substitue
    mc_processed_all = list(set(mc_processed_all))
    if len(mc_processed_all) < 4:
        num_need = 4 - len(mc_processed_all)
        for _ in range(num_need):
            mc_processed_all.append('do nothing')
    prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
    if add_mc is not None:
        mc_processed_all.append(add_mc)
        prefix_all.append('E) ')
    random.shuffle(mc_processed_all)

    # get full string
    mc_prompt = ''
    for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
        mc_prompt += prefix + mc
        if mc_ind < len(mc_processed_all) - 1:
            mc_prompt += '\n'
    add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
    return mc_prompt, mc_processed_all, add_mc_prefix

def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

def main():
    parser = argparse.ArgumentParser(description='KnowNo Demo')
    parser.add_argument('--instruction', type=str, default="Put the bottled water in the bin.",
                        help='Task instruction for the robot')
    parser.add_argument('--scene_objects', type=str, default="energy bar, bottled water, rice chips",
                        help='Objects in the scene, comma separated')
    args = parser.parse_args()

    instruction = args.instruction
    scene_objects = args.scene_objects

    # First, prompt the LLM to generate possible options with few-shot prompting
    demo_mc_gen_prompt = """
We: You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.

We: On the counter, there is an orange soda, a Pepsi, and an apple.
We: Put that drink in the top drawer.
You:
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it

We: On the counter, there is an energy bar, a banana, and a microwave.
We: Put the snack next to the microwave.
You:
A) pick up the energy bar and put it next to the microwave
B) pick up the banana and put it next to the energy bar
C) pick up the banana and put it next to the microwave
D) pick up the energy bar and put it next to the banana

We: On the counter, there is a Coke, a Sprite, and a sponge.
We: Can you dispose of the can? It should have expired.
You:
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin

We: On the counter, there is a bottled water, a bag of jalapeno chips, and a bag of rice chips.
We: I would like a bag of chips.
You:
A) pick up the bottled water
B) pick up the jalapeno chips
C) pick up the kettle chips
D) pick up the rice chips

We: On the counter, there is {scene_objects}
We: {task}
You:
"""

    demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{task}', instruction)
    demo_mc_gen_prompt = demo_mc_gen_prompt.replace('{scene_objects}', scene_objects)

    # Generate multiple choices
    _, demo_mc_gen_raw = lm(demo_mc_gen_prompt, stop_seq=['We:'], logit_bias={})
    demo_mc_gen_raw = demo_mc_gen_raw.strip()
    demo_mc_gen_full, demo_mc_gen_all, demo_add_mc_prefix = process_mc_raw(demo_mc_gen_raw)

    print('====== Prompt for generating possible options ======')
    print(demo_mc_gen_prompt)

    print('====== Generated options ======')
    print(demo_mc_gen_full)

    # Then we evaluate the probabilities of the LLM predicting each option (A/B/C/D/E)
    # get the part of the current scenario from the previous prompt
    demo_cur_scenario_prompt = demo_mc_gen_prompt.split('\n\n')[-1].strip()

    # get new prompt
    demo_mc_score_background_prompt = """
You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
""".strip()
    demo_mc_score_prompt = demo_mc_score_background_prompt + '\n\n' + demo_cur_scenario_prompt + '\n' + demo_mc_gen_full
    demo_mc_score_prompt += "\nWe: Which option is correct? Answer with a single letter."
    demo_mc_score_prompt += "\nYou:"

    # scoring
    mc_score_response, _ = lm(demo_mc_score_prompt, max_tokens=1, logprobs=5)
    
    # Handle the new API response format
    if hasattr(mc_score_response["choices"][0]["logprobs"], "top_logprobs"):
        top_logprobs_full = mc_score_response["choices"][0]["logprobs"].top_logprobs[0]
        top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        top_logprobs = [value for value in top_logprobs_full.values()]
    else:
        # Fallback if logprobs not available
        print("Warning: Logprobs not available in API response. Using dummy values for demonstration.")
        top_tokens = ['A', 'B', 'C', 'D', 'E']
        top_logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]

    print('====== Prompt for scoring options ======')
    print(demo_mc_score_prompt)

    print('\n====== Raw log probabilities for each option ======')
    for token, logprob in zip(top_tokens, top_logprobs):
        print('Option:', token, '\t', 'log prob:', logprob)

    # Construct prediction set
    # With the probabilities from the LLM, we can construct the prediction set
    # From calibration, we have determined the threshold to be 0.072 with a target success level of 0.8
    qhat = 0.928

    # get prediction set
    mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

    # include all options with score >= 1-qhat
    prediction_set = [
        token for token_ind, token in enumerate(top_tokens)
        if mc_smx_all[token_ind] >= 1 - qhat
    ]

    # print results
    print('Softmax scores:', mc_smx_all)
    print('Prediction set:', prediction_set)
    if len(prediction_set) != 1:
        print('Help needed!')
    else:
        print('No help needed!')

if __name__ == "__main__":
    main()
