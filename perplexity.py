# This file has been modified from the original implementation 
# of the HiddenKiller paper and can be found at 
# 
# https://github.com/thunlp/HiddenKiller/blob/a08e959e228327baa0c2906bf943e99a3c89961c/experiments/gptlm.py
import math
import numpy as np
import transformers

class GPT2LM:
    def __init__(self, device=None, little=False):
        """
        :Package Requirements:
            * **tensorflow** >= 2.0.0
            * **transformers**

        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__
        """
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
               
    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        import tensorflow as tf
        ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
        ret = self.lm(ipt)[0]
        loss = 0
        for i in range(ret.shape[0]):
            it = ret[i]
            it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
            it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
            it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
            loss += tf.reduce_mean(it)
            break
        return math.exp(-loss)
