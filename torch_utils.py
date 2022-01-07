from __future__ import absolute_import
from __future__ import print_function

import ast
import astor
import copy
import itertools
import json
import nltk
import os
import random
import re
import sys
import torch
import traceback

import numpy as np
import token as tk
import torch.nn as nn

from io import open
from io import StringIO
from itertools import cycle
from tokenize import generate_tokens
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils import model_zoo
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer)

nltk.download('punkt')

class Seq2Seq(nn.Module):
    def __init__(self, encoder,decoder,config,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            preds=[]
            zero=torch.LongTensor(1).fill_(0)    
            for i in range(source_ids.shape[0]):
                context=encoder_output[:,i:i+1]
                context_mask=source_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context=context.repeat(1, self.beam_size,1)
                context_mask=context_mask.repeat(self.beam_size,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)                
            return preds   
        
        

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch
        self.scores = self.tt.FloatTensor(size).zero_()
        self.prevKs = []
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        self._eos = eos
        self.eosTop = False
        self.finished = []

    def getCurrentState(self):
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def advance(self, wordLk):
        numWords = wordLk.size(1)

        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def canonicalize_intent(intent):
    QUOTED_STRING_RE = re.compile(r"(?P<quote>[`'\"])(?P<string>.*?)(?P=quote)")
    str_matches = QUOTED_STRING_RE.findall(intent)

    slot_map = dict()

    return intent, slot_map


def replace_strings_in_ast(py_ast, string2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue

            if isinstance(v, str):
                if v in string2slot:
                    val = string2slot[v]
                    setattr(node, k, val)
                else:
                    if str_key in string2slot:
                        val = string2slot[str_key]
                        if isinstance(val, str):
                            try: val = val.encode('ascii')
                            except: pass
                        setattr(node, k, val)


def canonicalize_code(code, slot_map):
    string2slot = {x[1]['value']: x[0] for x in list(slot_map.items())}

    py_ast = ast.parse(code)
    replace_strings_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast)

    return canonical_code


def decanonicalize_code(code, slot_map):
    try:
      slot2string = {x[0]: x[1]  for x in list(slot_map.items())}
      py_ast = ast.parse(code)
      replace_strings_in_ast(py_ast, slot2string)
      raw_code = astor.to_source(py_ast)
      for slot_name, slot_info in slot_map.items():
           raw_code = raw_code.replace(slot_info, slot_name)

      return raw_code.strip()
    except:
      return code


def detokenize_code(code_tokens):
    newline_pos = [i for i, x in enumerate(code_tokens) if x == '\n']
    newline_pos.append(len(code_tokens))
    start = 0
    lines = []
    for i in newline_pos:
        line = ' '.join(code_tokens[start: i])
        start = i + 1
        lines.append(line)

    code = '\n'.join(lines).strip()

    return code


def encode_tokenized_code(code_tokens):
    tokens = []
    for token in code_tokens:
        if token == '\t':
            tokens.append('_TAB_')
        elif token == '\n':
            tokens.append('_NEWLINE_')


def get_encoded_code_tokens(code):
    code = code.strip()
    #print(code)
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    indent_level = 0
    new_line = False

    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.NEWLINE:
            tokens.append('#NEWLINE#')
            new_line = True
        elif toknum == tk.INDENT:
            indent_level += 1
        elif toknum == tk.STRING:
            tokens.append(tokval.replace(' ', '#SPACE#').replace('\t', '#TAB#').replace('\r\n', '#NEWLINE#').replace('\n', '#NEWLINE#'))
        elif toknum == tk.DEDENT:
            indent_level -= 1
        else:
            tokval = tokval.replace('\n', '#NEWLINE#')
            if new_line:
                for i in range(indent_level):
                    tokens.append('#INDENT#')

            new_line = False
            tokens.append(tokval)

    if len(tokens[-1]) == 0:
        tokens = tokens[:-1]

    if '\n' in tokval:
        pass

    return tokens


def tokenize(code):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break

        tokens.append(tokval)

    return tokens


def compare_ast(node1, node2):
    if not isinstance(node1, str):
        if type(node1) is not type(node2):
            return False
    if isinstance(node1, ast.AST):
        for k, v in list(vars(node1).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, zip(node1, node2)))
    else:
        return node1 == node2


def encoded_code_tokens_to_code(encoded_tokens, indent=' '):
    decoded_tokens = []
    for i in range(len(encoded_tokens)):
        token = encoded_tokens[i]
        token = token.replace('#TAB#', '\t').replace('#SPACE#', ' ')

        if token == '#INDENT#': decoded_tokens.append(indent)
        elif token == '#NEWLINE#': decoded_tokens.append('\n')
        else:
            token = token.replace('#NEWLINE#', '\n')
            decoded_tokens.append(token)
            decoded_tokens.append(' ')

    code = ''.join(decoded_tokens).strip()

    return code


def find_sub_sequence(sequence, query_seq):
    for i in range(len(sequence)):
        if sequence[i: len(query_seq) + i] == query_seq:
            return i, len(query_seq) + i

    raise IndexError


def replace_sequence(sequence, old_seq, new_seq):
    matched = False
    for i in range(len(sequence)):
        if sequence[i: i + len(old_seq)] == old_seq:
            matched = True
            sequence[i:i + len(old_seq)] = new_seq
    return matched


def text_to_json(rewritten_intent, snippet):
  failed = False
  intent_tokens = []
  example = {}
  try:
    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
    snippet = snippet
    canonical_snippet = canonicalize_code(snippet, slot_map)
    intent_tokens = nltk.word_tokenize(canonical_intent)
    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)
    snippet_reconstr = astor.to_source(ast.parse(snippet)).strip()
    decanonical_snippet_reconstr = astor.to_source(ast.parse(decanonical_snippet)).strip()
    encoded_reconstr_code = get_encoded_code_tokens(decanonical_snippet_reconstr)
    decoded_reconstr_code = encoded_code_tokens_to_code(encoded_reconstr_code)

    if not compare_ast(ast.parse(decoded_reconstr_code), ast.parse(snippet)):
        print(i)
        print('Original Snippet: %s' % snippet_reconstr)
        print('Tokenized Snippet: %s' % ' '.join(encoded_reconstr_code))
        print('decoded_reconstr_code: %s' % decoded_reconstr_code)
  except:
    print('failed')
    failed = True
  finally:
    example['slot_map'] = slot_map

  encoded_reconstr_code = get_encoded_code_tokens(canonical_snippet.strip())

  if not intent_tokens:
      intent_tokens = nltk.word_tokenize(rewritten_intent)

  example['intent_tokens'] = intent_tokens
  example['snippet_tokens'] = encoded_reconstr_code

  return example


def clean_output(input_text):
    input_text = input_text.replace('#NEWLINE#', '')
    input_text = input_text.replace('#SPACE#', '')
    input_text = input_text.replace('< ', '<')
    input_text = input_text.replace(' >', '>')
    input_text = input_text.replace('</ ', '</')
    input_text = input_text.strip()

    start_idx = input_text.find('<')
    input_text = input_text[start_idx:]
    return input_text


def read_examples_simple(js, nl2code):
    examples=[]
    code=' '.join(js['snippet_tokens']).replace('\n',' ')
    code=' '.join(code.strip().split())
    nl=' '.join(js['intent_tokens']).replace('\n','')
    nl=' '.join(nl.strip().split())

    if nl2code == False:
        # code -> natural language
        examples.append(
            Example(
                    idx = 0,
                    source=code,
                    target = '',
                    ) 
        )
    else:
        # natural language -> code
        examples.append(
            Example(
                    idx = 0,
                    source=nl,
                    target = '',
                    ) 
        )
              
    return examples


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
        
    return features


def set_seed(seed, n_gpu):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def initialize_model():
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

    # CPU only
    device = torch.device("cpu")
    n_gpu = 0

    # Set seed
    set_seed(42, n_gpu)

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained('roberta-base')
    tokenizer = tokenizer_class.from_pretrained('roberta-base',do_lower_case=True)

    #build model
    encoder = model_class.from_pretrained('roberta-base',config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=10,max_length=128,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    # load model weights with CPU
    model.load_state_dict(model_zoo.load_url('https://hijax-model.s3.amazonaws.com/pytorch_model.bin', map_location=torch.device('cpu')))

    model.to(device)

    if n_gpu > 1:
      # multi-gpu training
      model = torch.nn.DataParallel(model)

    return device, tokenizer, model


def inference_new(device, tokenizer, model, text):
    js = text_to_json(text, '')
    eval_examples = read_examples_simple(js, True)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, 256, 128, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long) 
    eval_data = TensorDataset(all_source_ids,all_source_mask)   

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)
    model.eval()

    p=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask = batch

        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)

            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)

    if len(p) == 1:
        return clean_output(p[0])
    else:
        return 'error'