"""Unified LLM client for the multi-model DGA feature-engineering experiment.

Supports OpenAI chat-completions and Anthropic messages endpoints, robust retries,
full request/response logging (JSONL), and resumable runs.
"""
import json, os, re, sys, time, uuid, hashlib, datetime
import urllib.request, urllib.error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE, 'llm_logs')
os.makedirs(LOGS_DIR, exist_ok=True)

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36')
OC_KEY = 'oc_sk_cf8949b499e3_4ImTTQmfbShCSm06IEEVc7yawz9YShNp'
SN_KEY = 'sk-g3a1NG8th9OppI9cb1gEfKKk8iDCE1Ae'

# ------------------------------------------------------------------
# Model registry — cost-efficient tier only (8 models across 6 vendors)
# ------------------------------------------------------------------
MODELS = {
    'mimo-v2.6-flash': {
        'developer': 'Xiaomi', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'mimo-v2.6-flash', 'label': 'MiMo-v2.6-Flash'},
    'qwen3.8-flash': {
        'developer': 'Alibaba', 'api': 'anthropic',
        'url': 'https://opencode.ai/zen/go/v1/messages',
        'key': OC_KEY, 'model': 'qwen3.8-flash', 'label': 'Qwen3.8-Flash'},
    'qwen3.7-flash': {
        'developer': 'Alibaba', 'api': 'anthropic',
        'url': 'https://opencode.ai/zen/go/v1/messages',
        'key': OC_KEY, 'model': 'qwen3.7-flash', 'label': 'Qwen3.7-Flash'},
    'deepseek-v4.1-flash': {
        'developer': 'DeepSeek', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'deepseek-v4.1-flash', 'label': 'DeepSeek-V4.1-Flash'},
    'deepseek-v4-flash': {
        'developer': 'DeepSeek', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'deepseek-v4-flash', 'label': 'DeepSeek-V4-Flash'},
    'glm-5.3-flash': {
        'developer': 'Zhipu', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'glm-5.3-flash', 'label': 'GLM-5.3-Flash'},
    'hy4-preview': {
        'developer': 'Tencent', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'hy4-preview', 'label': 'Hunyuan-HY4-Preview'},
    'hy3': {
        'developer': 'Tencent', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'hy3', 'label': 'Hunyuan-HY3'},
    'longcat-2.5-preview-free': {
        'developer': 'Meituan', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'longcat-2.5-preview-free', 'label': 'LongCat-2.5-Preview-Free'},
    'mimo-v2.5': {
        'developer': 'Xiaomi', 'api': 'openai',
        'url': 'https://opencode.ai/zen/go/v1/chat/completions',
        'key': OC_KEY, 'model': 'mimo-v2.5', 'label': 'MiMo-v2.5'},
}

MODEL_ORDER = ['mimo-v2.6-flash', 'mimo-v2.5', 'qwen3.8-flash', 'deepseek-v4.1-flash',
               'deepseek-v4-flash', 'glm-5.3-flash', 'hy3', 'longcat-2.5-preview-free']

LOG_PATH = os.path.join(LOGS_DIR, 'llm_calls.jsonl')
_sess = str(uuid.uuid4())


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def _call_openai(cfg, system, user, temperature, max_tokens, timeout=240):
    msgs = []
    if system:
        msgs.append({'role': 'system', 'content': system})
    msgs.append({'role': 'user', 'content': user})
    body = json.dumps({'model': cfg['model'], 'messages': msgs,
                       'temperature': temperature, 'max_tokens': max_tokens}).encode()
    req = urllib.request.Request(cfg['url'], data=body, headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + cfg['key'],
        'User-Agent': UA, 'Accept': 'application/json',
        'Origin': 'https://opencode.ai', 'Referer': 'https://opencode.ai/',
        'x-opencode-session': _sess})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode('utf-8', 'replace'))
    msg = resp['choices'][0]['message']
    text = msg.get('content') or ''
    if not text and msg.get('reasoning_content'):
        text = ''
    return text, resp.get('usage', {}), resp.get('model', cfg['model'])


def _call_anthropic(cfg, system, user, temperature, max_tokens, timeout=240):
    payload = {'model': cfg['model'], 'max_tokens': max_tokens, 'temperature': temperature,
               'messages': [{'role': 'user', 'content': user}]}
    if system:
        payload['system'] = system
    body = json.dumps(payload).encode()
    req = urllib.request.Request(cfg['url'], data=body, headers={
        'Content-Type': 'application/json', 'x-api-key': cfg['key'],
        'anthropic-version': '2023-06-01', 'User-Agent': UA, 'Accept': 'application/json',
        'Origin': 'https://opencode.ai', 'Referer': 'https://opencode.ai/',
        'x-opencode-session': _sess})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode('utf-8', 'replace'))
    blocks = resp.get('content', [])
    text = ''.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
    u = resp.get('usage', {})
    usage = {'prompt_tokens': u.get('input_tokens'), 'completion_tokens': u.get('output_tokens'),
             'total_tokens': (u.get('input_tokens') or 0) + (u.get('output_tokens') or 0)}
    return text, usage, resp.get('model', cfg['model'])


def call_model(model_key, system, user, stage, rep, temperature, max_tokens,
               max_attempts=5, log=True, extra=None):
    """Call one model with retries; returns dict with text/usage/latency/status."""
    cfg = MODELS[model_key]
    attempt_log = []
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            if cfg['api'] == 'openai':
                text, usage, served = _call_openai(cfg, system, user, temperature, max_tokens)
            else:
                text, usage, served = _call_anthropic(cfg, system, user, temperature, max_tokens)
            dt = time.time() - t0
            rec = {'ts': _now(), 'model_key': model_key, 'model_id': cfg['model'],
                   'developer': cfg['developer'], 'stage': stage, 'rep': rep,
                   'temperature': temperature, 'max_tokens': max_tokens,
                   'attempt': attempt, 'status': 'ok', 'latency_s': round(dt, 2),
                   'served_model': served, 'usage': usage,
                   'system_prompt': system, 'user_prompt': user, 'response': text,
                   'extra': extra or {}}
            if log:
                with open(LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            return rec
        except urllib.error.HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', 'replace')[:400]
            except Exception:
                pass
            attempt_log.append(f'attempt {attempt}: HTTP {e.code} {body}')
            if e.code in (400, 401, 403, 404, 422) and attempt >= 2:
                break
            time.sleep(min(4 * attempt, 20))
        except Exception as e:
            attempt_log.append(f'attempt {attempt}: {type(e).__name__} {str(e)[:200]}')
            time.sleep(min(4 * attempt, 20))
    rec = {'ts': _now(), 'model_key': model_key, 'model_id': cfg['model'],
           'developer': cfg['developer'], 'stage': stage, 'rep': rep,
           'temperature': temperature, 'max_tokens': max_tokens,
           'status': 'failed', 'attempt_log': attempt_log,
           'system_prompt': system, 'user_prompt': user, 'response': '', 'usage': {},
           'extra': extra or {}}
    if log:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


_JSON_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


def extract_json(text):
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    candidates = []
    for m in _JSON_FENCE.finditer(text):
        candidates.append(m.group(1))
    i, j = text.find('{'), text.rfind('}')
    if i != -1 and j > i:
        candidates.append(text[i:j + 1])
    for cand in candidates:
        for c in (cand, re.sub(r',(\s*[}\]])', r'\1', cand)):  # repair trailing commas
            try:
                return json.loads(c)
            except Exception:
                continue
    return None
